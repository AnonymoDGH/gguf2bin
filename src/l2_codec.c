/* L2 — dequant + matmul fused Q4_0/Q8_0 — FIXED con AVX2 Q4_0 */
#include "g2b.h"
#include <string.h>
#include <stdlib.h>
u64 ggml_block_size(u32 type){
  switch(type){
    case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1:
    case T_Q8_0: case T_Q8_1: return 32;
    default: return 1;
  }
}
u64 ggml_type_bytes(u32 type){
  switch(type){
    case T_F32:  return 4;
    case T_F16:  return 2;
    case T_Q4_0: return 18;
    case T_Q4_1: return 20;
    case T_Q5_0: return 22;
    case T_Q5_1: return 24;
    case T_Q8_0: return 34;
    case T_Q8_1: return 36;
    default:     return 4;
  }
}
u64 ggml_type_size(u32 type, u64 ne){ return (ne/ggml_block_size(type))*ggml_type_bytes(type); }
f32 half_to_float(u16 h){
  u32 sign = (u32)(h>>15)&1u;
  u32 exp  = (u32)(h>>10)&0x1fu;
  u32 mant = (u32)h&0x3ffu;
  union { u32 u; f32 f; } r;
  if(exp==0){
    if(!mant){ r.u = sign<<31; return r.f; }
    /* Proper subnormal: 2^-14 * (mant/1024) */
    float f = (float)mant / 1024.f;
    f *= 1.f / 16384.f; /* 2^-14 */
    r.f = sign ? -f : f;
    return r.f;
  }
  if(exp==31){
    r.u = (sign<<31) | (0xffu<<23) | (mant<<13);
    return r.f;
  }
  r.u = (sign<<31) | ((exp+112u)<<23) | (mant<<13);
  return r.f;
}
static void deq_q4_0(u8 *b, f32 *o, u64 n){
  for(u64 i=0;i<n;i+=32){
    f32 s=half_to_float(*(u16*)b); b+=2;
    for(int j=0;j<16;j++){
      o[i+j]    = s*((b[j]&0x0f)-8);
      o[i+j+16] = s*((b[j]>>4)-8);
    }
    b+=16;
  }
}
static void deq_q4_1(u8 *b, f32 *o, u64 n){
  for(u64 i=0;i<n;i+=32){
    f32 s=half_to_float(*(u16*)b); b+=2;
    f32 m=half_to_float(*(u16*)b); b+=2;
    for(int j=0;j<16;j++){
      o[i+j]    = s*(b[j]&0x0f)+m;
      o[i+j+16] = s*(b[j]>>4)+m;
    }
    b+=16;
  }
}
static void deq_q8_0(u8 *b, f32 *o, u64 n){
  for(u64 i=0;i<n;i+=32){
    f32 s=half_to_float(*(u16*)b); b+=2;
    for(int j=0;j<32;j++) o[i+j]=s*(i8)b[j];
    b+=32;
  }
}
void gguf_dequant(u32 type, u8 *src, f32 *out, u64 ne){
  switch(type){
    case T_F32:  memcpy(out,src,ne*4); break;
    case T_F16:  for(u64 i=0;i<ne;i++) out[i]=half_to_float(((u16*)src)[i]); break;
    case T_Q4_0: deq_q4_0(src,out,ne); break;
    case T_Q4_1: deq_q4_1(src,out,ne); break;
    case T_Q8_0: deq_q8_0(src,out,ne); break;
    default: fprintf(stderr,"codec: tipo %u no soportado\n",type); memset(out,0,ne*4);
  }
}
#if defined(__AVX2__) && !defined(DISABLE_AVX2)
#include <immintrin.h>
void matmul_q8_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  #pragma omp parallel for schedule(static)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i*(size_t)nb*34;
    __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
    for(i32 b=0;b<nb;b++){
      f32 s = half_to_float(*(const u16*)row); row+=2;
      __m256 vs = _mm256_set1_ps(s);
      __m256i q8 = _mm256_loadu_si256((const __m256i*)row); row+=32;
      __m128i lo = _mm256_castsi256_si128(q8);
      __m128i hi = _mm256_extracti128_si256(q8,1);
      __m256 q0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs);
      __m256 q1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs);
      __m256 q2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs);
      __m256 q3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs);
      const f32 *xb = x + (size_t)b*32;
      a0=_mm256_fmadd_ps(q0,_mm256_loadu_ps(xb+0 ),a0);
      a1=_mm256_fmadd_ps(q1,_mm256_loadu_ps(xb+8 ),a1);
      a2=_mm256_fmadd_ps(q2,_mm256_loadu_ps(xb+16),a2);
      a3=_mm256_fmadd_ps(q3,_mm256_loadu_ps(xb+24),a3);
    }
    float t[8];
    _mm256_storeu_ps(t,a0); f32 s0=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,a1); f32 s1=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,a2); f32 s2=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,a3); f32 s3=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    out[i]=s0+s1+s2+s3;
  }
}
void matmul_q4_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  const __m128i mask0F = _mm_set1_epi8(0x0F);
  const __m128i maskF0 = _mm_set1_epi8((char)0xF0);
  const __m128i eight = _mm_set1_epi8(8);
  #pragma omp parallel for schedule(static)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i*(size_t)nb*18;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for(i32 b=0;b<nb;b++){
      f32 s = half_to_float(*(const u16*)row); row+=2;
      __m256 vs = _mm256_set1_ps(s);
      __m128i qs = _mm_loadu_si128((const __m128i*)row); row+=16;
      // low nibbles
      __m128i low = _mm_and_si128(qs, mask0F);
      // high nibbles: and F0, srli 4
      __m128i high = _mm_and_si128(qs, maskF0);
      high = _mm_srli_epi16(high,4);
      high = _mm_and_si128(high, mask0F);
      // low -8
      __m128i low_sub = _mm_sub_epi8(low, eight);
      __m128i high_sub = _mm_sub_epi8(high, eight);
      // convert 8 low bytes to 8 int32 -> float
      // first 8 bytes: low0, high0
      // low part: low_sub lower 8 bytes
      __m128i low_low = low_sub; // contains 16 bytes, we need first 8
      __m128i high_low = high_sub;
      // second 8 bytes for second half of block? Actually block has 16 bytes qs -> 32 q's
      // We already have low and high each 16 bytes, but we process in 8-byte chunks
      // Let's process 0..7 and 8..15 separately

      // 0..7 low/high
      __m128i low0 = _mm_loadl_epi64((__m128i*)&low_sub); // low 8 bytes
      __m128i high0 = _mm_loadl_epi64((__m128i*)&high_sub);
      // Actually need to load from low_sub/high_sub correctly
      // To avoid complexity, we will do scalar for the 8-byte groups but using AVX for conversion

      // Convert low 8 bytes to float
      __m256i low0_32 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&low_sub));
      __m256i high0_32 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&high_sub));
      __m256 low0_f = _mm256_cvtepi32_ps(low0_32);
      __m256 high0_f = _mm256_cvtepi32_ps(high0_32);
      low0_f = _mm256_mul_ps(low0_f, vs);
      high0_f = _mm256_mul_ps(high0_f, vs);
      const f32 *xb = x + b*32;
      __m256 x0 = _mm256_loadu_ps(xb+0);
      __m256 x1 = _mm256_loadu_ps(xb+16);
      acc0 = _mm256_fmadd_ps(low0_f, x0, acc0);
      acc1 = _mm256_fmadd_ps(high0_f, x1, acc1);

      // next 8 bytes (8..15)
      __m128i low1 = _mm_bsrli_si128(low_sub,8);
      __m128i high1 = _mm_bsrli_si128(high_sub,8);
      __m256i low1_32 = _mm256_cvtepi8_epi32(low1);
      __m256i high1_32 = _mm256_cvtepi8_epi32(high1);
      __m256 low1_f = _mm256_cvtepi32_ps(low1_32);
      __m256 high1_f = _mm256_cvtepi32_ps(high1_32);
      low1_f = _mm256_mul_ps(low1_f, vs);
      high1_f = _mm256_mul_ps(high1_f, vs);
      __m256 x2 = _mm256_loadu_ps(xb+8);
      __m256 x3 = _mm256_loadu_ps(xb+24);
      acc2 = _mm256_fmadd_ps(low1_f, x2, acc2);
      acc3 = _mm256_fmadd_ps(high1_f, x3, acc3);
    }
    // sum accs
    __m256 sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    // horizontal sum
    __m128 low128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum,1));
    low128 = _mm_add_ps(low128, _mm_movehl_ps(low128, low128));
    low128 = _mm_add_ss(low128, _mm_shuffle_ps(low128, low128, 1));
    out[i] = _mm_cvtss_f32(low128);
  }
}
#else
void matmul_q8_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  #pragma omp parallel for schedule(static)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i*(size_t)nb*34;
    f32 sum=0.f;
    for(i32 b=0;b<nb;b++){
      f32 s = half_to_float(*(const u16*)row); row+=2;
      const f32 *xb = x + (size_t)b*32;
      const i8 *qs = (const i8*)row; row+=32;
      for(int j=0;j<32;j++) sum += s * (f32)qs[j] * xb[j];
    }
    out[i]=sum;
  }
}
void matmul_q4_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  #pragma omp parallel for schedule(static)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i * (size_t)nb * 18;
    f32 sum=0.f;
    for(i32 b=0;b<nb;b++){
      f32 s = half_to_float(*(const u16*)row); row+=2;
      const f32 *xb = x + b*32;
      for(int j=0;j<16;j++){
        int q0 = (row[j]&0x0f)-8;
        int q1 = (row[j]>>4)-8;
        sum += s * ((f32)q0 * xb[j] + (f32)q1 * xb[j+16]);
      }
      row+=16;
    }
    out[i]=sum;
  }
}
#endif
static u64 row_stride(u32 type, i32 n){ return (n/ggml_block_size(type))*ggml_type_bytes(type); }
void matmul_q(f32 *out, f32 *x, u8 *w, u32 type, i32 n, i32 d, f32 *row){
  if(!out || !x || !w || n<=0 || d<=0){
    if(out && d>0) memset(out, 0, (size_t)d * sizeof(f32));
    return;
  }
  if(type==T_Q4_0){ matmul_q4_0(out,x,w,n,d); return; }
  if(type==T_Q8_0){ matmul_q8_0(out,x,w,n,d); return; }
  if(type==T_F32){ matmul(out,x,(f32*)w,n,d); return; }
  u64 rs=row_stride(type,n);
  /* Private per-row scratch: shared `row` would race under OpenMP. */
  #pragma omp parallel
  {
    f32 *tmp = (f32*)malloc((size_t)n * sizeof(f32));
    if(!tmp){
      #pragma omp for schedule(static)
      for(i32 i=0;i<d;i++) out[i]=0.f;
    } else {
      #pragma omp for schedule(static)
      for(i32 i=0;i<d;i++){
        gguf_dequant(type, w+(size_t)i*rs, tmp, (u64)n);
        f32 s=0; for(i32 j=0;j<n;j++) s+=tmp[j]*x[j];
        out[i]=s;
      }
      free(tmp);
    }
  }
  (void)row;
}
