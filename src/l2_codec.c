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
  const __m256i sub8  = _mm256_set1_epi8((char)8);
  #pragma omp parallel for schedule(static)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i*(size_t)nb*18;
    __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps(), acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
    i32 b=0;
    const i32 nb2 = nb & ~1; /* pares: 2 bloques = 64 valores por iteracion */
    for(; b<nb2; b+=2){
      /* layout por bloque Q4_0: [scale f16][16 nibbles] = 18 B; 2 bloques seguidos */
      __m256 vs0=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m128i lo=_mm_loadu_si128((const __m128i*)row); row+=16; /* bloque b */
      __m256 vs1=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m128i hi=_mm_loadu_si128((const __m128i*)row); row+=16; /* bloque b+1 */
      __m128i l0=_mm_and_si128(lo,mask0F), h0=_mm_and_si128(_mm_srli_epi16(lo,4),mask0F);
      __m128i l1=_mm_and_si128(hi,mask0F), h1=_mm_and_si128(_mm_srli_epi16(hi,4),mask0F);
      __m256i q0=_mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(l0),h0,1),sub8);
      __m256i q1=_mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(l1),h1,1),sub8);
      __m128i c0=_mm256_castsi256_si128(q0), c1=_mm256_extracti128_si256(q0,1);
      __m128i c2=_mm256_castsi256_si128(q1), c3=_mm256_extracti128_si256(q1,1);
      const f32 *xb = x + (size_t)b*32;
      acc0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c0)),vs0),_mm256_loadu_ps(xb)  ,acc0);
      acc1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c0,8))),vs0),_mm256_loadu_ps(xb+8) ,acc1);
      acc2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c1)),vs0),_mm256_loadu_ps(xb+16),acc2);
      acc3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c1,8))),vs0),_mm256_loadu_ps(xb+24),acc3);
      const f32 *xb2 = xb+32;
      acc0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c2)),vs1),_mm256_loadu_ps(xb2)  ,acc0);
      acc1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c2,8))),vs1),_mm256_loadu_ps(xb2+8) ,acc1);
      acc2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c3)),vs1),_mm256_loadu_ps(xb2+16),acc2);
      acc3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c3,8))),vs1),_mm256_loadu_ps(xb2+24),acc3);
    }
    for(; b<nb; b++){
      f32 s = half_to_float(*(const u16*)row); row+=2;
      __m256 vs = _mm256_set1_ps(s);
      __m128i qs = _mm_loadu_si128((const __m128i*)row); row+=16;
      __m128i ql = _mm_and_si128(qs, mask0F);
      __m128i qh = _mm_and_si128(_mm_srli_epi16(qs,4), mask0F);
      __m256i v = _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(ql), qh, 1), sub8);
      __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
      const f32 *xb = x + (size_t)b*32;
      acc0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs),_mm256_loadu_ps(xb),acc0);
      acc1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs),_mm256_loadu_ps(xb+8),acc1);
      acc2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs),_mm256_loadu_ps(xb+16),acc2);
      acc3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs),_mm256_loadu_ps(xb+24),acc3);
    }
    float t[8];
    _mm256_storeu_ps(t,acc0); f32 s0=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,acc1); f32 s1=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,acc2); f32 s2=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,acc3); f32 s3=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    out[i]=s0+s1+s2+s3;
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
