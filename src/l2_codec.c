/* L2 — dequant + matmul fused Q4_0/Q8_0 — FIXED con AVX2 Q4_0 */
#include "g2b.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
u64 ggml_block_size(u32 type){
  switch(type){
    case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1:
    case T_Q8_0: case T_Q8_1: return 32;
    case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K:
    case T_Q6_K: case T_Q8_K: return 256;
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
    case T_Q2_K: return 84;
    case T_Q3_K: return 110;
    case T_Q4_K: return 144;
    case T_Q5_K: return 176;
    case T_Q6_K: return 210;
    case T_Q8_K: return 292;
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
/* ── K-quant scalar dequant ── */
#define QK_K 256
static void deq_q2_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    f32 d = half_to_float(*(const u16*)(src + (size_t)b*84 + 80));
    f32 dmin = half_to_float(*(const u16*)(src + (size_t)b*84 + 82));
    const u8* scales = src + (size_t)b*84;
    const u8* q = src + (size_t)b*84 + 16;
    i32 is = 0;
    for(i32 n=0;n<QK_K;n+=128){
      i32 shift = 0;
      for(i32 j=0;j<4;j++){
        u8 sc = scales[is++];
        f32 dl = d * (f32)(sc & 0xF), ml = dmin * (f32)(sc >> 4);
        for(i32 l=0;l<16;l++) dst[b*QK_K+n+j*32+l] = dl*(f32)(i8)((q[j*32+l]>>shift)&3) - ml;
        sc = scales[is++];
        dl = d * (f32)(sc & 0xF), ml = dmin * (f32)(sc >> 4);
        for(i32 l=0;l<16;l++) dst[b*QK_K+n+j*32+16+l] = dl*(f32)(i8)((q[j*32+16+l]>>shift)&3) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}
static void deq_q3_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    const u8* hmask = src + (size_t)b*110;
    const u8* qs = hmask + 32;
    const u8* scales = qs + 64;
    f32 d = half_to_float(*(const u16*)(scales + 12));
    for(i32 n=0;n<QK_K;n+=64){
      u32 m = *(const u32*)(hmask + (n/64)*4);
      i32 shift = 0;
      for(i32 j=0;j<4;j++){
        u8 sc = scales[(n/64)*2 + j/2];
        f32 dl = (j&1)?d*(f32)(sc&0xF)*16.f:d*(f32)(sc>>4);
        for(i32 l=0;l<8;l++){
          i32 q = ((qs[j*8+l+(n/64)*32]>>shift)&3);
          if(m & (1u<<(j*8+l))) q -= 4;
          dst[b*QK_K+n+j*8+l] = dl * (f32)q;
        }
        shift += 2;
      }
    }
  }
}
static void deq_q4_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    f32 d = half_to_float(*(const u16*)(src + (size_t)b*144));
    f32 dmin = half_to_float(*(const u16*)(src + (size_t)b*144 + 2));
    const u8* scales = src + (size_t)b*144 + 4;
    const u8* qs = src + (size_t)b*144 + 16;
    for(i32 n=0;n<QK_K;n+=64){
      u8 sc = scales[n/64*2], sc2 = scales[n/64*2+1];
      f32 dl0 = d * (f32)(sc & 0xF), ml0 = dmin * (f32)(sc >> 4);
      f32 dl1 = d * (f32)(sc2 & 0xF), ml1 = dmin * (f32)(sc2 >> 4);
      for(i32 l=0;l<32;l++){
        i32 q = qs[n/2+l] & 0xF, idx = n+l;
        dst[b*QK_K+idx] = (idx<32?dl0:dl1)*(f32)(q-8) - (idx<32?ml0:ml1);
      }
    }
  }
}
static void deq_q5_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    f32 d = half_to_float(*(const u16*)(src + (size_t)b*176));
    f32 dmin = half_to_float(*(const u16*)(src + (size_t)b*176 + 2));
    const u8* scales = src + (size_t)b*176 + 4;
    const u8* qh = src + (size_t)b*176 + 16;
    const u8* qs = qh + 32;
    for(i32 n=0;n<QK_K;n+=64){
      u8 sc = scales[n/64*2], sc2 = scales[n/64*2+1];
      f32 dl0 = d * (f32)(sc & 0xF), ml0 = dmin * (f32)(sc >> 4);
      f32 dl1 = d * (f32)(sc2 & 0xF), ml1 = dmin * (f32)(sc2 >> 4);
      for(i32 l=0;l<32;l++){
        i32 q = (qs[n/2+l]&0xF) | (((qh[n/32+(l>=16?1:0)]>>((l&15)/8))&1)<<4);
        dst[b*QK_K+n+l] = (l<16?dl0:dl1)*(f32)(q-16) - (l<16?ml0:ml1);
      }
    }
  }
}
static void deq_q6_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    const u8* ql = src + (size_t)b*210;
    const u8* qh = ql + 128;
    const i8* scales = (const i8*)(qh + 64);
    f32 d = half_to_float(*(const u16*)(scales + 8));
    for(i32 n=0;n<QK_K;n+=128){
      for(i32 l=0;l<32;l++){
        i32 j = n/2+l;
        i32 q = ((ql[j] & 0xF) | ((qh[n/64+l/16] >> (2*(l%16)/8)) & 0x30));
        dst[b*QK_K+n+l]    = d * (f32)scales[n/128*4+l/16] * (f32)(q-32);
        q = ((ql[j] >> 4) | ((qh[n/64+l/16] >> (2*(l%16)/8)) & 0x30));
        dst[b*QK_K+n+l+32] = d * (f32)scales[n/128*4+l/16+2] * (f32)(q-32);
      }
    }
  }
}
void gguf_dequant(u32 type, u8 *src, f32 *out, u64 ne){
  switch(type){
    case T_F32:  memcpy(out,src,ne*4); break;
    case T_F16:  for(u64 i=0;i<ne;i++) out[i]=half_to_float(((u16*)src)[i]); break;
    case T_Q4_0: deq_q4_0(src,out,ne); break;
    case T_Q4_1: deq_q4_1(src,out,ne); break;
    case T_Q8_0: deq_q8_0(src,out,ne); break;
    case T_Q2_K: deq_q2_K(src,out,ne); break;
    case T_Q3_K: deq_q3_K(src,out,ne); break;
    case T_Q4_K: deq_q4_K(src,out,ne); break;
    case T_Q5_K: deq_q5_K(src,out,ne); break;
    case T_Q6_K: deq_q6_K(src,out,ne); break;
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
    i32 b=0;
    const i32 nb2 = nb & ~1; /* 2 bloques/iter: 68 bytes → 64 valores con 2 escalas */
    for(; b<nb2; b+=2){
      __m256 vs0=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m256i qw0=_mm256_loadu_si256((const __m256i*)row); row+=32;
      __m256 vs1=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m256i qw1=_mm256_loadu_si256((const __m256i*)row); row+=32;
      __m128i l0=_mm256_castsi256_si128(qw0), h0=_mm256_extracti128_si256(qw0,1);
      __m128i l1=_mm256_castsi256_si128(qw1), h1=_mm256_extracti128_si256(qw1,1);
      const f32 *xb = x + (size_t)b*32;
      a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l0)),vs0),_mm256_loadu_ps(xb)  ,a0);
      a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l0,8))),vs0),_mm256_loadu_ps(xb+8) ,a1);
      a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h0)),vs0),_mm256_loadu_ps(xb+16),a2);
      a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h0,8))),vs0),_mm256_loadu_ps(xb+24),a3);
      const f32 *xb2 = xb+32;
      a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l1)),vs1),_mm256_loadu_ps(xb2)  ,a0);
      a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l1,8))),vs1),_mm256_loadu_ps(xb2+8) ,a1);
      a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h1)),vs1),_mm256_loadu_ps(xb2+16),a2);
      a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h1,8))),vs1),_mm256_loadu_ps(xb2+24),a3);
    }
    for(; b<nb; b++){
      const u8 *rb = row + (size_t)(b-nb2)*34u; /* sigue despues de los 2-bloques */
      f32 s = half_to_float(*(const u16*)rb); rb+=2;
      __m256 vs = _mm256_set1_ps(s);
      __m256i q8 = _mm256_loadu_si256((const __m256i*)rb); rb+=32;
      __m128i lo = _mm256_castsi256_si128(q8), hi = _mm256_extracti128_si256(q8,1);
      const f32 *xb = x + (size_t)b*32;
      a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs),_mm256_loadu_ps(xb),a0);
      a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs),_mm256_loadu_ps(xb+8),a1);
      a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs),_mm256_loadu_ps(xb+16),a2);
      a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs),_mm256_loadu_ps(xb+24),a3);
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
    __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
    __m256 a4=_mm256_setzero_ps(), a5=_mm256_setzero_ps(), a6=_mm256_setzero_ps(), a7=_mm256_setzero_ps();
    i32 b=0;
    const i32 nb2 = nb & ~1;
    for(; b<nb2; b+=2){
      _mm_prefetch(row + 9*18, _MM_HINT_T0);
      __m256 vs0=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m128i q0=_mm_loadu_si128((const __m128i*)row); row+=16;
      __m256 vs1=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m128i q1=_mm_loadu_si128((const __m128i*)row); row+=16;
      __m128i l0=_mm_and_si128(q0,mask0F), h0=_mm_and_si128(_mm_srli_epi16(q0,4),mask0F);
      __m128i l1=_mm_and_si128(q1,mask0F), h1=_mm_and_si128(_mm_srli_epi16(q1,4),mask0F);
      __m256i v0=_mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(l0),h0,1),sub8);
      __m256i v1=_mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(l1),h1,1),sub8);
      __m128i c0=_mm256_castsi256_si128(v0), c1=_mm256_extracti128_si256(v0,1);
      __m128i c2=_mm256_castsi256_si128(v1), c3=_mm256_extracti128_si256(v1,1);
      const f32 *xb = x + (size_t)b*32;
      a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c0)),vs0),_mm256_loadu_ps(xb),a0);
      a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c0,8))),vs0),_mm256_loadu_ps(xb+8),a1);
      a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c1)),vs0),_mm256_loadu_ps(xb+16),a2);
      a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c1,8))),vs0),_mm256_loadu_ps(xb+24),a3);
      a4=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c2)),vs1),_mm256_loadu_ps(xb+32),a4);
      a5=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c2,8))),vs1),_mm256_loadu_ps(xb+40),a5);
      a6=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c3)),vs1),_mm256_loadu_ps(xb+48),a6);
      a7=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c3,8))),vs1),_mm256_loadu_ps(xb+56),a7);
    }
    a0=_mm256_add_ps(a0,a4); a1=_mm256_add_ps(a1,a5); a2=_mm256_add_ps(a2,a6); a3=_mm256_add_ps(a3,a7);
    for(; b<nb; b++){
      f32 s = half_to_float(*(const u16*)row); row+=2;
      __m256 vs = _mm256_set1_ps(s);
      __m128i qs = _mm_loadu_si128((const __m128i*)row); row+=16;
      __m128i ql = _mm_and_si128(qs, mask0F);
      __m128i qh = _mm_and_si128(_mm_srli_epi16(qs,4), mask0F);
      __m256i v = _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(ql), qh, 1), sub8);
      __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
      const f32 *xb = x + (size_t)b*32;
      a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs),_mm256_loadu_ps(xb),a0);
      a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs),_mm256_loadu_ps(xb+8),a1);
      a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs),_mm256_loadu_ps(xb+16),a2);
      a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs),_mm256_loadu_ps(xb+24),a3);
    }
    float t[8];
    _mm256_storeu_ps(t,a0); f32 s0=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,a1); f32 s1=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,a2); f32 s2=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    _mm256_storeu_ps(t,a3); f32 s3=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
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
  /* Fallback para tipos no fusionados (Q4_1, etc.): per-thread alloc, no per-row */
  u64 rs=row_stride(type,n);
  #pragma omp parallel
  {
    f32 *tmp = (f32*)malloc((size_t)n * sizeof(f32));
    if(tmp){
      #pragma omp for schedule(static)
      for(i32 i=0;i<d;i++){
        gguf_dequant(type, w+(size_t)i*rs, tmp, (u64)n);
        f32 s=0; for(i32 j=0;j<n;j++) s+=tmp[j]*x[j];
        out[i]=s;
      }
      free(tmp);
    } else {
      #pragma omp for schedule(static)
      for(i32 i=0;i<d;i++) out[i]=0.f;
    }
  }
  (void)row;
}
