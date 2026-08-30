/* L2 — dequant + matmul fused Q4_0/Q8_0 — FIXED con AVX2 Q4_0 */
#include "g2b.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#if defined(__F16C__)
#include <immintrin.h>
#endif
void matmul_q4_0s(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_q4_0s_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);
/* Threshold: skip OpenMP for small matmuls where fork/join overhead dominates. */
#define OMP_MM_MIN 131072 /* ~512*256 elements — below this, single-thread wins */
#if defined(_OPENMP)
#include <omp.h>
static f32 *g_fallback_buf[64] = {0};
static size_t g_fallback_cap[64] = {0};
static inline f32* fallback_buf(size_t n){
  int tid = omp_get_thread_num();
  if(tid<0||tid>=64) tid=0;
  if(g_fallback_cap[tid] < n){
    free(g_fallback_buf[tid]);
    g_fallback_buf[tid]=(f32*)malloc(n*sizeof(f32));
    g_fallback_cap[tid]= g_fallback_buf[tid] ? n : 0;
  }
  return g_fallback_buf[tid];
}
#else
static f32 *g_fallback_buf1=NULL; static size_t g_fallback_cap1=0;
static inline f32* fallback_buf(size_t n){
  if(g_fallback_cap1 < n){ free(g_fallback_buf1); g_fallback_buf1=(f32*)malloc(n*sizeof(f32)); g_fallback_cap1=g_fallback_buf1?n:0; }
  return g_fallback_buf1;
}
#endif
u64 ggml_block_size(u32 type){
  switch(type){
    case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1:
    case T_Q8_0: case T_Q8_1: return 32;
    case T_Q4_0S: case T_Q4_0S_PSY: return 256;
    case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K:
    case T_Q6_K: case T_Q8_K: return 256;
    /* i-quants: superbloques de 256 (IQ4_NL usa 32) */
    case T_IQ2_XXS: case T_IQ2_XS: case T_IQ2_S: case T_IQ3_XXS:
    case T_IQ3_S: case T_IQ1_S: case T_IQ1_M: case T_IQ4_XS: return 256;
    case T_IQ4_NL: return 32;
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
    case T_Q4_0S: return 130;
    case T_Q4_0S_PSY: return 132;
    case T_Q2_K: return 84;
    case T_Q3_K: return 110;
    case T_Q4_K: return 144;
    case T_Q5_K: return 176;
    case T_Q6_K: return 210;
    case T_Q8_K: return 292;
    /* i-quants (QK_K=256); ojo: qs de IQ2 es uint16_t[QK_K/8] = 64 bytes */
    case T_IQ2_XXS: return 66;   /* 2 + QK_K/8*sizeof(u16) */
    case T_IQ2_XS:  return 74;   /* 2 + QK_K/8*sizeof(u16) + QK_K/32 */
    case T_IQ2_S:   return 82;   /* 2 + QK_K/4 + QK_K/16   */
    case T_IQ3_XXS: return 98;   /* 2 + 3*QK_K/8   */
    case T_IQ3_S:   return 110;  /* 2 + 13*QK_K/32 + QK_K/64 */
    case T_IQ1_S:   return 50;   /* 2 + QK_K/8 + QK_K/16 (qh = 8xu16) */
    case T_IQ1_M:   return 56;   /* QK_K/8 + QK_K/16 + QK_K/32 */
    case T_IQ4_XS:  return 136;  /* 2 + 2 + QK_K/64 + QK_K/2 */
    case T_IQ4_NL:  return 18;   /* 2 + 16 (bloques de 32) */
    default:     return 4;
  }
}
u64 ggml_type_size(u32 type, u64 ne){ return (ne/ggml_block_size(type))*ggml_type_bytes(type); }
f32 half_to_float(u16 h){
#if defined(__F16C__)
  return _cvtsh_ss((short)h);
#else
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
#endif
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
static void deq_q5_0(u8 *b, f32 *o, u64 n){
  for(u64 i=0;i<n;i+=32){
    f32 s=half_to_float(*(u16*)b); b+=2;
    u32 qh; memcpy(&qh,b,4); b+=4;
    for(int j=0;j<16;j++){
      int v0=((b[j]&0xF)|(((qh>>j)&1)<<4))-16;
      int v1=((b[j]>>4)|(((qh>>(j+16))&1)<<4))-16;
      o[i+j]=s*(f32)v0; o[i+j+16]=s*(f32)v1;
    }
    b+=16;
  }
}
/* ── K-quant scalar dequant (interleave de escalas según referencia ggml) ── */
#define QK_K 256

static inline void get_scale_min_k4(int j, const u8 *q, u8 *d, u8 *m){
  if(j<4){ *d=(u8)(q[j]&63); *m=(u8)(q[j+4]&63); }
  else{
    *d=(u8)((q[j+4]&0xF)|((q[j-4]>>6)<<4));
    *m=(u8)((q[j+4]>>4)|((q[j-0]>>6)<<4));
  }
}
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
        for(i32 l=0;l<16;l++) dst[b*QK_K+n+j*32+l] = dl*(f32)((q[l]>>shift)&3) - ml;
        sc = scales[is++];
        dl = d * (f32)(sc & 0xF), ml = dmin * (f32)(sc >> 4);
        for(i32 l=0;l<16;l++) dst[b*QK_K+n+j*32+16+l] = dl*(f32)((q[l+16]>>shift)&3) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}
static void deq_q3_K(const u8 *src, f32 *dst, u64 ne){
  /* referencia ggml: escalas 6-bit con bias -32, salida secuencial, hmask con
     bit marchando por sub-bloque de 16 */
  const u32 kmask1=0x03030303u, kmask2=0x0f0f0f0fu;
  u64 nb = ne / QK_K;
  u32 aux[4];
  for(u64 b=0;b<nb;b++){
    const u8* base = src + (size_t)b*110;
    const u8* hm = base;
    const u8* q = base + 32;
    f32 d_all = half_to_float(*(const u16*)(base + 108));
    memcpy(aux, base + 96, 12);
    u32 tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const i8* scales = (const i8*)aux;
    f32 *y = dst + b*QK_K;
    i32 is = 0;
    u8 m = 1;
    for(i32 n=0;n<QK_K;n+=128){
      i32 shift = 0;
      for(i32 j=0;j<4;j++){
        f32 dl = d_all * (f32)(scales[is++] - 32);
        for(i32 l=0;l<16;l++) *y++ = dl*(f32)((i8)((q[l   ]>>shift)&3) - ((hm[l   ]&m)?0:4));
        dl = d_all * (f32)(scales[is++] - 32);
        for(i32 l=0;l<16;l++) *y++ = dl*(f32)((i8)((q[l+16]>>shift)&3) - ((hm[l+16]&m)?0:4));
        shift += 2;
        m = (u8)(m<<1);
      }
      q += 32;
    }
  }
}
static void deq_q4_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    const u8* base = src + (size_t)b*144;
    f32 d    = half_to_float(*(const u16*)base);
    f32 dmin = half_to_float(*(const u16*)(base+2));
    const u8* scales = base + 4;
    const u8* q = base + 16;
    i32 is = 0;
    for(i32 n=0;n<QK_K;n+=64){
      u8 sc4, m4;
      get_scale_min_k4(is+0, scales, &sc4, &m4);
      f32 d1 = d*sc4, m1 = dmin*m4;
      get_scale_min_k4(is+1, scales, &sc4, &m4);
      f32 d2 = d*sc4, m2 = dmin*m4;
      for(i32 l=0;l<32;l++) dst[(size_t)b*QK_K+n+l     ] = d1*(f32)(q[l]&0xF) - m1;
      for(i32 l=0;l<32;l++) dst[(size_t)b*QK_K+n+l+32] = d2*(f32)(q[l]>>4) - m2;
      q += 32;
      is += 2;
    }
  }
}
static void deq_q5_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    const u8* base = src + (size_t)b*176;
    f32 d    = half_to_float(*(const u16*)base);
    f32 dmin = half_to_float(*(const u16*)(base+2));
    const u8* scales = base + 4;
    const u8* qh = base + 16;
    const u8* ql = base + 48;
    i32 is = 0;
    u8 u1 = 1, u2 = 2;
    for(i32 n=0;n<QK_K;n+=64){
      u8 sc4, m4;
      get_scale_min_k4(is+0, scales, &sc4, &m4);
      f32 d1 = d*sc4, m1 = dmin*m4;
      get_scale_min_k4(is+1, scales, &sc4, &m4);
      f32 d2 = d*sc4, m2 = dmin*m4;
      for(i32 l=0;l<32;l++) dst[(size_t)b*QK_K+n+l     ] = d1*((f32)(ql[l]&0xF) + ((qh[l]&u1)?16.f:0.f)) - m1;
      for(i32 l=0;l<32;l++) dst[(size_t)b*QK_K+n+l+32] = d2*((f32)(ql[l]>>4) + ((qh[l]&u2)?16.f:0.f)) - m2;
      ql += 32;
      is += 2;
      u1 = (u8)(u1<<2); u2 = (u8)(u2<<2);
    }
  }
}
static void deq_q6_K(const u8 *src, f32 *dst, u64 ne){
  u64 nb = ne / QK_K;
  for(u64 b=0;b<nb;b++){
    const u8* ql = src + (size_t)b*210;
    const u8* qh = ql + 128;
    const i8* sc = (const i8*)(qh + 64);
    f32 d = half_to_float(*(const u16*)(sc + 16)); /* d: offset 208 */
    for(i32 n=0;n<QK_K;n+=128){
      for(i32 l=0;l<32;l++){
        i32 is = l/16;
        i32 q1 = ((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        i32 q2 = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        i32 q3 = ((ql[l+ 0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
        i32 q4 = ((ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
        dst[b*QK_K+n+l+ 0] = d * (f32)sc[is+0] * (f32)q1;
        dst[b*QK_K+n+l+32] = d * (f32)sc[is+2] * (f32)q2;
        dst[b*QK_K+n+l+64] = d * (f32)sc[is+4] * (f32)q3;
        dst[b*QK_K+n+l+96] = d * (f32)sc[is+6] * (f32)q4;
      }
      ql += 64; qh += 32; sc += 8;
    }
  }
}
/* ─────────────── i-quants (port fiel de llama.cpp ggml-quants.c, MIT) ─────────────── */
#include "l1_iq_tables.h"
#define IQ1S_DELTA 0.125f

/* IQ2_XXS: bloques de 256, 66 bytes: fp16 d + 32xu16 */
static void deq_iq2_xxs(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256; u32 aux32[2]; const u8 *aux8=(const u8*)aux32;
  for(u64 i=0;i<nb;i++,y+=256){
    const u8 *b=src+i*66;
    const f32 d=half_to_float(*(const u16*)b);
    const u16 *qs=(const u16*)(b+2);
    for(int ib32=0;ib32<8;ib32++){
      memcpy(aux32,qs+4*ib32,8);
      const f32 db=d*(0.5f+(f32)(aux32[1]>>28))*0.25f;
      for(int l=0;l<4;l++){
        const u8 *grid=(const u8*)(iq2xxs_grid+aux8[l]);
        const u8 signs=ksigns_iq2xs[(aux32[1]>>7*l)&127];
        for(int j=0;j<8;j++) y[32*ib32+8*l+j]=db*(f32)grid[j]*((signs&kmask_iq2xs[j])?-1.f:1.f);
      }
    }
  }
}
/* IQ2_XS: bloques de 256, 74 bytes: fp16 d + 32xu16 + 8 escalas */
static void deq_iq2_xs(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256;
  for(u64 i=0;i<nb;i++,y+=256){
    const u8 *b=src+i*74;
    const f32 d=half_to_float(*(const u16*)b);
    const u16 *qs=(const u16*)(b+2);
    const u8 *sc=b+66;
    for(int ib32=0;ib32<8;ib32++){
      const f32 db0=d*(0.5f+(f32)(sc[ib32]&0xf))*0.25f;
      const f32 db1=d*(0.5f+(f32)(sc[ib32]>>4))*0.25f;
      for(int l=0;l<4;l++){
        const u16 q=qs[4*ib32+l];
        const u8 *grid=(const u8*)(iq2xs_grid+(u32)(q&511));
        const u8 signs=ksigns_iq2xs[q>>9];
        const f32 db=(l<2)?db0:db1;
        for(int j=0;j<8;j++) y[32*ib32+8*l+j]=db*(f32)grid[j]*((signs&kmask_iq2xs[j])?-1.f:1.f);
      }
    }
  }
}
/* IQ2_S: bloques de 256, 82 bytes: fp16 d + qs[64] + qh[8] + scales[8] */
static void deq_iq2_s(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256;
  for(u64 i=0;i<nb;i++,y+=256){
    const u8 *b=src+i*82;
    const f32 d=half_to_float(*(const u16*)b);
    const u8 *qs=b+2,*qh=qs+64,*sc=qh+8,*signs=qs+32;
    for(int ib32=0;ib32<8;ib32++){
      const f32 db0=d*(0.5f+(f32)(sc[ib32]&0xf))*0.25f;
      const f32 db1=d*(0.5f+(f32)(sc[ib32]>>4))*0.25f;
      for(int l=0;l<4;l++){
        const f32 dl=(l<2)?db0:db1;
        const u8 *grid=(const u8*)(iq2s_grid+(u32)(qs[l]|((qh[ib32]<<(8-2*l))&0x300)));
        for(int j=0;j<8;j++) y[32*ib32+8*l+j]=dl*(f32)grid[j]*((signs[l]&kmask_iq2xs[j])?-1.f:1.f);
      }
      qs+=4; signs+=4;
    }
  }
}
/* IQ3_XXS: bloques de 256, 98 bytes: fp16 d + qs[96] (ultimos 32 = escalas+signos) */
static void deq_iq3_xxs(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256; u32 aux32;
  for(u64 i=0;i<nb;i++,y+=256){
    const u8 *b=src+i*98;
    const f32 d=half_to_float(*(const u16*)b);
    const u8 *qs=b+2,*ss=qs+64;
    for(int ib32=0;ib32<8;ib32++){
      memcpy(&aux32,ss+4*ib32,4);
      const f32 db=d*(0.5f+(f32)(aux32>>28))*0.5f;
      for(int l=0;l<4;l++){
        const u8 signs=ksigns_iq2xs[(aux32>>7*l)&127];
        const u8 *g1=(const u8*)(iq3xxs_grid+qs[2*l]);
        const u8 *g2=(const u8*)(iq3xxs_grid+qs[2*l+1]);
        for(int j=0;j<4;j++){
          y[32*ib32+8*l+j]   =db*(f32)g1[j]*((signs&kmask_iq2xs[j])  ?-1.f:1.f);
          y[32*ib32+8*l+4+j] =db*(f32)g2[j]*((signs&kmask_iq2xs[j+4])?-1.f:1.f);
        }
      }
      qs+=8;
    }
  }
}
/* IQ3_S: bloques de 256, 110 bytes: fp16 d + qs[64] + qh[8] + signs[32] + scales[4] */
static void deq_iq3_s(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256;
  for(u64 i=0;i<nb;i++){ /* y avanza 256 dentro del bloque */
    const u8 *b=src+i*110;
    const f32 d=half_to_float(*(const u16*)b);
    const u8 *qs=b+2,*qh=qs+64,*signs=qh+8,*sc=signs+32;
    for(int ib32=0;ib32<8;ib32+=2){
      const f32 db1=d*(1.f+2.f*(f32)(sc[ib32/2]&0xf));
      const f32 db2=d*(1.f+2.f*(f32)(sc[ib32/2]>>4));
      for(int l=0;l<4;l++){
        const u8 *g1=(const u8*)(iq3s_grid+(u32)(qs[2*l]  |((qh[0]<<(8-2*l))&256)));
        const u8 *g2=(const u8*)(iq3s_grid+(u32)(qs[2*l+1]|((qh[0]<<(7-2*l))&256)));
        for(int j=0;j<4;j++){
          y[8*l+j]   =db1*(f32)g1[j]*((signs[l]&kmask_iq2xs[j])  ?-1.f:1.f);
          y[8*l+4+j] =db1*(f32)g2[j]*((signs[l]&kmask_iq2xs[j+4])?-1.f:1.f);
        }
      }
      y+=32; qs+=8; signs+=4;
      for(int l=0;l<4;l++){
        const u8 *g1=(const u8*)(iq3s_grid+(u32)(qs[2*l]  |((qh[1]<<(8-2*l))&256)));
        const u8 *g2=(const u8*)(iq3s_grid+(u32)(qs[2*l+1]|((qh[1]<<(7-2*l))&256)));
        for(int j=0;j<4;j++){
          y[8*l+j]   =db2*(f32)g1[j]*((signs[l]&kmask_iq2xs[j])  ?-1.f:1.f);
          y[8*l+4+j] =db2*(f32)g2[j]*((signs[l]&kmask_iq2xs[j+4])?-1.f:1.f);
        }
      }
      y+=32;
      qh+=2; qs+=8; signs+=4;
    }
  }
}
/* IQ1_S: bloques de 256, 50 bytes: fp16 d + qs[32] + qh[8]u16 */
static void deq_iq1_s(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256;
  for(u64 i=0;i<nb;i++,y+=256){
    const u8 *b=src+i*50;
    const f32 d=half_to_float(*(const u16*)b);
    const u8 *qs=b+2;
    const u16 *qh=(const u16*)(b+34);
    for(int ib=0;ib<8;ib++){
      const f32 dl=d*(2.f*(f32)((qh[ib]>>12)&7)+1.f);
      const f32 delta=(qh[ib]&0x8000)?-IQ1S_DELTA:IQ1S_DELTA;
      for(int l=0;l<4;l++){
        const i8 *grid=(const i8*)(iq1s_grid+(u32)(qs[l]|(((qh[ib]>>3*l)&7)<<8)));
        for(int j=0;j<8;j++) y[32*ib+8*l+j]=dl*((f32)grid[j]+delta);
      }
      qs+=4;
    }
  }
}
/* IQ1_M: bloques de 256, 56 bytes: qs[32] + qh[16] + scales[8] */
static void deq_iq1_m(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256;
  for(u64 i=0;i<nb;i++,y+=256){
    const u8 *b=src+i*56;
    const u16 *sc=(const u16*)(b+48);
    u16 su=(u16)((sc[0]>>12)|((sc[1]>>8)&0x00f0)|((sc[2]>>4)&0x0f00)|(sc[3]&0xf000));
    const f32 d=half_to_float(su);
    const u8 *qs=b,*qh=b+32;
    for(int ib=0;ib<8;ib++){
      const f32 dl1=d*(2.f*(f32)((sc[ib/2]>>(6*(ib%2)+0))&7)+1.f);
      const f32 dl2=d*(2.f*(f32)((sc[ib/2]>>(6*(ib%2)+3))&7)+1.f);
      const u16 idx0=(u16)(qs[0]|((qh[0]<<8)&0x700)), idx1=(u16)(qs[1]|((qh[0]<<4)&0x700));
      const u16 idx2=(u16)(qs[2]|((qh[1]<<8)&0x700)), idx3=(u16)(qs[3]|((qh[1]<<4)&0x700));
      const f32 dv[4]={ (qh[0]&0x08)?-IQ1S_DELTA:IQ1S_DELTA, (qh[0]&0x80)?-IQ1S_DELTA:IQ1S_DELTA,
                        (qh[1]&0x08)?-IQ1S_DELTA:IQ1S_DELTA, (qh[1]&0x80)?-IQ1S_DELTA:IQ1S_DELTA };
      const u16 idx[4]={idx0,idx1,idx2,idx3};
      const f32 dl[4]={dl1,dl1,dl2,dl2};
      for(int l=0;l<4;l++){
        const i8 *grid=(const i8*)(iq1s_grid+idx[l]);
        for(int j=0;j<8;j++) y[32*ib+8*l+j]=dl[l]*((f32)grid[j]+dv[l]);
      }
      qs+=4; qh+=2;
    }
  }
}
/* IQ4_NL: bloques de 32, 18 bytes: fp16 d + 16 nibbles con LUT no lineal */
static void deq_iq4_nl(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/32;
  for(u64 i=0;i<nb;i++,y+=32){
    const u8 *b=src+i*18;
    const f32 d=half_to_float(*(const u16*)b);
    const u8 *qs=b+2;
    for(int j=0;j<16;j++){
      y[j]   =d*(f32)kvalues_iq4nl[qs[j]&0xf];
      y[j+16]=d*(f32)kvalues_iq4nl[qs[j]>>4];
    }
  }
}
/* IQ4_XS: bloques de 256, 136 bytes: fp16 d + u16 scales_h + scales_l[4] + qs[128] */
static void deq_iq4_xs(const u8 *src, f32 *y, u64 ne){
  const u64 nb=ne/256;
  for(u64 i=0;i<nb;i++,y+=256){
    const u8 *b=src+i*136;
    const f32 d=half_to_float(*(const u16*)b);
    const u16 sh=*(const u16*)(b+2);
    const u8 *sl=b+4,*qs=b+8;
    for(int ib=0;ib<8;ib++){
      const int ls=((sl[ib/2]>>4*(ib%2))&0xf)|(((sh>>2*ib)&3)<<4);
      const f32 dl=d*(f32)(ls-32);
      for(int j=0;j<16;j++){
        y[32*ib+j]   =dl*(f32)kvalues_iq4nl[qs[j]&0xf];
        y[32*ib+j+16]=dl*(f32)kvalues_iq4nl[qs[j]>>4];
      }
      qs+=16;
    }
  }
}

void gguf_dequant(u32 type, u8 *src, f32 *out, u64 ne){
  switch(type){
    case T_F32:  memcpy(out,src,ne*4); break;
    case T_F16:  for(u64 i=0;i<ne;i++) out[i]=half_to_float(((u16*)src)[i]); break;
    case T_Q4_0: deq_q4_0(src,out,ne); break;
    case T_Q4_1: deq_q4_1(src,out,ne); break;
    case T_Q5_0: deq_q5_0(src,out,ne); break;
    case T_Q4_0S: { /* escala fp16 por superbloque de 256 */
      u64 nsb=ne/256;
      for(u64 sb=0;sb<nsb;sb++){
        const u8 *b_=src+sb*130;
        f32 dsc=half_to_float(*(const u16*)b_);
        for(int g2=0;g2<8;g2++){
          const u8 *nb2=b_+2+g2*16;
          for(int j=0;j<16;j++){
            out[sb*256+g2*32+j]     = dsc*((nb2[j]&0xF)-8);
            out[sb*256+g2*32+16+j]  = dsc*((nb2[j]>>4)-8);
          }
        }
      }
    } break;
    case T_Q4_0S_PSY: { /* 2 escalas por 256: baja 128 + alta 128 */
      u64 nsb=ne/256;
      for(u64 sb=0;sb<nsb;sb++){
        const u8 *b_=src+sb*132;
        f32 d0=half_to_float(*(const u16*)(b_+0));
        f32 d1=half_to_float(*(const u16*)(b_+2));
        for(int g2=0;g2<8;g2++){
          f32 dsc = g2<4 ? d0 : d1;
          const u8 *nb2=b_+4+g2*16;
          for(int j=0;j<16;j++){
            out[sb*256+g2*32+j]     = dsc*((nb2[j]&0xF)-8);
            out[sb*256+g2*32+16+j]  = dsc*((nb2[j]>>4)-8);
          }
        }
      }
    } break;
    case T_Q8_0: deq_q8_0(src,out,ne); break;
    case T_Q2_K: deq_q2_K(src,out,ne); break;
    case T_Q3_K: deq_q3_K(src,out,ne); break;
    case T_Q4_K: deq_q4_K(src,out,ne); break;
    case T_Q5_K: deq_q5_K(src,out,ne); break;
    case T_Q6_K: deq_q6_K(src,out,ne); break;
    case T_IQ2_XXS: deq_iq2_xxs(src,out,ne); break;
    case T_IQ2_XS:  deq_iq2_xs(src,out,ne); break;
    case T_IQ2_S:   deq_iq2_s(src,out,ne); break;
    case T_IQ3_XXS: deq_iq3_xxs(src,out,ne); break;
    case T_IQ3_S:   deq_iq3_s(src,out,ne); break;
    case T_IQ1_S:   deq_iq1_s(src,out,ne); break;
    case T_IQ1_M:   deq_iq1_m(src,out,ne); break;
    case T_IQ4_NL:  deq_iq4_nl(src,out,ne); break;
    case T_IQ4_XS:  deq_iq4_xs(src,out,ne); break;
    default: fprintf(stderr,"codec: tipo %u no soportado\n",type); memset(out,0,ne*4);
  }
}
#if defined(__AVX2__) && !defined(DISABLE_AVX2)
#include <immintrin.h>
static inline i32 hsum_i32(__m256i v){
  __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v,1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
  s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
  return _mm_cvtsi128_si32(s);
}
void matmul_q8_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  #pragma omp parallel for schedule(static) if((i64)n*d > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i*(size_t)nb*34;
    __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
    __m256 a4=_mm256_setzero_ps(), a5=_mm256_setzero_ps(), a6=_mm256_setzero_ps(), a7=_mm256_setzero_ps();
    i32 b=0;
    const i32 nb2 = nb & ~1; /* 2 bloques/iter: 68 bytes → 64 valores con 2 escalas */
    for(; b<nb2; b+=2){
      _mm_prefetch(row+16*34, _MM_HINT_T2);
      __m256 vs0=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m256i qw0=_mm256_loadu_si256((const __m256i*)row); row+=32;
      __m256 vs1=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
      __m256i qw1=_mm256_loadu_si256((const __m256i*)row); row+=32;
      __m128i l0=_mm256_castsi256_si128(qw0), h0=_mm256_extracti128_si256(qw0,1);
      __m128i l1=_mm256_castsi256_si128(qw1), h1=_mm256_extracti128_si256(qw1,1);
      const f32 *xb = x + (size_t)b*32;
      /* block 0 → a0-a3 */
      a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l0)),vs0),_mm256_loadu_ps(xb)  ,a0);
      a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l0,8))),vs0),_mm256_loadu_ps(xb+8) ,a1);
      a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h0)),vs0),_mm256_loadu_ps(xb+16),a2);
      a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h0,8))),vs0),_mm256_loadu_ps(xb+24),a3);
      /* block 1 → a4-a7 (8 accumulators total, same as Q4_0) */
      const f32 *xb2 = xb+32;
      a4=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l1)),vs1),_mm256_loadu_ps(xb2)  ,a4);
      a5=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l1,8))),vs1),_mm256_loadu_ps(xb2+8) ,a5);
      a6=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h1)),vs1),_mm256_loadu_ps(xb2+16),a6);
      a7=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h1,8))),vs1),_mm256_loadu_ps(xb2+24),a7);
    }
    a0=_mm256_add_ps(a0,a4); a1=_mm256_add_ps(a1,a5); a2=_mm256_add_ps(a2,a6); a3=_mm256_add_ps(a3,a7);
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
void matmul_q8_0_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0) return;
  if(B==1){ matmul_q8_0(out,x,w,n,d); return; }
  i32 nb = n/32;
  const i32 nb2 = nb & ~1;
  for(i32 i=0;i<d;i++){
    for(i32 t=0;t<B;t++){
      const u8 *row = w + (size_t)i*(size_t)nb*34;
      __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
      __m256 a4=_mm256_setzero_ps(),a5=_mm256_setzero_ps(),a6=_mm256_setzero_ps(),a7=_mm256_setzero_ps();
      const f32 *xb=x+(size_t)t*n;
      i32 b=0;
      for(; b<nb2; b+=2){
        __m256 vs0=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
        __m256i qw0=_mm256_loadu_si256((const __m256i*)row); row+=32;
        __m256 vs1=_mm256_set1_ps(half_to_float(*(const u16*)row)); row+=2;
        __m256i qw1=_mm256_loadu_si256((const __m256i*)row); row+=32;
        __m128i l0=_mm256_castsi256_si128(qw0), h0=_mm256_extracti128_si256(qw0,1);
        __m128i l1=_mm256_castsi256_si128(qw1), h1=_mm256_extracti128_si256(qw1,1);
        a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l0)),vs0),_mm256_loadu_ps(xb+(size_t)b*32),   a0);
        a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l0,8))),vs0),_mm256_loadu_ps(xb+(size_t)b*32+8), a1);
        a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h0)),vs0),_mm256_loadu_ps(xb+(size_t)b*32+16),a2);
        a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h0,8))),vs0),_mm256_loadu_ps(xb+(size_t)b*32+24),a3);
        const f32 *xb2=xb+(size_t)(b+1)*32;
        a4=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l1)),vs1),_mm256_loadu_ps(xb2),   a4);
        a5=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l1,8))),vs1),_mm256_loadu_ps(xb2+8), a5);
        a6=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h1)),vs1),_mm256_loadu_ps(xb2+16),a6);
        a7=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h1,8))),vs1),_mm256_loadu_ps(xb2+24),a7);
      }
      a0=_mm256_add_ps(a0,a4); a1=_mm256_add_ps(a1,a5); a2=_mm256_add_ps(a2,a6); a3=_mm256_add_ps(a3,a7);
      for(; b<nb; b++){
        f32 sf=half_to_float(*(const u16*)row); row+=2;
        __m256i qw=_mm256_loadu_si256((const __m256i*)row); row+=32;
        __m128i lo=_mm256_castsi256_si128(qw), hi=_mm256_extracti128_si256(qw,1);
        __m256 vs=_mm256_set1_ps(sf);
        a0=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs),_mm256_loadu_ps(xb+(size_t)b*32),   a0);
        a1=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs),_mm256_loadu_ps(xb+(size_t)b*32+8), a1);
        a2=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs),_mm256_loadu_ps(xb+(size_t)b*32+16),a2);
        a3=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs),_mm256_loadu_ps(xb+(size_t)b*32+24),a3);
      }
      float t8[8];
      _mm256_storeu_ps(t8,a0); f32 s0=t8[0]+t8[1]+t8[2]+t8[3]+t8[4]+t8[5]+t8[6]+t8[7];
      _mm256_storeu_ps(t8,a1); f32 s1=t8[0]+t8[1]+t8[2]+t8[3]+t8[4]+t8[5]+t8[6]+t8[7];
      _mm256_storeu_ps(t8,a2); f32 s2=t8[0]+t8[1]+t8[2]+t8[3]+t8[4]+t8[5]+t8[6]+t8[7];
      _mm256_storeu_ps(t8,a3); f32 s3=t8[0]+t8[1]+t8[2]+t8[3]+t8[4]+t8[5]+t8[6]+t8[7];
      out[(size_t)t*d+i]=s0+s1+s2+s3;
    }
  }
}
/* Q4_0 bloque 32 elementos → dot entero contra activación Q8_0. Un hsum por bloque;
   al llamarla 4 veces independientes por iteración, los hsum solapan (ILP). */
static inline f32 q4_dot_block(const __m128i q4, const i8 *q8blk,
                               f32 wscale, f32 ascale,
                               const __m128i mask0F, const __m256i onei8,
                               const __m256i ones16, const __m256i eight16){
  __m128i ql = _mm_and_si128(q4, mask0F);
  __m128i qh = _mm_and_si128(_mm_srli_epi16(q4,4), mask0F);
  __m256i q4u = _mm256_set_m128i(qh, ql); /* 32 nibbles: [0..15]=low, [16..31]=high */
  __m256i q8v = _mm256_loadu_si256((const __m256i*)q8blk);
  __m256i msub = _mm256_maddubs_epi16(q4u, q8v);   /* Σ nib*q8 por par */
  __m256i q8ps = _mm256_maddubs_epi16(onei8, q8v); /* Σ q8 por par (offset -8) */
  __m256i dot16 = _mm256_sub_epi16(msub, _mm256_mullo_epi16(q8ps, eight16));
  __m256i dot32 = _mm256_madd_epi16(dot16, ones16);
  return wscale * ascale * (f32)hsum_i32(dot32);
}
/* Scratch Q8: cuantizado en el hilo maestro; workers solo leen. No usar TLS. */
static u8  *g_q4act; static f32 *g_q4scl; static f32 *g_q4sum; static i32 g_q4cap;
static int q4_scratch(i32 act_elems){
  if(g_q4cap >= act_elems) return 1;
  _mm_free(g_q4act); _mm_free(g_q4scl); _mm_free(g_q4sum);
  g_q4act=NULL; g_q4scl=NULL; g_q4sum=NULL; g_q4cap=0;
  g_q4act=(u8*)_mm_malloc((size_t)act_elems, 32);
  g_q4scl=(f32*)_mm_malloc((size_t)(act_elems/32)*sizeof(f32), 32);
  g_q4sum=(f32*)_mm_malloc((size_t)(act_elems/16)*sizeof(f32), 32);
  if(!g_q4act || !g_q4scl || !g_q4sum){
    _mm_free(g_q4act); _mm_free(g_q4scl); _mm_free(g_q4sum);
    g_q4act=NULL; g_q4scl=NULL; g_q4sum=NULL; return 0;
  }
  g_q4cap = act_elems;
  return 1;
}
/* Cuantiza la activación a Q8 por bloques de 32 (+ suma Q8 por mitades de 16,
   que los kernels K-quant usan como término de corrección). */
static void q4_quant_act(const f32 *x, i32 n, u8 *q8, f32 *q8d, f32 *sum16){
  i32 nb=n/32;
  const __m256 smask=_mm256_set1_ps(-0.f);
  const __m256 half=_mm256_set1_ps(0.5f);
  const __m256i vmin=_mm256_set1_epi32(-127), vmax=_mm256_set1_epi32(127);
  const __m256i perm=_mm256_setr_epi32(0,4,1,5,2,6,3,7);
  for(i32 b=0;b<nb;b++){
    const f32 *xb=x+(size_t)b*32;
    __m256 x0=_mm256_loadu_ps(xb), x1=_mm256_loadu_ps(xb+8);
    __m256 x2=_mm256_loadu_ps(xb+16), x3=_mm256_loadu_ps(xb+24);
    __m256 a=_mm256_max_ps(_mm256_andnot_ps(smask,x0),_mm256_andnot_ps(smask,x1));
    a=_mm256_max_ps(a,_mm256_andnot_ps(smask,x2));
    a=_mm256_max_ps(a,_mm256_andnot_ps(smask,x3));
    __m128 lo=_mm256_castps256_ps128(a), hi=_mm256_extractf128_ps(a,1);
    lo=_mm_max_ps(lo,hi); lo=_mm_max_ps(lo,_mm_movehl_ps(lo,lo));
    lo=_mm_max_ss(lo,_mm_shuffle_ps(lo,lo,1));
    f32 amax=_mm_cvtss_f32(lo);
    if(!(amax>0.f)) amax=1.f;
    f32 dq=amax/127.f;
    q8d[b]=dq;
    __m256 idq=_mm256_set1_ps(1.f/dq);
    /* round-half-away (equiv. lroundf) then clamp to [-127,127] */
    #define Q8RND(xx) _mm256_max_epi32(vmin,_mm256_min_epi32(vmax, \
      _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps((xx),idq), \
        _mm256_or_ps(half,_mm256_and_ps((xx),smask))))))
    __m256i i0=Q8RND(x0), i1=Q8RND(x1), i2=Q8RND(x2), i3=Q8RND(x3);
    #undef Q8RND
    __m256i p8=_mm256_packs_epi16(_mm256_packs_epi32(i0,i1),_mm256_packs_epi32(i2,i3));
    p8=_mm256_permutevar8x32_epi32(p8,perm);
    _mm256_storeu_si256((__m256i*)(q8+(size_t)b*32),p8);
    if(sum16){
      const i8 *q=(const i8*)(q8+(size_t)b*32);
      f32 s0=0.f,s1=0.f;
      for(i32 j=0;j<16;j++) s0+=(f32)q[j];
      for(i32 j=16;j<32;j++) s1+=(f32)q[j];
      sum16[(size_t)b*2]=s0*dq; sum16[(size_t)b*2+1]=s1*dq;
    }
  }
}
void matmul_q4_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  const __m128i mask0F = _mm_set1_epi8(0x0F);
  const __m256i onei8 = _mm256_set1_epi8(1);
  const __m256i ones16 = _mm256_set1_epi16(1);
  const __m256i eight16 = _mm256_set1_epi16(8);
  if(!q4_scratch(n)){ memset(out,0,(size_t)d*sizeof(f32)); return; }
  u8 *q8=g_q4act; f32 *q8d=g_q4scl;
  q4_quant_act(x,n,q8,q8d,NULL);
#define Q4_BLK_ACC(rp,off,xv,scv,acc) do{ \
  __m128i ql=_mm_and_si128(_mm_loadu_si128((const __m128i*)((rp)+(off)+2)),mask0F); \
  __m128i qh=_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i*)((rp)+(off)+2)),4),mask0F); \
  __m256i q4u=_mm256_set_m128i(qh,ql); \
  __m256i ms=_mm256_maddubs_epi16(q4u,(xv)); \
  __m256i qs=_mm256_maddubs_epi16(onei8,(xv)); \
  __m256i dt=_mm256_madd_epi16(_mm256_sub_epi16(ms,_mm256_mullo_epi16(qs,eight16)),ones16); \
  (acc)=_mm256_fmadd_ps(_mm256_set1_ps(scv),_mm256_cvtepi32_ps(dt),(acc)); \
}while(0)
#define Q4_ROW_HSUM(acc,outp) do{ \
  __m128 lo=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1)); \
  lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo)); lo=_mm_add_ss(lo,_mm_shuffle_ps(lo,lo,1)); \
  *(outp)=_mm_cvtss_f32(lo); \
}while(0)
  #pragma omp parallel for schedule(static) if((i64)n*d > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *pa=w+(size_t)i*(size_t)nb*18;
    __m256 accA=_mm256_setzero_ps();
    i32 b=0;
    for(; b+3<nb; b+=4){
      _mm_prefetch(pa+(b+72)*18,_MM_HINT_T0);
      __m256i xv0=_mm256_loadu_si256((const __m256i*)(q8+(size_t)b*32));
      __m256i xv1=_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+1)*32));
      __m256i xv2=_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+2)*32));
      __m256i xv3=_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+3)*32));
      Q4_BLK_ACC(pa,b*18   ,xv0,(f32)half_to_float(*(const u16*)(pa+(size_t)b*18   ))*q8d[b]  ,accA);
      Q4_BLK_ACC(pa,b*18+18,xv1,(f32)half_to_float(*(const u16*)(pa+(size_t)b*18+18))*q8d[b+1],accA);
      Q4_BLK_ACC(pa,b*18+36,xv2,(f32)half_to_float(*(const u16*)(pa+(size_t)b*18+36))*q8d[b+2],accA);
      Q4_BLK_ACC(pa,b*18+54,xv3,(f32)half_to_float(*(const u16*)(pa+(size_t)b*18+54))*q8d[b+3],accA);
    }
    for(; b<nb; b++){
      const u8 *p_=pa+(size_t)b*18;
      Q4_BLK_ACC(p_,0,_mm256_loadu_si256((const __m256i*)(q8+(size_t)b*32)),
                 (f32)half_to_float(*(const u16*)p_)*q8d[b],accA);
    }
    Q4_ROW_HSUM(accA,out+i);
  }
}
/* Batcheado para prefill: x es [B][n], out es [B][d]; cada fila de W se lee
   una vez y se reusa para los B tokens (aritmética intensiva en B). */
void matmul_q4_0_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0) return;
  if(B==1){ matmul_q4_0(out,x,w,n,d); return; }
  i32 nb = n/32;
  const __m128i mask0F = _mm_set1_epi8(0x0F);
  const __m256i onei8 = _mm256_set1_epi8(1);
  const __m256i ones16 = _mm256_set1_epi16(1);
  const __m256i eight16 = _mm256_set1_epi16(8);
  if(!q4_scratch(n*B)){ memset(out,0,(size_t)B*(size_t)d*sizeof(f32)); return; }
  u8 *q8=g_q4act; f32 *q8d=g_q4scl;
  for(i32 t=0;t<B;t++)
    q4_quant_act(x+(size_t)t*n,n,q8+(size_t)t*n,q8d+(size_t)t*(n/32),g_q4sum+(size_t)t*(n/16));
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row0 = w + (size_t)i*(size_t)nb*18;
    for(i32 t=0;t<B;t++){
      const u8 *pa=row0;
      const u8 *qa=q8+(size_t)t*n;
      __m256 acc=_mm256_setzero_ps();
      i32 b=0;
      for(; b+3<nb; b+=4){
        _mm_prefetch(pa+(b+72)*18,_MM_HINT_T0);
        Q4_BLK_ACC(pa,(size_t)b*18   ,_mm256_loadu_si256((const __m256i*)(qa+(size_t)b*32)),
                  (f32)half_to_float(*(const u16*)(pa+(size_t)b*18   ))*q8d[(size_t)t*nb+b]  ,acc);
        Q4_BLK_ACC(pa,(size_t)b*18+18,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+1)*32)),
                  (f32)half_to_float(*(const u16*)(pa+(size_t)b*18+18))*q8d[(size_t)t*nb+b+1],acc);
        Q4_BLK_ACC(pa,(size_t)b*18+36,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+2)*32)),
                  (f32)half_to_float(*(const u16*)(pa+(size_t)b*18+36))*q8d[(size_t)t*nb+b+2],acc);
        Q4_BLK_ACC(pa,(size_t)b*18+54,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+3)*32)),
                  (f32)half_to_float(*(const u16*)(pa+(size_t)b*18+54))*q8d[(size_t)t*nb+b+3],acc);
      }
      for(; b<nb; b++){
        const u8 *p_=pa+(size_t)b*18;
        Q4_BLK_ACC(p_,0,_mm256_loadu_si256((const __m256i*)(qa+(size_t)b*32)),
                   (f32)half_to_float(*(const u16*)p_)*q8d[(size_t)t*nb+b],acc);
      }
      Q4_ROW_HSUM(acc,out+(size_t)t*d+i);
    }
  }
}
/* Q4_0S: escala fp16 por superbloque de 256 (130B); nibbles con convencion Q4_0 */
void matmul_q4_0s(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb=n/32, nsb=n/256;
  const __m128i mask0F=_mm_set1_epi8(0x0F);
  const __m256i onei8=_mm256_set1_epi8(1), ones16=_mm256_set1_epi16(1), eight16=_mm256_set1_epi16(8);
  if(!q4_scratch(n)){ memset(out,0,(size_t)d*sizeof(f32)); return; }
  u8 *q8=g_q4act; f32 *q8d=g_q4scl;
  q4_quant_act(x,n,q8,q8d,NULL);
  #pragma omp parallel for schedule(static) if((i64)n*d > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)nsb*130;
    __m256 acc=_mm256_setzero_ps();
    for(i32 b=0;b<nb;b+=8){
      _mm_prefetch(row+((b>>3)+9)*130,_MM_HINT_T0);
      f32 ws=(f32)half_to_float(*(const u16*)(row+(size_t)(b>>3)*130));
      const u8 *rb=row+(size_t)(b>>3)*130;
      Q4_BLK_ACC(rb,  16*0,_mm256_loadu_si256((const __m256i*)(q8+(size_t)b*32)),      ws*q8d[b]  ,acc);
      Q4_BLK_ACC(rb,  16*1,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+1)*32)),  ws*q8d[b+1],acc);
      Q4_BLK_ACC(rb,  16*2,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+2)*32)),  ws*q8d[b+2],acc);
      Q4_BLK_ACC(rb,  16*3,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+3)*32)),  ws*q8d[b+3],acc);
      Q4_BLK_ACC(rb,  16*4,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+4)*32)),  ws*q8d[b+4],acc);
      Q4_BLK_ACC(rb,  16*5,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+5)*32)),  ws*q8d[b+5],acc);
      Q4_BLK_ACC(rb,  16*6,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+6)*32)),  ws*q8d[b+6],acc);
      Q4_BLK_ACC(rb,  16*7,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+7)*32)),  ws*q8d[b+7],acc);
    }
    Q4_ROW_HSUM(acc,out+i);
  }
}
/* batched: cada fila de pesos se streamea UNA vez para los B tokens */
void matmul_q4_0s_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0) return;
  if(B==1){ matmul_q4_0s(out,x,w,n,d); return; }
  i32 nb=n/32, nsb=n/256;
  const __m128i mask0F=_mm_set1_epi8(0x0F);
  const __m256i onei8=_mm256_set1_epi8(1), ones16=_mm256_set1_epi16(1), eight16=_mm256_set1_epi16(8);
  if(!q4_scratch(n*B)){ memset(out,0,(size_t)B*(size_t)d*sizeof(f32)); return; }
  u8 *q8=g_q4act; f32 *q8d=g_q4scl;
  for(i32 t=0;t<B;t++)
    q4_quant_act(x+(size_t)t*n,n,q8+(size_t)t*n,q8d+(size_t)t*(n/32),NULL);
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)nsb*130;
    for(i32 t=0;t<B;t++){
      const u8 *qa=q8+(size_t)t*n;
      __m256 acc=_mm256_setzero_ps();
      for(i32 b=0;b<nb;b+=8){
        f32 ws=(f32)half_to_float(*(const u16*)(row+(size_t)(b>>3)*130));
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*0,_mm256_loadu_si256((const __m256i*)(qa+(size_t)b*32)),      ws*q8d[(size_t)t*nb+b]  ,acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*1,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+1)*32)),ws*q8d[(size_t)t*nb+b+1],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*2,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+2)*32)),ws*q8d[(size_t)t*nb+b+2],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*3,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+3)*32)),ws*q8d[(size_t)t*nb+b+3],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*4,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+4)*32)),ws*q8d[(size_t)t*nb+b+4],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*5,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+5)*32)),ws*q8d[(size_t)t*nb+b+5],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*6,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+6)*32)),ws*q8d[(size_t)t*nb+b+6],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*130+16*7,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+7)*32)),ws*q8d[(size_t)t*nb+b+7],acc);
      }
      Q4_ROW_HSUM(acc,out+(size_t)t*d+i);
    }
  }
}
/* Q4_0S_PSY: 2 escalas por 256 (132B) — baja 128 + alta 128 */
void matmul_q4_0s_psy(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb=n/32, nsb=n/256;
  const __m128i mask0F=_mm_set1_epi8(0x0F);
  const __m256i onei8=_mm256_set1_epi8(1), ones16=_mm256_set1_epi16(1), eight16=_mm256_set1_epi16(8);
  if(!q4_scratch(n)){ memset(out,0,(size_t)d*sizeof(f32)); return; }
  u8 *q8=g_q4act; f32 *q8d=g_q4scl;
  q4_quant_act(x,n,q8,q8d,NULL);
  #pragma omp parallel for schedule(static) if((i64)n*d > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)nsb*132;
    __m256 acc=_mm256_setzero_ps();
    for(i32 b=0;b<nb;b+=8){
      _mm_prefetch(row+((b>>3)+9)*132,_MM_HINT_T0);
      f32 ws0=(f32)half_to_float(*(const u16*)(row+(size_t)(b>>3)*132+0));
      f32 ws1=(f32)half_to_float(*(const u16*)(row+(size_t)(b>>3)*132+2));
      const u8 *rb=row+(size_t)(b>>3)*132+4;
      Q4_BLK_ACC(rb, 16*0,_mm256_loadu_si256((const __m256i*)(q8+(size_t)b*32)),      ws0*q8d[b]  ,acc);
      Q4_BLK_ACC(rb, 16*1,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+1)*32)),  ws0*q8d[b+1],acc);
      Q4_BLK_ACC(rb, 16*2,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+2)*32)),  ws0*q8d[b+2],acc);
      Q4_BLK_ACC(rb, 16*3,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+3)*32)),  ws0*q8d[b+3],acc);
      Q4_BLK_ACC(rb, 16*4,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+4)*32)),  ws1*q8d[b+4],acc);
      Q4_BLK_ACC(rb, 16*5,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+5)*32)),  ws1*q8d[b+5],acc);
      Q4_BLK_ACC(rb, 16*6,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+6)*32)),  ws1*q8d[b+6],acc);
      Q4_BLK_ACC(rb, 16*7,_mm256_loadu_si256((const __m256i*)(q8+(size_t)(b+7)*32)),  ws1*q8d[b+7],acc);
    }
    Q4_ROW_HSUM(acc,out+i);
  }
}
void matmul_q4_0s_psy_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0) return;
  if(B==1){ matmul_q4_0s_psy(out,x,w,n,d); return; }
  i32 nb=n/32, nsb=n/256;
  const __m128i mask0F=_mm_set1_epi8(0x0F);
  const __m256i onei8=_mm256_set1_epi8(1), ones16=_mm256_set1_epi16(1), eight16=_mm256_set1_epi16(8);
  if(!q4_scratch(n*B)){ memset(out,0,(size_t)B*(size_t)d*sizeof(f32)); return; }
  u8 *q8=g_q4act; f32 *q8d=g_q4scl;
  for(i32 t=0;t<B;t++) q4_quant_act(x+(size_t)t*n,n,q8+(size_t)t*n,q8d+(size_t)t*(n/32),NULL);
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)nsb*132;
    for(i32 t=0;t<B;t++){
      const u8 *qa=q8+(size_t)t*n;
      __m256 acc=_mm256_setzero_ps();
      for(i32 b=0;b<nb;b+=8){
        f32 ws0=(f32)half_to_float(*(const u16*)(row+(size_t)(b>>3)*132+0));
        f32 ws1=(f32)half_to_float(*(const u16*)(row+(size_t)(b>>3)*132+2));
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*0,_mm256_loadu_si256((const __m256i*)(qa+(size_t)b*32)),      ws0*q8d[(size_t)t*nb+b]  ,acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*1,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+1)*32)),ws0*q8d[(size_t)t*nb+b+1],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*2,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+2)*32)),ws0*q8d[(size_t)t*nb+b+2],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*3,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+3)*32)),ws0*q8d[(size_t)t*nb+b+3],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*4,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+4)*32)),ws1*q8d[(size_t)t*nb+b+4],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*5,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+5)*32)),ws1*q8d[(size_t)t*nb+b+5],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*6,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+6)*32)),ws1*q8d[(size_t)t*nb+b+6],acc);
        Q4_BLK_ACC(row,(size_t)(b>>3)*132+4+16*7,_mm256_loadu_si256((const __m256i*)(qa+(size_t)(b+7)*32)),ws1*q8d[(size_t)t*nb+b+7],acc);
      }
      Q4_ROW_HSUM(acc,out+(size_t)t*d+i);
    }
  }
}
/* K-quant fused dots (sin dequantizar la fila) ── */
/* dos mitades de 16 bytes → 32 nibbles contiguos dot contra 32 bytes i8 */
static inline i32 nib_dot2(const __m128i pa, const __m128i pb, const u8 *act){
  __m256i q4u=_mm256_set_m128i(pb,pa);
  __m256i p=_mm256_maddubs_epi16(q4u,_mm256_loadu_si256((const __m256i*)act));
  p=_mm256_madd_epi16(p,_mm256_set1_epi16(1));
  return hsum_i32(p);
}
/* fila Q4_K completa: Σ d_pair·nib·x − m_pair·Σx (con x≈xq·escala) */
static f32 q4k_row_dot(const u8 *row, const u8 *act, const f32 *scl, const f32 *sum16, i32 n){
  const i32 nb256=n/256;
  f32 acc=0.f;
  for(i32 blk=0;blk<nb256;blk++,row+=144){
    f32 d =half_to_float(*(const u16*)row);
    f32 dmin=half_to_float(*(const u16*)(row+2));
    const u8 *sc=row+4, *q=row+16;
    i32 is=0;
    for(i32 j=0;j<4;j++,is+=2,q+=32){
      u8 s1,m1,s2,m2;
      get_scale_min_k4(is  ,sc,&s1,&m1);
      get_scale_min_k4(is+1,sc,&s2,&m2);
      const __m128i mask0F=_mm_set1_epi8(0x0F);
      __m128i qlo=_mm_and_si128(_mm_loadu_si128((const __m128i*)q),mask0F);
      __m128i qlo2=_mm_and_si128(_mm_loadu_si128((const __m128i*)(q+16)),mask0F);
      __m128i qhi=_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i*)q),4),mask0F);
      __m128i qhi2=_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i*)(q+16)),4),mask0F);
      /* elems [b*32,+32)=BAJOS de los 32 bytes (par is); [b+1)=ALTOS (par is+1).
         Σ(w·x)=(d·s)·Σnib·x − (dmin·m)·Σx; sum16 ya es Σx real (dq·Σxq) → sin scl */
      i32 b=blk*8+j*2;
      acc += (f32)nib_dot2(qlo ,qlo2,act+(size_t)b   *32) * (d*(f32)s1)*scl[b]
           - (dmin*(f32)m1)*(sum16[(size_t)b*2]+sum16[(size_t)b*2+1]);
      acc += (f32)nib_dot2(qhi ,qhi2,act+(size_t)(b+1)*32) * (d*(f32)s2)*scl[b+1]
           - (dmin*(f32)m2)*(sum16[(size_t)(b+1)*2]+sum16[(size_t)(b+1)*2+1]);
    }
  }
  return acc;
}
void matmul_q4_K_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0 || !q4_scratch(n*B)) { memset(out,0,(size_t)d*B*sizeof(f32)); return; }
  u8 *qa=g_q4act; f32 *scl=g_q4scl, *sum16=g_q4sum;
  for(i32 t=0;t<B;t++)
    q4_quant_act(x+(size_t)t*n,n,qa+(size_t)t*n,scl+(size_t)t*(n/32),sum16+(size_t)t*(n/16));
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)(n/256)*144;
    for(i32 t=0;t<B;t++)
      out[(size_t)t*d+i]=q4k_row_dot(row,qa+(size_t)t*n,scl+(size_t)t*(n/32),sum16+(size_t)t*(n/16),n);
  }
}
/* fila Q6_K: valor=d·sc[r]·(byte−32); el byte se reensambla por slice de 16 */
static f32 q6k_row_dot(const u8 *row, const u8 *act, const f32 *scl, const f32 *sum16, i32 n){
  const i32 nb256=n/256;
  f32 acc=0.f;
  u8 raw[16];
  for(i32 blk=0;blk<nb256;blk++,row+=210){
    const u8 *ql=row, *qh=row+128;
    const i8 *sc=(const i8*)(row+192);
    f32 d=half_to_float(*(const u16*)(row+208));
    for(i32 c=0;c<2;c++){
      const u8 *qa=ql+(size_t)c*64, *ha=qh+(size_t)c*32;
      const i8 *sc8=sc+(size_t)c*8;
      for(i32 r=0;r<8;r++){
        i32 sl=r>>1, l0=(r&1)*16;
        i32 e=blk*256+c*128+r*16;
        i32 b=e>>5;
        for(i32 j=0;j<16;j++){
          i32 l=l0+j, hi;
          u8 lo2;
          switch(sl){
            case 0: lo2=(u8)(qa[l]&0xF);    hi=ha[l]   &3; break;
            case 1: lo2=(u8)(qa[l+32]&0xF); hi=(ha[l]>>2)&3; break;
            case 2: lo2=(u8)(qa[l]>>4);     hi=(ha[l]>>4)&3; break;
            default:lo2=(u8)(qa[l+32]>>4);  hi=(ha[l]>>6)&3; break;
          }
          raw[j]=(u8)(lo2|(hi<<4));
        }
        __m128i rv=_mm_loadu_si128((const __m128i*)raw);          /* 16 u8 */
        __m128i xv=_mm_loadu_si128((const __m128i*)(act+(size_t)e)); /* 16 i8 */
        __m128i p=_mm_maddubs_epi16(rv,xv);
        p=_mm_madd_epi16(p,_mm_set1_epi16(1));
        p=_mm_add_epi32(p,_mm_srli_si128(p,8));
        p=_mm_add_epi32(p,_mm_srli_si128(p,4));
        i32 dot=_mm_cvtsi128_si32(p);
        acc += d*(f32)sc8[r]*((f32)dot*scl[b] - 32.f*sum16[e>>4]);
      }
    }
  }
  return acc;
}
void matmul_q6_K_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0 || !q4_scratch(n*B)) { memset(out,0,(size_t)d*B*sizeof(f32)); return; }
  u8 *qa=g_q4act; f32 *scl=g_q4scl, *sum16=g_q4sum;
  for(i32 t=0;t<B;t++)
    q4_quant_act(x+(size_t)t*n,n,qa+(size_t)t*n,scl+(size_t)t*(n/32),sum16+(size_t)t*(n/16));
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)(n/256)*210;
    for(i32 t=0;t<B;t++)
      out[(size_t)t*d+i]=q6k_row_dot(row,qa+(size_t)t*n,scl+(size_t)t*(n/32),sum16+(size_t)t*(n/16),n);
  }
}
/* ── Q5_0 fusionado: valor=d·((nib)|(bit<<4)−16); el bit alto entra como máscara
   {0,128} por byte vía LUT y sale como término exacto 16·Σbit·xq ── */
static u64 g_exp5[256]; static int g_exp5_init=0;
static f32 q5_row_dot(const u8 *row, const u8 *act, const f32 *scl, const f32 *sum16, i32 n){
  if(!g_exp5_init){
    for(i32 v=0;v<256;v++){
      u64 m=0; for(int k=0;k<8;k++) if(v&(1<<k)) m |= (u64)128<<(8*k);
      g_exp5[v]=m;
    }
    g_exp5_init=1;
  }
  const __m128i mask0F=_mm_set1_epi8(0x0F);
  const __m256i ones16=_mm256_set1_epi16(1);
  const i32 nb=n/32;
  f32 acc=0.f;
  for(i32 b=0;b<nb;b++,row+=22){
    f32 d=half_to_float(*(const u16*)row);
    u32 qh; memcpy(&qh,row+2,4);
    const u8 *q=row+6;
    __m128i qlo=_mm_and_si128(_mm_loadu_si128((const __m128i*)q),mask0F);
    __m128i qhi=_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i*)q),4),mask0F);
    __m256i xv=_mm256_loadu_si256((const __m256i*)(act+(size_t)b*32));
    __m256i q4u=_mm256_set_m128i(qhi,qlo); /* elems 0..15=bajos, 16..31=altos */
    __m256i p=_mm256_madd_epi16(_mm256_maddubs_epi16(q4u,xv),ones16);
    i32 DOT=hsum_i32(p);
    __m256i mv=_mm256_set_epi64x((i64)g_exp5[(qh>>24)&255],(i64)g_exp5[(qh>>16)&255],
                                 (i64)g_exp5[(qh>>8)&255],(i64)g_exp5[qh&255]);
    i32 tb=hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(mv,xv),ones16))>>7; /* =Σbit·xq */
    acc += d*( (f32)scl[b]*(f32)(DOT + 16*tb) - 16.f*(sum16[(size_t)b*2]+sum16[(size_t)b*2+1]) );
  }
  return acc;
}
void matmul_q5_0_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0 || !q4_scratch(n*B)) { memset(out,0,(size_t)d*B*sizeof(f32)); return; }
  u8 *qa=g_q4act; f32 *scl=g_q4scl, *sum16=g_q4sum;
  for(i32 t=0;t<B;t++)
    q4_quant_act(x+(size_t)t*n,n,qa+(size_t)t*n,scl+(size_t)t*(n/32),sum16+(size_t)t*(n/16));
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)(n/32)*22;
    for(i32 t=0;t<B;t++)
      out[(size_t)t*d+i]=q5_row_dot(row,qa+(size_t)t*n,scl+(size_t)t*(n/32),sum16+(size_t)t*(n/16),n);
  }
}
/* ── IQ1_S fused: 50 bytes per 256, grid 2048*8 + delta ── */
static f32 iq1_s_row_dot(const u8 *row, const u8 *act, const f32 *scl, i32 n){
  int nb256=n/256;
  f32 acc=0.f;
  for(int blk=0; blk<nb256; blk++){
    const u8 *b = row + blk*50;
    f32 d = half_to_float(*(const u16*)b);
    const u8 *qs = b+2;
    const u16 *qh = (const u16*)(b+34);
    for(int ib=0; ib<8; ib++){
      f32 dl = d * (2.f*((qh[ib]>>12)&7)+1.f);
      f32 delta = (qh[ib]&0x8000)? -IQ1S_DELTA : IQ1S_DELTA;
      f32 s = scl[blk*8 + ib];
      const i8 *xa = (const i8*)(act + (blk*8+ib)*32);
      for(int l=0;l<4;l++){
        int idx = qs[l] | (((qh[ib] >> (3*l)) &7)<<8);
        const i8 *grid = (const i8*)iq1s_grid + idx*8;
        for(int j=0;j<8;j++) acc += dl * s * (grid[j]+delta) * (f32)xa[l*8+j];
      }
      qs+=4;
    }
  }
  return acc;
}
void matmul_iq1_s_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0 || !q4_scratch(n*B)) { memset(out,0,(size_t)d*B*sizeof(f32)); return; }
  u8 *qa=g_q4act; f32 *scl=g_q4scl;
  for(i32 t=0;t<B;t++) q4_quant_act(x+(size_t)t*n,n,qa+(size_t)t*n,scl+(size_t)t*(n/32),NULL);
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row=w+(size_t)i*(size_t)(n/256)*50;
    for(i32 t=0;t<B;t++) out[(size_t)t*d+i]=iq1_s_row_dot(row, qa+(size_t)t*n, scl+(size_t)t*(n/32), n);
  }
}

#else
void matmul_q8_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  #pragma omp parallel for schedule(static) if((i64)n*d > OMP_MM_MIN)
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
void matmul_q8_0_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0) return;
  i32 nb = n/32;
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i*(size_t)nb*34;
    for(i32 t=0;t<B;t++){
      f32 sum=0.f;
      const f32 *xb=x+(size_t)t*n;
      const u8 *rb=row;
      for(i32 b=0;b<nb;b++){
        f32 s = half_to_float(*(const u16*)rb); rb+=2;
        for(int j=0;j<32;j++) sum += s*(f32)((i8)rb[j])*xb[(size_t)b*32+j];
        rb+=32;
      }
      out[(size_t)t*d+i]=sum;
    }
  }
}
void matmul_q4_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d){
  i32 nb = n/32;
  #pragma omp parallel for schedule(static) if((i64)n*d > OMP_MM_MIN)
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
void matmul_q4_0_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B){
  if(B<=0) return;
  i32 nb = n/32;
  #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
  for(i32 i=0;i<d;i++){
    const u8 *row = w + (size_t)i * (size_t)nb * 18;
    for(i32 t=0;t<B;t++){
      f32 sum=0.f;
      const f32 *xb=x+(size_t)t*n;
      const u8 *rb=row;
      for(i32 b=0;b<nb;b++){
        f32 s = half_to_float(*(const u16*)rb); rb+=2;
        for(int j=0;j<16;j++){
          int q0=(rb[j]&0x0f)-8, q1=(rb[j]>>4)-8;
          sum += s*((f32)q0*xb[(size_t)b*32+j] + (f32)q1*xb[(size_t)b*32+j+16]);
        }
        rb+=16;
      }
      out[(size_t)t*d+i]=sum;
    }
  }
}
#endif
u64 row_stride(u32 type, i32 n){ return (n/ggml_block_size(type))*ggml_type_bytes(type); }
/* matmul de un rango de filas [r0,r1) — para dual band CPU+GPU del head */
void matmul_q_rows(f32 *out, const f32 *x, const u8 *w, u32 type, i32 n, i32 r0, i32 r1){
  if(!out||!x||!w||r1<=r0) return;
  u64 st=row_stride(type,n);
  i32 d=r1-r0;
  if(type==T_Q4_0){ matmul_q4_0(out+r0,x,w+(size_t)r0*st,n,d); return; }
  if(type==T_Q8_0){ matmul_q8_0(out+r0,x,w+(size_t)r0*st,n,d); return; }
#if defined(__AVX2__) && !defined(DISABLE_AVX2)
  if(type==T_Q4_0S){ matmul_q4_0s(out+r0,x,w+(size_t)r0*st,n,d); return; }
#endif
}
void matmul_q_b(f32 *out, const f32 *x, u8 *w, u32 type, i32 n, i32 d, i32 B){
  if(!out || !x || !w || n<=0 || d<=0 || B<=0){
    if(out && d>0 && B>0) memset(out, 0, (size_t)d*B*sizeof(f32));
    return;
  }
  if(type==T_Q4_0){ matmul_q4_0_b(out,x,w,n,d,B); return; }
#if defined(__AVX2__) && !defined(DISABLE_AVX2)
  if(type==T_Q4_0S){ matmul_q4_0s_b(out,x,w,n,d,B); return; }
#endif
  if(type==T_Q8_0){ matmul_q8_0_b(out,x,w,n,d,B); return; }
#if defined(__AVX2__) && !defined(DISABLE_AVX2)
  if(type==T_Q5_0){ matmul_q5_0_b(out,x,w,n,d,B); return; }
  if(type==T_Q4_K){ matmul_q4_K_b(out,x,w,n,d,B); return; }
  if(type==T_Q6_K){ matmul_q6_K_b(out,x,w,n,d,B); return; }
  if(type==T_Q4_0S){ matmul_q4_0s_b(out,x,w,n,d,B); return; }
  if(type==T_Q4_0S_PSY){ matmul_q4_0s_psy_b(out,x,w,n,d,B); return; }
#endif
  if(type==T_F32){
    #pragma omp parallel for schedule(static) if((i64)n*d*B > OMP_MM_MIN)
    for(i32 i=0;i<d;i++){
      const f32 *wr=(f32*)w+(size_t)i*n;
      for(i32 t=0;t<B;t++){
        const f32 *xb=x+(size_t)t*n; f32 s=0;
        for(i32 j=0;j<n;j++) s+=wr[j]*xb[j];
        out[(size_t)t*d+i]=s;
      }
    }
    return;
  }
  /* Tipos no fusionados: dequant por fila UNA vez y reúso para los B tokens */
  u64 rs=row_stride(type,n);
  int use_omp = ((i64)n*d*B > OMP_MM_MIN);
  #pragma omp parallel if(use_omp)
  {
    f32 *tmp = fallback_buf((size_t)n);
    if(tmp){
      #pragma omp for schedule(static)
      for(i32 i=0;i<d;i++){
        gguf_dequant(type, w+(size_t)i*rs, tmp, (u64)n);
        for(i32 t=0;t<B;t++){
          const f32 *xb=x+(size_t)t*n; f32 s=0;
          for(i32 j=0;j<n;j++) s+=tmp[j]*xb[j];
          out[(size_t)t*d+i]=s;
        }
      }
    } else {
      #pragma omp for schedule(static)
      for(i32 i=0;i<d*B;i++) out[i]=0.f;
    }
  }
}
void matmul_q(f32 *out, f32 *x, u8 *w, u32 type, i32 n, i32 d, f32 *row){
  if(!out || !x || !w || n<=0 || d<=0){
    if(out && d>0) memset(out, 0, (size_t)d * sizeof(f32));
    return;
  }
  if(type==T_Q4_0){ matmul_q4_0(out,x,w,n,d); return; }
#if defined(__AVX2__) && !defined(DISABLE_AVX2)
  if(type==T_Q4_0S){ matmul_q4_0s(out,x,w,n,d); return; }
  if(type==T_Q4_0S_PSY){ matmul_q4_0s_psy(out,x,w,n,d); return; }
#endif

  if(type==T_Q8_0){ matmul_q8_0(out,x,w,n,d); return; }
#if defined(__AVX2__) && !defined(DISABLE_AVX2)
  if(type==T_Q5_0){ matmul_q5_0_b(out,x,w,n,d,1); return; }
  if(type==T_Q4_K){ matmul_q4_K_b(out,x,w,n,d,1); return; }
  if(type==T_Q6_K){ matmul_q6_K_b(out,x,w,n,d,1); return; }
#endif
  if(type==T_F32){ matmul(out,x,(f32*)w,n,d); return; }
  /* Fallback para tipos no fusionados (Q4_1, etc.): per-thread alloc, no per-row */
  u64 rs=row_stride(type,n);
  int use_omp = ((i64)n*d > OMP_MM_MIN);
  #pragma omp parallel if(use_omp)
  {
    f32 *tmp = fallback_buf((size_t)n);
    if(tmp){
      #pragma omp for schedule(static)
      for(i32 i=0;i<d;i++){
        gguf_dequant(type, w+(size_t)i*rs, tmp, (u64)n);
        f32 s=0; for(i32 j=0;j<n;j++) s+=tmp[j]*x[j];
        out[i]=s;
      }
    } else {
      #pragma omp for schedule(static)
      for(i32 i=0;i<d;i++) out[i]=0.f;
    }
  }
  (void)row;
}
