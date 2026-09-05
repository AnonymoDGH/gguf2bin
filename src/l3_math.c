/* L3 — math + rope + rmsnorm + softmax + silu */
#include "internal/g2b.h"
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
u16 f32_to_half(f32 f){
  union { f32 f; u32 u; } x; x.f=f;
  u32 bits=x.u, sign=(bits>>16)&0x8000u;
  i32 exp=(i32)((bits>>23)&0xffu);
  u32 mant=bits&0x7fffffu;
  if(exp==255) return (u16)(sign|0x7c00u);
  i32 e=exp-127+15;
  if(e<=0) return (u16)sign;
  if(e>=31) return (u16)(sign|0x7c00u);
  return (u16)(sign|((u32)e<<10)|(mant>>13));
}
void matmul(f32 *xout, f32 *x, f32 *w, i32 n, i32 d){
  #pragma omp parallel for schedule(static) if((i64)n*d > 131072)
  for(i32 i=0;i<d;i++){ f32 s=0; f32 *wr=w+(size_t)i*n; for(i32 j=0;j<n;j++) s+=wr[j]*x[j]; xout[i]=s; }
}
void rmsnorm(f32 *o, f32 *x, f32 *w, i32 n, f32 eps){
#if defined(__AVX2__)
  __m256 ss0=_mm256_setzero_ps(), ss1=_mm256_setzero_ps(), ss2=_mm256_setzero_ps(), ss3=_mm256_setzero_ps();
  i32 j;
  for(j=0;j+31<n;j+=32){
    __m256 x0=_mm256_loadu_ps(x+j), x1=_mm256_loadu_ps(x+j+8), x2=_mm256_loadu_ps(x+j+16), x3=_mm256_loadu_ps(x+j+24);
    ss0=_mm256_fmadd_ps(x0,x0,ss0); ss1=_mm256_fmadd_ps(x1,x1,ss1); ss2=_mm256_fmadd_ps(x2,x2,ss2); ss3=_mm256_fmadd_ps(x3,x3,ss3);
  }
  __m256 ss=_mm256_add_ps(_mm256_add_ps(ss0,ss1),_mm256_add_ps(ss2,ss3));
  __m128 slo=_mm_add_ps(_mm256_castps256_ps128(ss),_mm256_extractf128_ps(ss,1)); slo=_mm_add_ps(slo,_mm_movehl_ps(slo,slo)); slo=_mm_add_ss(slo,_mm_shuffle_ps(slo,slo,1));
  f32 sum=_mm_cvtss_f32(slo);
  for(;j<n;j++) sum+=x[j]*x[j]; /* cola completa: n no multiple de 32 */
#else
  f32 sum=0; for(i32 j=0;j<n;j++) sum+=x[j]*x[j];
#endif
  f32 inv=1.f/sqrtf(sum/(f32)n+eps);
#if defined(__AVX2__)
  __m256 vinv=_mm256_set1_ps(inv);
  for(j=0;j+31<n;j+=32){
    __m256 x0=_mm256_loadu_ps(x+j), x1=_mm256_loadu_ps(x+j+8), x2=_mm256_loadu_ps(x+j+16), x3=_mm256_loadu_ps(x+j+24);
    __m256 w0=_mm256_loadu_ps(w+j), w1=_mm256_loadu_ps(w+j+8), w2=_mm256_loadu_ps(w+j+16), w3=_mm256_loadu_ps(w+j+24);
    _mm256_storeu_ps(o+j  ,_mm256_mul_ps(_mm256_mul_ps(x0,w0),vinv));
    _mm256_storeu_ps(o+j+8,_mm256_mul_ps(_mm256_mul_ps(x1,w1),vinv));
    _mm256_storeu_ps(o+j+16,_mm256_mul_ps(_mm256_mul_ps(x2,w2),vinv));
    _mm256_storeu_ps(o+j+24,_mm256_mul_ps(_mm256_mul_ps(x3,w3),vinv));
  }
  for(;j<n;j++) o[j]=w[j]*x[j]*inv;
#else
  for(i32 j=0;j<n;j++) o[j]=w[j]*x[j]*inv;
#endif
}
/* exp AVX2 ~1e-7 rel err (poly deg 5 + ln2 hi/lo) — base para softmax/silu rápidos */
#if defined(__AVX2__)
static inline __m256 exp256(__m256 x){
  const __m256 log2e=_mm256_set1_ps(1.4426950408889634f);
  const __m256 ln2hi=_mm256_set1_ps(0.693359375f);
  const __m256 ln2lo=_mm256_set1_ps(-2.12194440e-4f);
  const __m256 half=_mm256_set1_ps(0.5f);
  __m256 n=_mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x,log2e),half));
  /* r = x - n*ln2 en [-ln2/2, ln2/2] */
  __m256 r=_mm256_fnmadd_ps(n,ln2hi,x);
  r=_mm256_fnmadd_ps(n,ln2lo,r);
  /* Horner grado 6: 1+r+r^2/2+r^3/6+r^4/24+r^5/120+r^6/720 (err rel <1e-9 en |r|<=ln2/2) */
  __m256 p=_mm256_fmadd_ps(_mm256_set1_ps(1.f/720.f),r,_mm256_set1_ps(1.f/120.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/24.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/6.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(0.5f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f));
  /* 2^n con clamp a [-126,127] para no generar NaN en under/overflow */
  __m256i ni=_mm256_cvtps_epi32(n);
  ni=_mm256_max_epi32(ni,_mm256_set1_epi32(-126));
  ni=_mm256_min_epi32(ni,_mm256_set1_epi32(127));
  __m256 s=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni,_mm256_set1_epi32(127)),23));
  return _mm256_mul_ps(p,s);
}
#endif

void softmax(f32 *x, i32 n){
#if defined(__AVX2__)
  i32 j;
  __m256 mx=_mm256_set1_ps(x[0]);
  for(j=0;j+7<n;j+=8) mx=_mm256_max_ps(mx,_mm256_loadu_ps(x+j));
  __m128 mlo=_mm256_castps256_ps128(mx), mhi=_mm256_extractf128_ps(mx,1);
  mlo=_mm_max_ps(mlo,mhi); mlo=_mm_max_ps(mlo,_mm_movehl_ps(mlo,mlo)); mlo=_mm_max_ss(mlo,_mm_shuffle_ps(mlo,mlo,1));
  f32 m=_mm_cvtss_f32(mlo);
  for(;j<n;j++) if(x[j]>m) m=x[j];
  __m256 vm=_mm256_set1_ps(-m), vs=_mm256_setzero_ps();
  for(j=0;j+7<n;j+=8){
    __m256 e=exp256(_mm256_add_ps(_mm256_loadu_ps(x+j),vm));
    _mm256_storeu_ps(x+j,e); vs=_mm256_add_ps(vs,e);
  }
  __m128 slo=_mm_add_ps(_mm256_castps256_ps128(vs),_mm256_extractf128_ps(vs,1));
  slo=_mm_add_ps(slo,_mm_movehl_ps(slo,slo)); slo=_mm_add_ss(slo,_mm_shuffle_ps(slo,slo,1));
  f32 s=_mm_cvtss_f32(slo);
  for(;j<n;j++){ x[j]=expf(x[j]-m); s+=x[j]; }
  f32 inv=1.f/s;
  __m256 vi=_mm256_set1_ps(inv);
  for(j=0;j+7<n;j+=8) _mm256_storeu_ps(x+j,_mm256_mul_ps(_mm256_loadu_ps(x+j),vi));
  for(;j<n;j++) x[j]*=inv;
#else
  f32 mx=x[0]; for(i32 i=1;i<n;i++) if(x[i]>mx) mx=x[i];
  f32 s=0; for(i32 i=0;i<n;i++){ x[i]=expf(x[i]-mx); s+=x[i]; }
  for(i32 i=0;i<n;i++) x[i]/=s;
#endif
}
void silu(f32 *x, i32 n){
#if defined(__AVX2__)
  i32 nb=n&~7;
#if defined(_OPENMP)
  #pragma omp parallel for schedule(static) if(n > 4096)
#endif
  for(i32 b=0;b<nb;b+=8){
    __m256 g=_mm256_loadu_ps(x+b);
    __m256 e=exp256(_mm256_sub_ps(_mm256_setzero_ps(),g));
    __m256 d=_mm256_add_ps(_mm256_set1_ps(1.f),e);
    __m256 rc=_mm256_rcp_ps(d);
    rc=_mm256_mul_ps(rc,_mm256_fnmadd_ps(d,rc,_mm256_set1_ps(2.f)));
    _mm256_storeu_ps(x+b,_mm256_mul_ps(g,rc));
  }
  for(i32 i=nb;i<n;i++) x[i]=x[i]/(1.f+expf(-x[i]));
#else
  for(i32 i=0;i<n;i++) x[i]=x[i]/(1.f+expf(-x[i]));
#endif
}
void silu_mul(f32 *gate, const f32 *up, i32 n){
#if defined(__AVX2__)
  i32 nb=n&~7;
#if defined(_OPENMP)
  #pragma omp parallel for schedule(static) if(n > 4096)
#endif
  for(i32 b=0;b<nb;b+=8){
    __m256 g=_mm256_loadu_ps(gate+b), u=_mm256_loadu_ps(up+b);
    __m256 e=exp256(_mm256_sub_ps(_mm256_setzero_ps(),g));
    __m256 d=_mm256_add_ps(_mm256_set1_ps(1.f),e);
    __m256 rc=_mm256_rcp_ps(d);
    rc=_mm256_mul_ps(rc,_mm256_fnmadd_ps(d,rc,_mm256_set1_ps(2.f)));
    _mm256_storeu_ps(gate+b,_mm256_mul_ps(_mm256_mul_ps(g,rc),u));
  }
  for(i32 i=nb;i<n;i++) gate[i]=gate[i]/(1.f+expf(-gate[i]))*up[i];
#else
  for(i32 i=0;i<n;i++) gate[i]=gate[i]/(1.f+expf(-gate[i]))*up[i];
#endif
}

/* LLaMA style RoPE: rotate adjacent pairs (0,1), (2,3)... — frecuencias precomputadas sin powf en loop */
void rope_th_llama(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta){
  i32 half=head_dim/2;
  if(half>512){ fprintf(stderr,"rope: head_dim %d > 1024 (unsupported)\n",head_dim); return; }
  f32 cs[1024];
  f32 inv_step = powf(theta, -2.0f/(f32)head_dim);
  f32 freq = 1.0f;
  for(i32 i=0;i<half;i++){
    f32 ang=(f32)pos*freq; cs[2*i]=cosf(ang); cs[2*i+1]=sinf(ang); freq*=inv_step;
  }
  for(i32 h=0;h<len;h+=head_dim){
    i32 pair=0;
    for(i32 i=0;i<head_dim;i+=2,pair++){
      f32 c=cs[2*pair], s=cs[2*pair+1];
      f32 a=x[h+i], b=x[h+i+1];
      x[h+i]=a*c-b*s; x[h+i+1]=a*s+b*c;
    }
  }
}
/* NEOX style RoPE: rotate half-split pairs (0, D/2), (1, D/2+1)... */
void rope_th_neox(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta){
  i32 half = head_dim/2;
  if(half>512){ fprintf(stderr,"rope: head_dim %d > 1024 (unsupported)\n",head_dim); return; }
  f32 cs[1024];
  f32 inv_step = powf(theta, -2.0f/(f32)head_dim);
  f32 freq = 1.0f;
  for(i32 j=0;j<half;j++){
    f32 ang=(f32)pos*freq; cs[2*j]=cosf(ang); cs[2*j+1]=sinf(ang); freq*=inv_step;
  }
  for(i32 h=0;h<len;h+=head_dim){
    for(i32 j=0;j<half;j++){
      f32 c=cs[2*j], s=cs[2*j+1];
      f32 a=x[h+j];
      f32 b=x[h+j+half];
      x[h+j]=a*c-b*s;
      x[h+j+half]=a*s+b*c;
    }
  }
}
/* Legacy alias: prefer rope_th_neox / rope_th_llama via model arch in l5. */
void rope_th(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta){
  rope_th_neox(x,len,pos,head_dim,theta);
}
void qk_rmsnorm(f32 *x, const f32 *w, i32 n_heads, i32 head_dim, f32 eps){
#if defined(__AVX2__)
  for(i32 h=0;h<n_heads;h++){
    f32 *xh=x+h*head_dim;
    __m256 ss0=_mm256_setzero_ps(), ss1=_mm256_setzero_ps(), ss2=_mm256_setzero_ps(), ss3=_mm256_setzero_ps();
    i32 j;
    for(j=0;j+31<head_dim;j+=32){
      __m256 x0=_mm256_loadu_ps(xh+j), x1=_mm256_loadu_ps(xh+j+8), x2=_mm256_loadu_ps(xh+j+16), x3=_mm256_loadu_ps(xh+j+24);
      ss0=_mm256_fmadd_ps(x0,x0,ss0); ss1=_mm256_fmadd_ps(x1,x1,ss1); ss2=_mm256_fmadd_ps(x2,x2,ss2); ss3=_mm256_fmadd_ps(x3,x3,ss3);
    }
    __m256 ss=_mm256_add_ps(_mm256_add_ps(ss0,ss1),_mm256_add_ps(ss2,ss3));
    __m128 slo=_mm_add_ps(_mm256_castps256_ps128(ss),_mm256_extractf128_ps(ss,1)); slo=_mm_add_ps(slo,_mm_movehl_ps(slo,slo)); slo=_mm_add_ss(slo,_mm_shuffle_ps(slo,slo,1));
    f32 sum=_mm_cvtss_f32(slo);
    for(;j<head_dim;j++) sum+=xh[j]*xh[j];
    f32 inv=1.f/sqrtf(sum/head_dim+eps);
    __m256 vinv=_mm256_set1_ps(inv);
    for(j=0;j+31<head_dim;j+=32){
      __m256 x0=_mm256_loadu_ps(xh+j), x1=_mm256_loadu_ps(xh+j+8), x2=_mm256_loadu_ps(xh+j+16), x3=_mm256_loadu_ps(xh+j+24);
      __m256 w0=_mm256_loadu_ps(w+j), w1=_mm256_loadu_ps(w+j+8), w2=_mm256_loadu_ps(w+j+16), w3=_mm256_loadu_ps(w+j+24);
      _mm256_storeu_ps(xh+j,  _mm256_mul_ps(_mm256_mul_ps(x0,w0),vinv));
      _mm256_storeu_ps(xh+j+8,_mm256_mul_ps(_mm256_mul_ps(x1,w1),vinv));
      _mm256_storeu_ps(xh+j+16,_mm256_mul_ps(_mm256_mul_ps(x2,w2),vinv));
      _mm256_storeu_ps(xh+j+24,_mm256_mul_ps(_mm256_mul_ps(x3,w3),vinv));
    }
    for(;j<head_dim;j++) xh[j]=w[j]*xh[j]*inv;
  }
#else
  for(i32 h=0;h<n_heads;h++){
    f32 *xh=x+h*head_dim; f32 ss=0;
    for(i32 j=0;j<head_dim;j++) ss+=xh[j]*xh[j];
    f32 inv=1.f/sqrtf(ss/head_dim+eps);
    for(i32 j=0;j<head_dim;j++) xh[j]=w[j]*xh[j]*inv;
  }
#endif
}
