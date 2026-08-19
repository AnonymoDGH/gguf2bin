#include "g2b.h"
#include <math.h>
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
  #pragma omp parallel for schedule(static)
  for(i32 i=0;i<d;i++){ f32 s=0; f32 *wr=w+(size_t)i*n; for(i32 j=0;j<n;j++) s+=wr[j]*x[j]; xout[i]=s; }
}
void rmsnorm(f32 *o, f32 *x, f32 *w, i32 n, f32 eps){
  f32 ss=0; for(i32 j=0;j<n;j++) ss+=x[j]*x[j];
  f32 inv=1.f/sqrtf(ss/n+eps);
  for(i32 j=0;j<n;j++) o[j]=w[j]*x[j]*inv;
}
void softmax(f32 *x, i32 n){
  f32 mx=x[0]; for(i32 i=1;i<n;i++) if(x[i]>mx) mx=x[i];
  f32 s=0; for(i32 i=0;i<n;i++){ x[i]=expf(x[i]-mx); s+=x[i]; }
  for(i32 i=0;i<n;i++) x[i]/=s;
}
void silu(f32 *x, i32 n){ for(i32 i=0;i<n;i++) x[i]=x[i]/(1.f+expf(-x[i])); }

/* LLaMA style RoPE: rotate adjacent pairs (0,1), (2,3)... — frecuencias precomputadas sin powf en loop */
void rope_th_llama(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta){
  i32 half=head_dim/2;
  if(half>512){ fprintf(stderr,"rope: head_dim %d > 1024 (no soportado)\n",head_dim); return; }
  f32 cs[1024];
  f32 inv_step = powf(theta, -1.0f/(f32)head_dim);
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
  if(half>512){ fprintf(stderr,"rope: head_dim %d > 1024 (no soportado)\n",head_dim); return; }
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
  for(i32 h=0;h<n_heads;h++){
    f32 *xh=x+h*head_dim; f32 ss=0;
    for(i32 j=0;j<head_dim;j++) ss+=xh[j]*xh[j];
    f32 inv=1.f/sqrtf(ss/head_dim+eps);
    for(i32 j=0;j<head_dim;j++) xh[j]=w[j]*xh[j]*inv;
  }
}
