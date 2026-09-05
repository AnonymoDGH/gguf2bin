/* valida matmul_q4_0 / matmul_q8_0 vs dequant+dot naive */
#include "g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static unsigned long long rs=88172645463325252ull;
static double frand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (double)((rs>>11)*(1.0/9007199254740992.0)); }

static void quant_q8_0(const f32 *x, int n, u8 *out){
  int nb=n/32;
  for(int b=0;b<nb;b++){
    f32 amax=0;
    for(int i=0;i<32;i++){ f32 a=fabsf(x[b*32+i]); if(a>amax) amax=a; }
    f32 d=amax/127.f;
    u16 h=f32_to_half(d);
    out[b*34]=h&0xff; out[b*34+1]=(h>>8)&0xff;
    for(int i=0;i<32;i++){
      f32 v=x[b*32+i]/d;
      int q=(int)lrintf(v); if(q>127)q=127; if(q<-128)q=-128;
      out[b*34+2+i]=(u8)(q);
    }
  }
}
static void quant_q4_0(const f32 *x, int n, u8 *out){
  int nb=n/32;
  for(int b=0;b<nb;b++){
    f32 amax=0, max_=0;
    for(int i=0;i<32;i++){ f32 a=x[b*32+i]; if(a>max_) max_=a; if(-a>amax) amax=-a; }
    f32 d=amax/((1<<3)-1), id=d?1.f/d:0.f;
    f32 xd = d?f32_to_half(d):0;
    u16 h=(u16)xd;
    out[b*18]=h&0xff; out[b*18+1]=(h>>8)&0xff;
    f32 xv=half_to_float(h);
    id = xv?1.f/xv:0.f;
    for(int i=0;i<16;i++){
      int q0=(int)lrintf(x[b*32+2*i]*id)+8;
      int q1=(int)lrintf(x[b*32+2*i+1]*id)+8;
      if(q0>15)q0=15; if(q0<0)q0=0;
      if(q1>15)q1=15; if(q1<0)q1=0;
      out[b*18+2+i]=(u8)(q0|(q1<<4));
    }
  }
}

int main(void){
  const int n=256, d=64;
  f32 *x=malloc(n*4), *ref=malloc(d*4), *got=malloc(d*4), *w=malloc((size_t)n*d*4);
  for(int j=0;j<n*d;j++) w[j]=(f32)(frand()*2-1);
  for(int j=0;j<n;j++) x[j]=(f32)(frand()*2-1);

  /* F32 reference */
  for(int i=0;i<d;i++){ f32 s=0; for(int j=0;j<n;j++) s+=w[(size_t)i*n+j]*x[j]; ref[i]=s; }

  /* Q8_0 */
  {
    size_t rs=row_stride(T_Q8_0,n);
    u8 *wq=malloc(rs*d);
    for(int i=0;i<d;i++) quant_q8_0(w+(size_t)i*n,n,wq+(size_t)i*rs);
    memset(got,0,d*4);
    /* naive: dequant row then dot */
    for(int i=0;i<d;i++){
      f32 *tmp=malloc(n*4);
      gguf_dequant(T_Q8_0,wq+(size_t)i*rs,tmp,(u64)n);
      f32 s=0; for(int j=0;j<n;j++) s+=tmp[j]*x[j];
      got[i]=s; free(tmp);
    }
    double md=0; int wi=0;
    for(int i=0;i<d;i++){ double e=fabs(got[i]-ref[i]); if(e>md){md=e;wi=i;} }
    printf("Q8_0 naive-dequant: max|diff|=%g (fila %d)\n",md,wi);
    matmul_q8_0(got,x,wq,n,d);
    md=0; wi=0;
    for(int i=0;i<d;i++){ double e=fabs(got[i]-ref[i]); if(e>md){md=e;wi=i;} }
    printf("Q8_0 kernel      : max|diff|=%g (fila %d)\n",md,wi);
    free(wq);
  }
  /* Q4_0 */
  {
    size_t rs=row_stride(T_Q4_0,n);
    u8 *wq=malloc(rs*d);
    for(int i=0;i<d;i++) quant_q4_0(w+(size_t)i*n,n,wq+(size_t)i*rs);
    matmul_q4_0(got,x,wq,n,d);
    double md=0; int wi=0;
    for(int i=0;i<d;i++){ double e=fabs(got[i]-ref[i]); if(e>md){md=e;wi=i;} }
    printf("Q4_0 kernel      : max|diff|=%g (fila %d, escala Q4 ~%g esperable)\n",md,wi,md/fabs(ref[wi]+1e-9));
    free(wq);
  }
  return 0;
}
