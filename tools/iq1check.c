/* iq1check.c — valida matmul_iq1_s[_b] (dot entero AVX2 con act Q8).
 * Doble referencia (misma entrada Q8 escalar en ambas):
 *  A) dequant escalar punteado con la activación Q8 reconstruida (q8*dq):
 *     debe coincidir ~al ulp (tol 1e-4). Valida índices/grid/escalas/delta
 *     Y la equivalencia del redondeo Q8 escalar vs q4_quant_act.
 *  B) dequant escalar punteado con x float (verdad GGML): tol 15%%.
 *     Valida que la aproximación Q8 es sana (con ref≈0 por cancelación la
 *     métrica relativa explota por ruido legítimo: B es holgada, manda A).
 * Uso: iq1check [n] [d] [B]  (n múltiplo de 256) */
#include "g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
void matmul_iq1_s(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_iq1_s_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);

static int fails=0;

static void check(int n,int d,int B){
  size_t nb=(size_t)n/256;
  u8 *w=malloc(nb*50*(size_t)d);
  f32 *x=malloc((size_t)n*B*sizeof(f32));
  f32 *got=calloc((size_t)d*B,sizeof(f32));
  f32 *reQ=calloc((size_t)d*B,sizeof(f32));   /* vs Q8-reconstruida */
  f32 *reF=calloc((size_t)d*B,sizeof(f32));   /* vs float */
  f32 *tmp=malloc((size_t)n*sizeof(f32));
  i8 *q8=malloc((size_t)n*B);
  f32 *dq=malloc((size_t)(n/32)*B*sizeof(f32));
  if(!w||!x||!got||!reQ||!reF||!tmp||!q8||!dq){ printf("OOM\n"); fails=1; return; }
  srand(21);
  for(size_t i=0;i<(size_t)n*B;i++) x[i]=(f32)rand()/RAND_MAX*2.f-1.f;
  { u16 dh=f32_to_half(0.5f); u8 *p=w;
    for(size_t k=0;k<(size_t)d*nb;k++){ memcpy(p,&dh,2); p+=2;
      for(int j=0;j<32;j++) *p++=(u8)((k*131+7+j*37)&0xff);
      for(int j=0;j<8;j++){ u16 q=(u16)((k*17+j*257)&0xffff); memcpy(p,&q,2); p+=2; } } }
  /* Q8 escalar (igual que q4_quant_act: amax/127, lroundf, clamp ±127) */
  for(int t=0;t<B;t++) for(size_t b=0;b<(size_t)n/32;b++){
    const f32 *xb=x+(size_t)t*n+b*32;
    f32 amax=0.f;
    for(int j=0;j<32;j++){ f32 a=fabsf(xb[j]); if(a>amax) amax=a; }
    if(!(amax>0.f)) amax=1.f;
    f32 dd=amax/127.f; dq[(size_t)t*(n/32)+b]=dd;
    for(int j=0;j<32;j++){ i32 v=(i32)lroundf(xb[j]/dd);
      if(v>127)v=127; else if(v<-127)v=-127; q8[(size_t)t*n+b*32+j]=(i8)v; }
  }
  if(B==1) matmul_iq1_s(got,x,w,n,d);
  else matmul_iq1_s_b(got,x,w,n,d,B);
  for(int t=0;t<B;t++) for(int i=0;i<d;i++){
    gguf_dequant(T_IQ1_S, w+(size_t)i*nb*50, tmp, (u64)n);
    const f32 *xb=x+(size_t)t*n;
    const i8 *qb=q8+(size_t)t*n;
    const f32 *sb=dq+(size_t)t*(n/32);
    double sq=0,sf=0;
    for(int j=0;j<n;j++){ sq+=(double)tmp[j]*qb[j]*sb[j/32]; sf+=(double)tmp[j]*xb[j]; }
    reQ[(size_t)t*d+i]=(f32)sq; reF[(size_t)t*d+i]=(f32)sf;
  }
  double maxaQ=0,maxrQ=0,maxaF=0,maxrF=0,maxabsF=0; int badQ=0,badF=0;
  for(size_t i=0;i<(size_t)d*B;i++){ double a=fabs((double)reF[i]); if(a>maxabsF)maxabsF=a; }
  for(size_t i=0;i<(size_t)d*B;i++){
    double aq=fabs((double)got[i]-reQ[i]);
    double rq=aq/(fabs((double)reQ[i])+1e-3);
    if(aq>maxaQ)maxaQ=aq; if(rq>maxrQ)maxrQ=rq;
    if(rq>1e-3) badQ++;
    double af=fabs((double)got[i]-reF[i]);
    double rf=af/(fabs((double)reF[i])+1e-3);
    if(af>maxaF)maxaF=af; if(rf>maxrF)maxrF=rf;
    if(rf>0.15 && af>0.02*maxabsF) badF++;
  }
  printf("iq1check n=%d d=%d B=%d:\n",n,d,B);
  printf("  vs-Q8  : max|diff|=%g maxrel=%g off=%d  [tol 1e-3]\n",maxaQ,maxrQ,badQ);
  printf("  vs-float: max|diff|=%g maxrel=%g off=%d  [tol 15%% + 2%%abs]\n",maxaF,maxrF,badF);
  int fail = !(maxaQ==maxaQ) || maxrQ>1e-3 || badF>0;
  printf("  %s\n", fail?"FAIL":"OK");
  if(fail) fails=1;
  free(w); free(x); free(got); free(reQ); free(reF); free(tmp); free(q8); free(dq);
}

int main(int argc, char **argv){
  if(argc>2){ check(atoi(argv[1]),atoi(argv[2]),argc>3?atoi(argv[3]):1); }
  else { check(512,64,1); check(512,64,4); check(5120,16,1); }
  printf(fails?"iq1check: FAIL\n":"iq1check: OK\n");
  return fails?1:0;
}
