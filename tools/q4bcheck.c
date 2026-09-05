/* q4bcheck.c â€” valida matmul_q4_0_b (prefill bloqueado) contra Bx matmul_q4_0.
 * Uso: q4bcheck [n] [d] [B] â€” prueba nb par/impar y colas de B (G<4).
 * Criterio: bit-idÃ©ntico (mismo orden de fmadds por salida) salvo re-asociaciÃ³n
 * de la combinaciÃ³n de cadenas (tolerancia 1e-4 relativa). */
#include "g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
void matmul_q4_0_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);
void matmul_q4_0s(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_q4_0s_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);

static int fails=0;
static void check_q4_0(int n,int d,int B){
  size_t nb=(size_t)n/32;
  u8 *w=malloc(nb*18*(size_t)d);
  f32 *x=malloc((size_t)n*B*sizeof(f32));
  f32 *got=calloc((size_t)d*B,sizeof(f32));
  f32 *ref=calloc((size_t)d*B,sizeof(f32));
  if(!w||!x||!got||!ref){ printf("OOM\n"); fails=1; return; }
  srand(7);
  for(size_t i=0;i<(size_t)n*B;i++) x[i]=(f32)rand()/RAND_MAX*2.f-1.f;
  { u16 sh=f32_to_half(0.05f); u8 *p=w;
    for(size_t k=0;k<(size_t)d*nb;k++){ memcpy(p,&sh,2); p+=2;
      for(int j=0;j<16;j++) *p++=(u8)((k*131+7+j*37)&0xff); } }
  matmul_q4_0_b(got,x,w,n,d,B);
  for(int t=0;t<B;t++) matmul_q4_0(ref+(size_t)t*d,x+(size_t)t*n,w,n,d);
  double maxa=0,maxr=0; int bad=0;
  for(size_t i=0;i<(size_t)d*B;i++){
    double a=fabs((double)got[i]-ref[i]);
    double r=a/(fabs((double)ref[i])+1e-6);
    if(a>maxa)maxa=a; if(r>maxr)maxr=r;
    if(!(a<1e-4) && r>1e-4) bad++;
  }
  printf("q4bcheck/Q4_0 n=%d d=%d B=%d nb=%d: max|diff|=%g maxrel=%g elems_off=%d/%d\n",
    n,d,B,(int)nb,maxa,maxr,bad,d*B);
  int fail = !(maxa==maxa) || maxr>1e-4;
  printf("  %s\n", fail?"FAIL":"OK");
  if(fail) fails=1;
  free(w); free(x); free(got); free(ref);
}

static void check_q4_0s(int n,int d,int B){
  if(n%256){ printf("q4bcheck/Q4_0S: n must be a multiple of 256\n"); fails=1; return; }
  size_t nsb=(size_t)n/256;
  u8 *w=malloc(nsb*130*(size_t)d);
  f32 *x=malloc((size_t)n*B*sizeof(f32));
  f32 *got=calloc((size_t)d*B,sizeof(f32));
  f32 *ref=calloc((size_t)d*B,sizeof(f32));
  if(!w||!x||!got||!ref){ printf("OOM\n"); fails=1; return; }
  srand(11);
  for(size_t i=0;i<(size_t)n*B;i++) x[i]=(f32)rand()/RAND_MAX*2.f-1.f;
  { u16 sh=f32_to_half(0.05f); u8 *p=w;
    for(size_t k=0;k<(size_t)d*nsb;k++){ memcpy(p,&sh,2); p+=2;
      for(int j=0;j<128;j++) *p++=(u8)((k*131+7+j*37)&0xff); } }
  matmul_q4_0s_b(got,x,w,n,d,B);
  for(int t=0;t<B;t++) matmul_q4_0s(ref+(size_t)t*d,x+(size_t)t*n,w,n,d);
  double maxa=0,maxr=0; int bad=0;
  for(size_t i=0;i<(size_t)d*B;i++){
    double a=fabs((double)got[i]-ref[i]);
    double r=a/(fabs((double)ref[i])+1e-6);
    if(a>maxa)maxa=a; if(r>maxr)maxr=r;
    if(!(a<1e-4) && r>1e-4) bad++;
  }
  printf("q4bcheck/Q4_0S n=%d d=%d B=%d nsb=%d: max|diff|=%g maxrel=%g elems_off=%d/%d\n",
    n,d,B,(int)nsb,maxa,maxr,bad,d*B);
  int fail = !(maxa==maxa) || maxr>1e-4;
  printf("  %s\n", fail?"FAIL":"OK");
  if(fail) fails=1;
  free(w); free(x); free(got); free(ref);
}

int main(int argc, char **argv){
  if(argc>3){ /* modo puntual: q4bcheck n d B (solo Q4_0) */
    check_q4_0(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
  } else {
    check_q4_0(1024,256,16);
    check_q4_0(96,256,5);
    check_q4_0(3072,512,16);
    check_q4_0s(1024,256,16);
    check_q4_0s(2048,256,5);
  }
  printf(fails?"q4bcheck: FAIL\n":"q4bcheck: OK\n");
  return fails?1:0;
}
