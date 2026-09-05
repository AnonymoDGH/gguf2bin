/* q4check.c â€” valida matmul_q4_0 AVX2 contra referencia dequant+dot (float exacto).
   Tolerancia 2%: la ruta AVX2 cuantiza la activaciÃ³n a Q8_0 (error ~1%),
   pero cualquier bug de Ã­ndice/nibble/escala produce error O(1). */
#include "internal/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv){
  int n = argc>1? atoi(argv[1]) : 1024;
  int d = argc>2? atoi(argv[2]) : 4096;
  int seed = argc>3? atoi(argv[3]) : 1;
  if(n<=0 || n%32){ printf("n must be a multiple of 32\n"); return 1; }
  size_t nb = (size_t)n/32;
  size_t wbytes = nb*18*(size_t)d;
  u8 *w = malloc(wbytes);
  f32 *x = malloc((size_t)n*sizeof(f32));
  f32 *got = calloc((size_t)d, sizeof(f32));
  f32 *ref = calloc((size_t)d, sizeof(f32));
  if(!w||!x||!got||!ref){ printf("OOM\n"); return 1; }
  srand(seed);
  for(int i=0;i<n;i++) x[i]=(f32)rand()/RAND_MAX*2.f-1.f;
  /* pesos: escala half finita (0.25f) + nibbles arbitrarios; escala arbitraria
     podria codificar NaN/Inf y romper la comparacion */
  {
    u16 sh = f32_to_half(0.25f);
    u8 *p=w;
    for(size_t b=0;b<(size_t)d*nb;b++){
      memcpy(p,&sh,2); p+=2;
      for(int j=0;j<16;j++) *p++=(u8)((b*131+7+j*37)&0xff);
    }
  }
  /* referencia: replica el algoritmo del kernel en escalar â€” activaciÃ³n cuantizada
     a Q8_0 (igual que matmul_q4_0) + dot entero por bloque. ComparaciÃ³n exacta
     salvo reasociaciÃ³n de float. */
  i8 *q8 = malloc((size_t)n);
  f32 *d32 = malloc((size_t)nb*sizeof(f32));
  if(!q8||!d32){ printf("OOM\n"); return 1; }
  for(size_t b=0;b<nb;b++){
    const f32 *xb = x + b*32;
    f32 amax=0.f;
    for(int j=0;j<32;j++){ f32 a=fabsf(xb[j]); if(a>amax) amax=a; }
    if(!(amax>0.f)) amax=1.f;
    f32 dq=amax/127.f; d32[b]=dq;
    for(int j=0;j<32;j++){ i32 vq=(i32)lroundf(xb[j]/dq); if(vq>127)vq=127; else if(vq<-127)vq=-127; q8[b*32+j]=(i8)vq; }
  }
  for(int i=0;i<d;i++){
    const u8 *row = w + (size_t)i*nb*18;
    double s=0;
    for(size_t b=0;b<nb;b++){
      f32 d0 = half_to_float(*(const u16*)row); row+=2;
      const i8 *qa = q8 + b*32;
      i32 sumi=0;
      for(int j=0;j<16;j++){
        int q0=(row[j]&0x0f)-8, q1=(row[j]>>4)-8;
        sumi += q0*qa[j] + q1*qa[j+16];
      }
      row+=16;
      s += (double)d0 * d32[b] * sumi;
    }
    ref[i]=(f32)s;
  }
  free(q8); free(d32);
  matmul_q4_0(got, x, w, n, d);
  double maxabs=0, maxrel=0;
  for(int i=0;i<d;i++){
    double a=fabs((double)got[i]-ref[i]);
    double r = a/(fabs((double)ref[i])+1e-6);
    if(a>maxabs) maxabs=a;
    if(r>maxrel) maxrel=r;
  }
  printf("q4check n=%d d=%d: max|diff|=%g maxrel=%g\n", n, d, maxabs, maxrel);
  int bad = !(maxabs==maxabs) || maxrel > 0.02; /* NaN => FAIL */
  if(bad) printf("FAIL: maxrel > 2%% (o NaN)\n");
  else printf("OK\n");
  free(w); free(x); free(got); free(ref);
  return bad?1:0;
}
