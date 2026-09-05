/* mmbench.c — mide el rendimiento real de los kernels matmul Q8_0 vs Q4_0 vs F32
   para la geometria de la capa de salida (n=dim, d=vocab). */
#include "internal/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

static double now_sec(void){
#if defined(_WIN32)
  static int init=0; static double inv; LARGE_INTEGER c,f;
  if(!init){ QueryPerformanceFrequency(&f); inv=1.0/(double)f.QuadPart; init=1; }
  QueryPerformanceCounter(&c); return (double)c.QuadPart*inv;
#else
  return (double)clock()/CLOCKS_PER_SEC;
#endif
}

int main(int argc,char**argv){
  int n = argc>1? atoi(argv[1]) : 1024;
  int d = argc>2? atoi(argv[2]) : 151936;
  int iters = argc>3? atoi(argv[3]) : 20;
#if defined(_OPENMP)
  if(argc>4) omp_set_num_threads(atoi(argv[4]));
#endif
  size_t nbq8 = (size_t)(n/32)*34*(size_t)d;
  size_t nbq4 = (size_t)(n/32)*18*(size_t)d;
  u8 *w8=malloc(nbq8), *w4=malloc(nbq4);
  f32 *x=malloc((size_t)n*4), *o=malloc((size_t)d*4), *tmp=malloc((size_t)(n*4));
  srand(1); for(int i=0;i<n;i++) x[i]=((f32)rand()/RAND_MAX-.5f)*2;
  /* relleno deterministico de q4 blocks (scale + nibbles) */
  { u8 *p=w4; for(size_t b=0;b<nbq4;b++) p[b]=(u8)(b&0xff); }
  { u8 *p=w8; for(size_t b=0;b<nbq8;b++) p[b]=(u8)(b&0xff); }

  double t0=now_sec();
  for(int it=0;it<iters;it++) matmul_q8_0(o,x,w8,n,d);
  double t8=(now_sec()-t0)/iters;
  t0=now_sec();
  for(int it=0;it<iters;it++) matmul_q(o,x,w4,T_Q4_0,n,d,tmp); /* fuerza Q4 path */
  double t4=(now_sec()-t0)/iters;
  free(tmp);
  printf("geom n=%d d=%d iters=%d\n",n,d,iters);
  printf("Q8_0: %6.1f ms  %6.1f GB/s\n", t8*1e3, nbq8/1e9/t8);
  printf("Q4_0: %6.1f ms  %6.1f GB/s  (pesos %zu MB vs %zu MB)\n", t4*1e3, nbq4/1e9/t4,
         nbq4>>20, nbq8>>20);
  free(w8);free(w4);free(x);free(o);
  return 0;
}