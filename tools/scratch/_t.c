#include "g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void matmul_q4_0s(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_q4_0s_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);
int main(void){
  i32 n=2048,d=2048,B=8;
  u8 *w=malloc((size_t)(n/256)*130*d); if(!w){printf("oom w\n");return 1;}
  for(size_t i=0;i<(size_t)(n/256)*130*d;i++) w[i]=(u8)(rand());
  f32 *x=malloc((size_t)n*B*4); for(i32 i=0;i<n*B;i++) x[i]=(float)((i%97)-48)/100.f;
  f32 *o=malloc((size_t)d*B*4);
  clock_t c=clock();
  matmul_q4_0s(o,x,w,n,d);
  printf("decode x1: %.3fs\n",(double)(clock()-c)/CLOCKS_PER_SEC);
  c=clock();
  matmul_q4_0s_b(o,x,w,n,d,B);
  printf("batched B=%d: %.3fs\n",B,(double)(clock()-c)/CLOCKS_PER_SEC);
  return 0;
}
