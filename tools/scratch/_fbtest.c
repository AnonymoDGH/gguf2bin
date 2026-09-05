#include "../include/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(int argc,char**argv){
  GGUF g; if(gguf_load(argv[1],&g)){ printf("load fail\n"); return 1; }
  struct { u32 ty; const char *nom; } ts[]={ {13,"Q6_K"},{12,"Q4_K"},{11,"Q3_K"} };
  i32 n=1024,B=2;
  for(unsigned t=0;t<3;t++){
    for(u64 i=0;i<g.n_tensors;i++){
      GTensor *x=&g.t[i];
      if(x->type!=ts[t].ty || x->n_dims<2) continue;
      u64 ne=1; for(u32 k=0;k<x->n_dims;k++) ne*=x->dims[k];
      if(ne<(u64)n*64) continue;
      i32 d=(i32)(ne/n); if(d>128) d=128;
      u64 rs=row_stride(x->type,n);
      u8 *w=gguf_tensor_ptr(&g,x);
      f32 *xx=malloc((size_t)n*B*4),*o1=malloc((size_t)d*B*4),*o2=malloc((size_t)d*B*4),*tmp=malloc((size_t)n*4);
      for(i32 j=0;j<n*B;j++) xx[j]=((j%13)-6)/16.f;
      memset(o1,0,(size_t)d*B*4); memset(o2,0,(size_t)d*B*4);
      matmul_q_b(o1,xx,w,x->type,n,d,B);
      for(i32 r=0;r<d;r++){
        gguf_dequant(x->type,w+(size_t)r*rs,tmp,n);
        for(i32 tt=0;tt<B;tt++){ f32 s=0; for(i32 j=0;j<n;j++) s+=tmp[j]*xx[(size_t)tt*n+j]; o2[(size_t)tt*d+r]=s; }
      }
      f32 maxd=0,fm=0; for(i32 k=0;k<d*B;k++){ f32 e=fabsf(o1[k]-o2[k]); if(e>maxd) maxd=e; if(fabsf(o2[k])>fm) fm=fabsf(o2[k]); }
      printf("%s %s n=%d d=%d: maxdiff=%g (escala ref=%g)\n",x->name,ts[t].nom,n,d,maxd,fm);
      fflush(stdout);
      free(xx);free(o1);free(o2);free(tmp);
      break;
    }
  }
  return 0;
}
