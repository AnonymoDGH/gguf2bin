#include "../include/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void matmul(f32 *xout, f32 *x, f32 *w, i32 n, i32 d){for(int i=0;i<d;i++){f32 s=0;for(int j=0;j<n;j++)s+=w[(size_t)i*n+j]*x[j];xout[i]=s;}}
static const char *tn[]={0};
int main(int argc,char**argv){
  GGUF g; if(gguf_load(argv[1],&g)){ printf("load fail\n"); return 1; }
  u64 maxne=0; for(u64 i=0;i<g.n_tensors;i++){ u64 ne=1; for(u32 k=0;k<g.t[i].n_dims;k++) ne*=g.t[i].dims[k]; if(ne>maxne) maxne=ne; }
  f32 *buf=malloc((size_t)maxne*4+4096); u8 *guard=buf?(u8*)buf+(size_t)maxne*4:0;
  if(!buf){ printf("OOM\n"); return 1; }
  int fails=0;
  for(u64 i=0;i<g.n_tensors;i++){
    GTensor *t=&g.t[i];
    if(t->type==T_F32) continue;
    u64 ne=1; for(u32 k=0;k<t->n_dims;k++) ne*=t->dims[k];
    /* relleno de guardia con patrón */
    for(int z=0;z<256;z++) guard[z]=0xA5;
    gguf_dequant(t->type,gguf_tensor_ptr(&g,t),buf,ne);
    int bad=0; for(int z=0;z<256;z++) if(guard[z]!=0xA5) bad=1;
    f32 s=0; for(u64 k=0;k<ne;k++) s+=buf[k];
    if(bad||!(s==s)||s>1e30f||s<-1e30f){
      printf("MAL %s tipo=%u ne=%llu suma=%g overrun=%d\n",t->name,t->type,(unsigned long long)ne,s,bad);
      fails++;
      if(fails>8) break;
    }
  }
  printf("%s\n",fails?"FALLOS":"todos los dequants OK");
  free(buf); return fails?1:0;
}
