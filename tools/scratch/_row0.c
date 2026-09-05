#include "../include/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
  GGUF g; if(gguf_load(argv[1],&g)){return 1;}
  for(unsigned i=0;i<g.n_tensors;i++){
    if(!strcmp(g.t[i].name,"blk.0.ffn_down.weight")){
      u64 ne=1; for(unsigned k=0;k<g.t[i].n_dims;k++) ne*=g.t[i].dims[k];
      printf("ne %llu type %u\n",(unsigned long long)ne, g.t[i].type);
      unsigned char *ptr=gguf_tensor_ptr(&g,&g.t[i]);
      int n=3072;
      float *buf=malloc(n*4);
      gguf_dequant(g.t[i].type, ptr, buf, n);
      printf("c W[0][:8] %g %g %g %g %g %g %g %g\n",buf[0],buf[1],buf[2],buf[3],buf[4],buf[5],buf[6],buf[7]);
      FILE *o=fopen("_c_row0.f32","wb"); fwrite(buf,4,n,o); fclose(o);
      break;
    }
  }
  return 0;
}
