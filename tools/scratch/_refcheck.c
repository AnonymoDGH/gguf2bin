#include "../include/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "l1_iq_tables.h"

/* dump: dequantiza el primer tensor de cada tipo IQ/K y escribe _c_<tipo>.f32 (hasta 256*8 elems) */
int main(int argc,char**argv){
  GGUF g; if(gguf_load(argv[1],&g)){ printf("load fail\n"); return 1; }
  u32 tipos[]={16,17,18,19,21,22,23,29,10,11,12,13,14};
  for(unsigned t=0;t<sizeof tipos/sizeof tipos[0];t++){
    for(u64 i=0;i<g.n_tensors;i++){
      GTensor *x=&g.t[i];
      if(x->type!=tipos[t] || x->n_dims<2) continue;
      u64 ne=1; for(u32 k=0;k<x->n_dims;k++) ne*=x->dims[k];
      if(ne<(u64)256*4) continue;
      u64 take=ne<(u64)256*8?ne:(u64)256*8;
      f32 *buf=malloc((size_t)take*4);
      u8 *tp=gguf_tensor_ptr(&g,x);
      gguf_dequant(x->type,tp,buf,take);
      if(x->type==18){ printf("off=%llu data_off=%llu bytes:",(unsigned long long)x->offset,(unsigned long long)g.data_off);
        for(int z=0;z<12;z++) printf(" %02x",tp[z]); printf("\n");
        { const u8 *b=tp; u32 a32; memcpy(&a32,b+66,4);
          printf("d=%g aux32=%08x s=%u g197=%d %d %d %d signs0=%u\n",
            half_to_float(*(const u16*)b),a32,a32>>28,
            ((const u8*)iq3xxs_grid)[788],((const u8*)iq3xxs_grid)[789],
            ((const u8*)iq3xxs_grid)[790],((const u8*)iq3xxs_grid)[791],
            ksigns_iq2xs[(a32>>0)&127]);
        } }
      if(x->type==13){ printf("Q6K off=%llu bytes16:",(unsigned long long)x->offset);
        for(int z=0;z<16;z++) printf(" %02x",tp[z]); printf("  b192..209:");
        for(int z=192;z<210;z++) printf(" %02x",tp[z]); printf("\n"); }
      char fn[64]; snprintf(fn,sizeof fn,"_c_%u.f32",tipos[t]);
      FILE *o=fopen(fn,"wb"); fwrite(buf,4,(size_t)take,o); fclose(o);
      printf("%s tipo=%u ne=%llu -> %s\n",x->name,x->type,(unsigned long long)ne,fn);
      free(buf); break;
    }
  }
  return 0;
}
