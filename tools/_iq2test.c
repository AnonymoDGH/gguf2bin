#include "../include/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
int main(int argc,char**argv){
  GGUF g; if(gguf_load(argv[1],&g)) return 1;
  for(unsigned i=0;i<g.n_tensors;i++) if(!strcmp(g.t[i].name,"blk.0.attn_k.weight")){
    uint8_t *p=gguf_tensor_ptr(&g,&g.t[i]);
    uint8_t *blk=p; // first block 66 bytes
    printf("blk bytes: ");
    for(int k=0;k<12;k++) printf("%02x ",blk[k]);
    printf("\n");
    uint16_t d_raw=*(uint16_t*)blk;
    float d = half_to_float(d_raw);
    printf("d %04x %g\n", d_raw, d);
    // our dequant first 8
    float out[256];
    gguf_dequant(16, blk, out, 256);
    printf("our first 8: ");
    for(int k=0;k<8;k++) printf("%g ",out[k]);
    printf("\n");
    break;
  }
  return 0;
}
