#include "../include/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } block_q2_K_ref;
static float hf(uint16_t h){ union{uint32_t u; float f;} r; uint32_t s=(h>>15)&1; uint32_t e=(h>>10)&0x1f; uint32_t m=h&0x3ff; if(e==0){ if(!m){r.u=s<<31; return r.f;} float f=m/1024.0f/16384.0f; return s?-f:f;} if(e==31){r.u=(s<<31)|(0xff<<23); return r.f;} r.u=(s<<31)|((e+112)<<23)|(m<<13); return r.f; }
static void deq_q2_ref(const uint8_t *src, float *dst){
  const block_q2_K_ref *b=(const block_q2_K_ref*)src;
  float d=hf(b->d), mn=hf(b->dmin);
  const uint8_t *q=b->qs; int is=0;
  for(int n=0;n<256;n+=128){
    int shift=0;
    for(int j=0;j<4;j++){
      uint8_t sc=b->scales[is++]; float dl=d*(sc&0xF), ml=mn*(sc>>4);
      for(int l=0;l<16;l++) dst[n+j*32+l]=dl*((q[l]>>shift)&3)-ml;
      sc=b->scales[is++]; dl=d*(sc&0xF); ml=mn*(sc>>4);
      for(int l=0;l<16;l++) dst[n+j*32+16+l]=dl*((q[l+16]>>shift)&3)-ml;
      shift+=2;
    }
    q+=32;
  }
}
int main(int argc,char**argv){
  GGUF g; if(gguf_load(argv[1],&g)) return 1;
  for(unsigned i=0;i<g.n_tensors;i++) if(!strcmp(g.t[i].name,"blk.8.attn_v.weight")){
    uint8_t *p=gguf_tensor_ptr(&g,&g.t[i]);
    uint8_t *blk=p+7*84;
    float out_ref[256], out_ours[256];
    deq_q2_ref(blk,out_ref);
    gguf_dequant(10, blk, out_ours, 256);
    for(int k=0;k<256;k++) if(fabsf(out_ref[k]-out_ours[k])>1e-5) {printf("mismatch at %d ref %g ours %g\n",k,out_ref[k],out_ours[k]); return 0;}
    printf("match\n");
    break;
  }
  return 0;
}
