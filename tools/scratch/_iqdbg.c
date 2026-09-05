#include "../include/g2b.h"
#include <stdio.h>
int main(int argc,char**argv){
  printf("bs(IQ1_S)=%llu bytes(IQ1_S)=%llu bs(16)=%llu b(16)=%llu\n",
    (unsigned long long)ggml_block_size(T_IQ1_S),(unsigned long long)ggml_type_bytes(T_IQ1_S),
    (unsigned long long)ggml_block_size(16),(unsigned long long)ggml_type_bytes(16));
  GGUF g; if(gguf_load(argv[1],&g)){ printf("load fail\n"); return 1; }
  printf("size=%llu data_off=%llu n_tensors=%llu\n",(unsigned long long)g.size,
    (unsigned long long)g.data_off,(unsigned long long)g.n_tensors);
  GTensor *t=gguf_by_name(&g,"blk.27.ffn_up.weight");
  if(!t){ printf("tensor no encontrado\n"); return 1; }
  u64 ne=1; for(u32 i=0;i<t->n_dims;i++) ne*=t->dims[i];
  printf("%s tipo=%u dims=%llu,%llu ne=%llu nbytes=%llu start=%llu fin=%llu\n",
    t->name,t->type,(unsigned long long)t->dims[0],(unsigned long long)t->dims[1],
    (unsigned long long)ne,(unsigned long long)ggml_type_size(t->type,ne),
    (unsigned long long)(g.data_off+t->offset),
    (unsigned long long)(g.data_off+t->offset+ggml_type_size(t->type,ne)));
  printf("ptr=%p\n",(void*)gguf_tensor_ptr(&g,t));
  return 0;
}
void matmul(f32 *xout, f32 *x, f32 *w, i32 n, i32 d){for(int i=0;i<d;i++){f32 s=0;for(int j=0;j<n;j++)s+=w[(size_t)i*n+j]*x[j];xout[i]=s;}}
