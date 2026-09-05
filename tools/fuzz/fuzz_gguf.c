/* fuzz_gguf.c — LibFuzzer: gguf_load_mem + gguf_free (parser GGUF).
 * Compilar (CI): clang -fsanitize=fuzzer,address -Iinclude -Isrc ...
 * Local sin libFuzzer: -DFUZZ_STANDALONE + lista de archivos.
 */
#include "internal/g2b.h"
#include "fuzz_mem.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size){
  if(size > (1u<<20)) return 0; /* corpus razonable (el parser ya acota) */
  GGUF g;
  if(gguf_load_mem(data,size,&g)==0){
    /* toca la superficie que un packer usaría: lookup + punteros */
    volatile uint64_t n = g.n_tensors;
    (void)n;
    if(g.n_tensors>0 && g.t){
      GTensor *t=gguf_by_name(&g,g.t[0].name);
      if(t) gguf_tensor_ptr(&g,t);
    }
    gguf_free(&g);
  }
  return 0;
}

#ifdef FUZZ_STANDALONE
int main(int argc, char **argv){ return fuzz_run_files(argc,argv,LLVMFuzzerTestOneInput); }
#endif
