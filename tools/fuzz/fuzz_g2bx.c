/* fuzz_g2bx.c — LibFuzzer: g2bx_read_header (formato propio).
 * Igual compilación que fuzz_gguf.c.
 */
#include "internal/g2b.h"
#include "internal/g2bx_io.h"
#include "fuzz_mem.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size){
  if(size > (1u<<24)) return 0;
  FILE *f=fuzz_memfile(data,size);
  if(!f) return 0;
  G2bxHeader h;
  if(g2bx_read_header(f,&h)==0){
    volatile uint64_t w=h.weight_bytes;
    (void)w;
    g2bx_header_free(&h);
  }
  fclose(f);
  return 0;
}

#ifdef FUZZ_STANDALONE
int main(int argc, char **argv){ return fuzz_run_files(argc,argv,LLVMFuzzerTestOneInput); }
#endif
