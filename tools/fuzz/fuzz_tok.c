/* fuzz_tok.c — LibFuzzer: tok_read_section (tokenizer serializado).
 * OJO ownership: tok_read_section libera internals al fallar; el llamador
 * solo libera el struct (igual que model.c). En éxito, tok_free completo.
 */
#include "internal/g2b.h"
#include "fuzz_mem.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size){
  if(size > (1u<<20)) return 0;
  FILE *f=fuzz_memfile(data,size);
  if(!f) return 0;
  Tokenizer *t=malloc(sizeof *t);
  if(t){
    if(tok_read_section(f,t)==0){
      volatile int n=t->n;
      (void)n;
      tok_free(t);
    }
    free(t);
  }
  fclose(f);
  return 0;
}

#ifdef FUZZ_STANDALONE
int main(int argc, char **argv){ return fuzz_run_files(argc,argv,LLVMFuzzerTestOneInput); }
#endif
