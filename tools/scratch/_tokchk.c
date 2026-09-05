#include "../include/g2b.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char**argv){
  for(int a=1;a<argc;a++){
    GGUF g; if(gguf_load(argv[a],&g)){ printf("%s: no abro\n",argv[a]); continue; }
    Tokenizer tk; if(tok_from_gguf(&g,&tk)){ printf("%s: sin tokenizer\n",argv[a]); gguf_free(&g); continue; }
    i32 *ids=NULL; i32 n=tok_encode(&tk,"Hola, preséntate en dos frases.",&ids);
    printf("%s: n=%d bos=%d eos=%d ids:",argv[a],n,tk.bos,tk.eos);
    for(int i=0;i<(n<12?n:12);i++) printf(" %d",ids[i]);
    char *dec=tok_decode(&tk,ids,n);
    printf("\n  decode='%s'\n",dec?dec:"(null)");
    free(dec); free(ids); tok_free(&tk); gguf_free(&g);
  }
  return 0;
}
