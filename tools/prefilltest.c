/* prefilltest.c — verifica que model_prefill (batcheado) produce los mismos
   logits que el forward secuencial token a token. */
#include "g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv){
  if(argc<2){ fprintf(stderr,"uso: prefilltest <model.g2bx> [ntok]\n"); return 1; }
  int ntok = argc>2 ? atoi(argv[2]) : 40;
  Model m, s;
  if(model_load_g2bx(argv[1],&m)){ return 1; }
  if(model_load_g2bx(argv[1],&s)){ model_free(&m); return 1; }
  if(!m.ix_global[R_TOK_EMBD]){ fprintf(stderr,"sin token_embd\n"); model_free(&m); model_free(&s); return 2; }
  int ctx = m.c.seq_len; if(ctx>512) ctx=512;
  if(ntok>ctx-4) ntok=ctx-4;
  model_set_ctx(&m,ctx);
  model_set_ctx(&s,ctx);

  f32 *lb=malloc((size_t)m.c.vocab*sizeof(f32));
  f32 *ls=malloc((size_t)m.c.vocab*sizeof(f32));
  i32 *toks=malloc((size_t)ntok*sizeof(i32));
  u64 rs=12345;
  for(int i=0;i<ntok;i++){ rs^=rs>>12; rs^=rs<<25; rs^=rs>>27; toks[i]=(i32)((rs*2685821657736338717ull)%(u64)(m.c.vocab>16?m.c.vocab:16)); }

  if(model_prefill(&m,toks,ntok,0,lb)){
    fprintf(stderr,"prefill fallo (rc!=0): sin buffers o error\n");
    free(lb); free(ls); free(toks); model_free(&m); model_free(&s); return 3;
  }
  for(int i=0;i<ntok;i++)
    model_forward_ex(&s,toks[i],i,(i==ntok-1)?ls:NULL,i==ntok-1);

  double maxdiff=0, sumsq=0; int ma=0, ms=0, bad=0;
  for(i32 i=0;i<m.c.vocab;i++){
    double d=fabs((double)lb[i]-(double)ls[i]);
    if(d>maxdiff){ maxdiff=d; }
    sumsq+=d*d;
    if(d>1e-2) bad++;
  }
  for(i32 i=1;i<m.c.vocab;i++){ if(lb[i]>lb[ma]) ma=i; if(ls[i]>ls[ms]) ms=i; }
  printf("prefilltest: ntok=%d ctx=%d vocab=%d\n",ntok,ctx,m.c.vocab);
  printf("  max|batched-seq|=%g rmse=%g elems>1e-2: %d/%d argmax %s (%d vs %d)\n",
    maxdiff,sqrt(sumsq/m.c.vocab),bad,m.c.vocab,ma==ms?"OK":"DIVERGE",ma,ms);
  int ok = (maxdiff < 5e-2) && (ma==ms);
  printf("  %s\n", ok?"EQUIVALENTE":"NO EQUIVALENTE");
  free(lb); free(ls); free(toks);
  model_free(&m); model_free(&s);
  return ok?0:1;
}
