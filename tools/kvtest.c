/* kvtest.c â€” compara logits con KV cache F32 vs Q8_0 (misma arquitectura/pesos). */
#include "internal/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv){
  if(argc<2){ fprintf(stderr,"usage: kvtest <model.g2bx> [steps]\n"); return 1; }
  const char *path=argv[1];
  int steps = argc>2 ? atoi(argv[2]) : 32;
  if(steps<1) steps=1;
  int ctx = steps+8;

  Model a, b;
  if(model_load_g2bx(path,&a)){ fprintf(stderr,"no cargo %s\n",path); return 1; }
  if(!a.ix_global[R_TOK_EMBD]){ fprintf(stderr,"model without token_embd (draft); kvtest not applicable\n"); model_free(&a); return 2; }
  if(model_load_g2bx(path,&b)){ model_free(&a); return 1; }
  b.flags |= F_KV_Q8;

  if(ctx > a.c.seq_len) ctx = a.c.seq_len;
  if(model_set_ctx(&a,ctx)){ fprintf(stderr,"set_ctx A failed\n"); return 1; }
  if(model_set_ctx(&b,ctx)){ fprintf(stderr,"set_ctx B failed\n"); return 1; }

  f32 *la=malloc((size_t)a.c.vocab*sizeof(f32));
  f32 *lb=malloc((size_t)a.c.vocab*sizeof(f32));
  double maxdiff=0, sumsq=0; int ncmp=0, agree=0, nsteps=0;
  int step = (a.c.vocab>4096)? a.c.vocab/4096 : 1;
  for(int p=0;p<steps;p++){
    i32 tok = 10 + (p % ((a.c.vocab-10)>1? (a.c.vocab-10) : 1));
    model_forward(&a, tok, p, la);
    model_forward(&b, tok, p, lb);
    for(i32 i=0;i<a.c.vocab;i+=step){
      double d=fabs((double)la[i]-(double)lb[i]);
      if(d>maxdiff) maxdiff=d;
      sumsq+=d*d; ncmp++;
    }
    i32 ma=0, mb=0;
    for(i32 i=1;i<a.c.vocab;i++){ if(la[i]>la[ma]) ma=i; if(lb[i]>lb[mb]) mb=i; }
    if(ma==mb) agree++;
    nsteps++;
  }
  double rmse = ncmp? sqrt(sumsq/ncmp) : 0.0;
  int vocab=a.c.vocab;
  model_free(&a); model_free(&b);
  printf("kvtest: steps=%d ctx=%d vocab=%d\n", steps, ctx, vocab);
  printf("  max|dF32-dQ8| = %g   rmse=%g   argmax(greedy) match %d/%d = %.0f%%\n",
    maxdiff, rmse, agree, nsteps, 100.0*(double)agree/(double)(nsteps?nsteps:1));
  printf("  (KV q8: rmse~0.1-0.3 esperado en logits; argmax >90%% indica calidad OK)\n");
  free(la); free(lb);
  return 0;
}
