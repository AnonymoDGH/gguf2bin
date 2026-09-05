/* apitest.c — prueba de la API PÚBLICA (Fase 1).
 *
 * Solo incluye <gguf2bin.h>: si este archivo compila y pasa, la API pública
 * es suficiente para abrir un modelo, inspeccionarlo y generar texto.
 * Se compila SOLO con -Iinclude (sin acceso a internals).
 *
 * Uso: apitest <model.g2bx> [-n N] [--seed S]
 */
#include "gguf2bin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void on_tok(const char *piece, void *ud){
  (void)ud;
  printf("%s", piece);
  fflush(stdout);
}

int main(int argc, char **argv){
  if(argc<2){ fprintf(stderr,"usage: %s <model.g2bx> [-n N] [--seed S]\n",argv[0]); return 1; }
  int n_tok=16; uint64_t seed=42;
  for(int i=2;i<argc;i++){
    if(!strcmp(argv[i],"-n")&&i+1<argc) n_tok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=(uint64_t)strtoull(argv[++i],NULL,10);
  }
  g2b_model_info fi;
  if(g2b_info(argv[1],&fi)!=G2B_OK){ fprintf(stderr,"apitest: g2b_info failed\n"); return 1; }
  printf("apitest: %s dim=%d L=%d heads=%d vocab=%d ctx=%d pesos=%lluMB est=%lluMB tok=%d\n",
    fi.arch,fi.dim,fi.n_layers,fi.n_heads,fi.vocab,fi.ctx_max,
    (unsigned long long)(fi.weight_bytes>>20),
    (unsigned long long)(fi.est_ram_bytes>>20),fi.has_tokenizer);

  g2b_config cfg;
  memset(&cfg,0,sizeof cfg);
  cfg.q8_kv=-1; cfg.seed=seed;
  g2b_session *s=NULL;
  g2b_error e=g2b_open(argv[1],&cfg,&s);
  if(e!=G2B_OK){ fprintf(stderr,"apitest: g2b_open: %s\n",g2b_strerror(e)); return 1; }
  if(g2b_model_info_of(s,&fi)!=G2B_OK){ fprintf(stderr,"apitest: info_of failed\n"); g2b_close(s); return 1; }
  printf("apitest: opened bos=%d eos=%d\n",fi.bos_id,fi.eos_id);

  int32_t prompt[8]={1,2,3,4,5,6,7,8};
  if(fi.vocab>0){ for(int i=0;i<8;i++) prompt[i]=(int32_t)(prompt[i]%fi.vocab); }
  g2b_gen_params p;
  memset(&p,0,sizeof p);
  p.temp=0.7f; p.top_k=40; p.top_p=0.9f; p.repeat_penalty=1.1f;
  p.max_tokens=n_tok; p.seed=seed; p.on_token=on_tok;
  e=g2b_generate(s,prompt,8,&p);
  printf("\n");
  if(e!=G2B_OK){ fprintf(stderr,"apitest: generate: %s\n",g2b_strerror(e)); g2b_close(s); return 1; }
  printf("apitest: ctx_used=%d\n",g2b_ctx_used(s));

  float dec=0, pre=0;
  if(g2b_bench(s,8,0,&dec,&pre)==G2B_OK)
    printf("apitest: bench decode=%.1f tok/s\n",dec);
  g2b_close(s);
  printf("apitest: OK (v%s)\n",g2b_version());
  return 0;
}
