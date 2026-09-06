/* CLI sobre la API pública (include/gguf2bin.h).
 * run/chat/bench/info/pack/vkinfo van por sesiones; ppl usa g2b_ppl.
 * LoRA inference (--cyber <f.lora>) carga pesos reales vía cyber_load_lora. */
#include "gguf2bin.h"
#include "internal/g2b.h"
#include "internal/g2bx_io.h"
#include "internal/opts.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_WIN32)
#include <windows.h>
#endif

/* Construye g2b_config desde los flags comunes run/chat/bench/ppl. */
static void fill_cfg(g2b_config *c, i32 ctx, int nthr, int q8kv, int f32kv,
                     u64 max_ram_mb, const char *swap, int fast,
                     int gpu, const char *lora, u64 seed){
  memset(c,0,sizeof *c);
  c->ctx=ctx; c->threads=nthr;
  c->q8_kv = f32kv?0:(q8kv?1:-1);
  c->max_ram_bytes=max_ram_mb<<20;
  c->swap_path=swap; c->fast=fast;
  c->gpu=gpu; c->lora_path=lora; c->seed=seed;
}

static void run_print_tok(const char *piece, void *ud){
  (void)ud;
  printf("%s",piece); fflush(stdout);
}

static void usage(const char *a0){
  fprintf(stderr,
    "gguf2bin2 - GGUF -> G2BX (own format) -> Qwen3/Llama/LFM2 inference v" G2B_VERSION "\n\n"
    "  %s pack   <model.gguf> <out.g2bx> [--q4] [--prune F]\n"
    "                     --prune 0.5  prune 50%% of the FFN by importance\n"
    "                     (MoE effect: fewer weights/token -> more tok/s)\n"
    "  %s info   <model.g2bx|gguf>\n"
    "  %s run    <model> [text] [opts]\n"
    "      -n N / --n N  tokens to generate (default 64; chat default 256)\n"
    "      -t TEMP       temperature (0=greedy, default 0.7)\n"
    "      --top-k K     top-k (default 40)\n"
    "      --top-p P     nucleus p (default 0.9)\n"
    "      --repeat-penalty R  (default 1.1)\n"
    "      --tokens a,b  raw token ids\n"
    "      --bos         prepend BOS\n"
    "      --gpu         CPU+GPU dual band for the head (needs stable Vulkan)\n"
  "  %s synth  <out.g2bx>\n"
  "  %s verify <model.g2bx>\n"
  "  %s bench  <model> [-n N] [--prefill N]\n"
    "  %s chat   <model> [-n N] [-t TEMP] [--system TXT|--no-system] [--no-think|--think]\n"
    "  %s ppl    <model> [-f file|-] [-n max_tokens]\n\n"
    "RAM / context (run, chat, bench):\n"
    "      -c N / --ctx N     effective context (KV cache) at runtime (default: model's)\n"
    "      --q8-kv            quantized Q8_0 KV cache (~3.8x less RAM than F32)\n"
    "      --f32-kv           force F32 KV cache (disables auto-Q8 above 1GB)\n"
    "      --max-ram MB       RAM budget: auto-Q8 + shrink ctx until it fits (e.g. 2048)\n"
    "      --swap [PATH]      file-backed KV cache; without PATH uses D:\\ if present, else system temp\n"
    "      --fast             everything out: high priority, max threads, no swap\n"
    "      --threads N        OpenMP core count (default: all)\n"
     "      --drop N           skip the N least-influential blocks (ShortGPT)\n\n"
    "bench extras:\n"
    "      --prefill N        measure prompt-processing tok/s (no logits)\n\n"
    "Examples:\n"
    "  %s run qwen.g2bx \"The capital of France is\" -n 20 -t 0\n"
    "  %s chat llama.g2bx --fast --q8-kv -n 256\n"
    "  %s bench qwen.g2bx --max-ram 2048\n",
    a0,a0,a0,a0,a0,a0,a0,a0,a0,a0,a0);
}

static int cmd_pack(int argc, char **argv){
  if(argc<4){ usage(argv[0]); return 1; }
  g2b_pack_opts o; memset(&o,0,sizeof o);
  for(int i=4;i<argc;i++){
    if(!strcmp(argv[i],"--q4")) o.downquant=1;
    else if(!strcmp(argv[i],"--q4s")){ o.downquant=1; o.out_quant=1; }
    else if(!strcmp(argv[i],"--q4s_psy")||!strcmp(argv[i],"--psy")){ o.downquant=1; o.out_quant=2; }
    else if(!strcmp(argv[i],"--q4vvc")||!strcmp(argv[i],"--vvc")){ o.downquant=1; o.out_quant=3; }
    else if(!strcmp(argv[i],"--prune")&&i+1<argc) o.prune=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--calib")&&i+1<argc) o.calib_path=argv[++i];
  }
  g2b_error e=g2b_pack(argv[2],argv[3],&o);
  if(e==G2B_ERR_CONTEXT) fprintf(stderr,"pack: --prune must be in (0,0.9)\n");
  else if(e!=G2B_OK) fprintf(stderr,"pack: %s\n",g2b_strerror(e));
  return e?1:0;
}

static int cmd_info(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  g2b_model_info fi;
  if(g2b_info(argv[2],&fi)!=G2B_OK){ fprintf(stderr,"info: %s\n",argv[2]); return 1; }
  printf("%s arch=%s dim=%d layers=%d heads=%d kv=%d head_dim=%d vocab=%d ctx=%d\n",
    argv[2],fi.arch,fi.dim,fi.n_layers,fi.n_heads,fi.n_kv_heads,
    fi.head_dim,fi.vocab,fi.ctx_max);
  printf("  bos=%d eos=%d tokenizer=%s pesos=%llu MB est_ram=%llu MB\n",
    fi.bos_id,fi.eos_id,fi.has_tokenizer?"yes":"no",
    (unsigned long long)(fi.weight_bytes>>20),
    (unsigned long long)(fi.est_ram_bytes>>20));
  return 0;
}

static int cmd_run(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *path=argv[2];
  i32 n_tok=64; f32 temp=0.7f; int top_k=40; float top_p=0.9f; float rep_pen=1.1f; int use_bos=0;
  const char *cyber=NULL;
  OptsCommon co; opts_common_init(&co);
  i32 prompt[1024]; i32 np=0;
  char text[8192]; text[0]=0;
  for(int i=3;i<argc;i++){
    if(opts_common_try(&co,argc,argv,&i)) continue;
    if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--n"))&&i+1<argc) n_tok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-t")&&i+1<argc) temp=(f32)atof(argv[++i]);
    else if(!strcmp(argv[i],"--top-k")&&i+1<argc) top_k=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--top-p")&&i+1<argc) top_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--repeat-penalty")&&i+1<argc) rep_pen=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--bos")) use_bos=1;
    else if(!strcmp(argv[i],"--cyber")&&i+1<argc) cyber=argv[++i];
    else if(!strcmp(argv[i],"--tokens")&&i+1<argc){
      char *s=argv[++i], *tok;
      for(tok=strtok(s,","); tok&&np<1024; tok=strtok(NULL,",")) prompt[np++]=atoi(tok);
    } else {
      strncat(text,argv[i],sizeof(text)-1-strlen(text));
      strncat(text," ",sizeof(text)-1-strlen(text));
    }
  }
  g2b_config cfg;
  fill_cfg(&cfg,co.ctx,co.threads,co.q8kv,co.f32kv,co.max_ram_mb,co.swap,co.fast,co.gpu,cyber,co.seed);
  g2b_session *s=NULL;
  if(g2b_open(path,&cfg,&s)!=G2B_OK){ fprintf(stderr,"run: cannot open %s\n",path); return 1; }
  if(co.seed) fprintf(stderr,"seed=%llu\n",(unsigned long long)co.seed);
  if(co.ndrop>0) g2b_autodrop(s,co.ndrop);
  g2b_model_info fi;
  if(g2b_model_info_of(s,&fi)!=G2B_OK){ g2b_close(s); return 1; }
  if(text[0] && fi.has_tokenizer){
    i32 *enc=NULL; i32 en=g2b_encode(s,text,&enc);
    if(en>0){
      np=0;
      if(use_bos && fi.bos_id>=0) prompt[np++]=fi.bos_id;
      for(i32 i=0;i<en && np<1024;i++) prompt[np++]=enc[i];
    }
    free(enc);
  }
  if(!np) prompt[np++]=1;
  /* eco del prompt (igual que antes, sin tokenizar de nuevo) */
  if(fi.has_tokenizer){
    if(use_bos && np>1 && prompt[0]==fi.bos_id){
      char *p=g2b_decode(s,&prompt[1],np-1); if(p){ printf("%s",p); free(p); }
    } else {
      char *p=g2b_decode(s,prompt,np); if(p){ printf("%s",p); free(p); }
    }
  } else { for(i32 j=0;j<np;j++) printf("%d ",prompt[j]); }
  g2b_gen_params p;
  memset(&p,0,sizeof p);
  p.temp=temp; p.top_k=top_k; p.top_p=top_p; p.repeat_penalty=rep_pen;
  p.max_tokens=n_tok; p.on_token=run_print_tok;
  g2b_error ge=g2b_generate(s,prompt,np,&p);
  if(ge==G2B_ERR_CONTEXT) fprintf(stderr,"\nrun: token id fuera de vocabulario (%d)\n",fi.vocab);
  printf("\n"); g2b_close(s); return ge?1:0;
}

/* ppl: cross-entropy del modelo sobre un texto (harness de calidad).
   Uso: ppl <model> [-f fichero|-] [-n max_tokens] [opts RAM] */
static int cmd_ppl(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *path=argv[2];
  const char *file=NULL; i32 maxtok=4096;
  const char *cyber=NULL;
  OptsCommon co; opts_common_init(&co);
  for(int i=3;i<argc;i++){
    if(opts_common_try(&co,argc,argv,&i)) continue;
    if(!strcmp(argv[i],"-f")&&i+1<argc) file=argv[++i];
    else if(!strcmp(argv[i],"-n")&&i+1<argc) maxtok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--cyber")&&i+1<argc) cyber=argv[++i];
  }
  g2b_config cfg;
  fill_cfg(&cfg,co.ctx,co.threads,co.q8kv,co.f32kv,co.max_ram_mb,co.swap,0,0,cyber,0);
  g2b_session *s=NULL;
  if(g2b_open(path,&cfg,&s)!=G2B_OK){ fprintf(stderr,"ppl: cannot open %s\n",path); return 1; }
  if(co.ndrop>0) g2b_autodrop(s,co.ndrop);
  char *text=NULL; size_t len=0;
  if(file && strcmp(file,"-")){
    if(os_read_file(file,&text,&len)){ fprintf(stderr,"ppl: cannot open %s\n",file); g2b_close(s); return 1; }
  } else {
    char buf[65536]; size_t r;
    while((r=fread(buf,1,sizeof buf,stdin))>0){
      char *nt2=realloc(text,len+r+1);
      if(!nt2){ fprintf(stderr,"ppl: OOM\n"); free(text); g2b_close(s); return 1; }
      text=nt2; memcpy(text+len,buf,r); len+=r;
    }
  }
  if(!text || len==0){ fprintf(stderr,"ppl: empty text\n"); free(text); g2b_close(s); return 1; }
  text[len]=0;
  g2b_ppl_result pr;
  g2b_error pe=g2b_ppl(s,text,maxtok,&pr);
  free(text);
  if(pe==G2B_ERR_UNSUPPORTED){ fprintf(stderr,"ppl: model has no tokenizer\n"); g2b_close(s); return 1; }
  if(pe){ fprintf(stderr,"ppl: <2 tokens\n"); g2b_close(s); return 1; }
  printf("ppl(%s): tokens=%d evaluated=%d nll/token=%.4f perplexity=%.3f\n",
    path,pr.tokens,pr.evaluated,pr.nll_per_token,pr.perplexity);
  g2b_close(s); return 0;
}

static int cmd_synth(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  return exp_synth_qwen_tiny(argv[2])?1:0;
}

static int cmd_chat(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *path=argv[2];
  i32 n_tok=256; f32 temp=0.7f; int top_k=40; float top_p=0.9f; float rep_pen=1.05f;
  int no_think=1; int show_think=0;
  const char *sys_txt="You are a helpful assistant.";
  int no_sys=0;
  OptsCommon co; opts_common_init(&co);
  for(int i=3;i<argc;i++){
    if(opts_common_try(&co,argc,argv,&i)) continue;
    if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--n"))&&i+1<argc) n_tok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-t")&&i+1<argc) temp=(f32)atof(argv[++i]);
    else if(!strcmp(argv[i],"--top-k")&&i+1<argc) top_k=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--top-p")&&i+1<argc) top_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--no-think")) { no_think=1; show_think=0; }
    else if(!strcmp(argv[i],"--think")) { no_think=0; show_think=1; }
    else if(!strcmp(argv[i],"--repeat-penalty")&&i+1<argc) rep_pen=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--system")&&i+1<argc){ sys_txt=argv[++i]; no_sys=0; }
    else if(!strcmp(argv[i],"--no-system")){ no_sys=1; sys_txt=NULL; }
  }
  g2b_config cfg;
  fill_cfg(&cfg,co.ctx,co.threads,co.q8kv,co.f32kv,co.max_ram_mb,co.swap,co.fast,co.gpu,NULL,co.seed);
  g2b_session *s=NULL;
  if(g2b_open(path,&cfg,&s)!=G2B_OK){ fprintf(stderr,"chat: cannot open %s\n",path); return 1; }
  if(co.seed) fprintf(stderr,"seed=%llu\n",(unsigned long long)co.seed);
  if(co.ndrop>0) g2b_autodrop(s,co.ndrop);
  g2b_model_info fi;
  if(g2b_model_info_of(s,&fi)!=G2B_OK || !fi.has_tokenizer){
    fprintf(stderr,"chat: model has no tokenizer. Re-pack it\n"); g2b_close(s); return 1;
  }
  if(g2b_chat_begin(s,(no_sys?NULL:sys_txt),no_think)!=G2B_OK){
    fprintf(stderr,"chat: no template (neither ChatML <|im_start|> nor Llama 3 <|eot_id|>)\n");
    g2b_close(s); return 1;
  }
  g2b_gen_params p;
  memset(&p,0,sizeof p);
  p.temp=temp; p.top_k=top_k; p.top_p=top_p; p.repeat_penalty=rep_pen;
  p.max_tokens=n_tok; p.show_think=show_think; p.on_token=run_print_tok;
  char line[8192];
  printf("=== Chat %s %s (type 'exit') ===\n", fi.arch,
    no_think?"[no-think]":(show_think?"[thinking]":"[think oculto]"));
  for(;;){
    printf("\nYou> "); fflush(stdout);
    if(!fgets(line,sizeof line,stdin)) break;
    size_t ll=strlen(line); while(ll>0 && (line[ll-1]=='\n'||line[ll-1]=='\r')) line[--ll]=0;
    if(!strcmp(line,"salir")||!strcmp(line,"exit")) break;
    if(ll==0) continue;
    printf("%s> ",fi.arch); fflush(stdout);
    if(g2b_chat_turn(s,line,&p)!=G2B_OK){ fprintf(stderr,"\nchat: turno interrumpido\n"); break; }
    printf("\n");
  }
  g2b_close(s); return 0;
}

static int cmd_bench(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  i32 n=32;
  OptsCommon co; opts_common_init(&co);
  i32 prefill_n=0; int json=0;
  for(int i=3;i<argc;i++){
    if(opts_common_try(&co,argc,argv,&i)) continue;
    if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--n"))&&i+1<argc) n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--prefill")&&i+1<argc) prefill_n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--json")) json=1;
  }
  g2b_config cfg;
  fill_cfg(&cfg,co.ctx,co.threads,co.q8kv,co.f32kv,co.max_ram_mb,co.swap,co.fast,co.gpu,NULL,0);
  g2b_session *s=NULL;
  if(g2b_open(argv[2],&cfg,&s)!=G2B_OK){ fprintf(stderr,"bench: cannot open %s\n",argv[2]); return 1; }
  if(co.ndrop>0) g2b_autodrop(s,co.ndrop);
  g2b_model_info fi;
  if(g2b_model_info_of(s,&fi)!=G2B_OK){ g2b_close(s); return 1; }
  float dec=0, pre=0;
  if(g2b_bench(s,n,prefill_n,&dec,&pre)!=G2B_OK){ g2b_close(s); return 1; }
  if(json)
    printf("{\"model\":\"%s\",\"decode_tps\":%.1f,\"prefill_tps\":%.1f,"
           "\"dim\":%d,\"layers\":%d,\"ctx\":%d}\n",
      argv[2],dec,pre,fi.dim,fi.n_layers,fi.ctx_eff);
  else if(prefill_n>0){
    int pn=prefill_n, mc=fi.ctx_eff>0?fi.ctx_eff:fi.ctx_max;
    if(pn>mc-4) pn=mc-4;
    fprintf(stderr,"bench-prefill: %d tokens in %.3fs -> %.1f tok/s (min of 3) (dim=%d L=%d ctx=%d)\n",
      pn,pn/(pre>0?pre:1),pre,fi.dim,fi.n_layers,fi.ctx_eff);
  } else {
    fprintf(stderr,"bench: %d tokens in %.4fs -> %.1f tok/s (min de 3)  (dim=%d L=%d hd=%d ctx=%d)\n",
      n,n/(dec>0?dec:1),dec,fi.dim,fi.n_layers,fi.head_dim,fi.ctx_eff);
  }
  g2b_close(s); return 0;
}

int main(int argc, char **argv){
#if defined(_WIN32)
  SetConsoleOutputCP(65001);
  SetConsoleCP(65001);
#endif
  fprintf(stderr,"[main] start\n");
  if(argc<2){ usage(argv[0]); return 1; }
  if(!strcmp(argv[1],"run") || !strcmp(argv[1],"chat") ||
     !strcmp(argv[1],"ppl") || !strcmp(argv[1],"bench")){
    for(int i=3;i<argc;i++){
      if(!strcmp(argv[i],"--mv") || !strcmp(argv[i],"--bvh")){
        fprintf(stderr,"%s was removed in v5.1; see docs/ROADMAP_PERF.md\n",argv[i]);
        return 1;
      }
    }
  }
  if(!strcmp(argv[1],"--gpu-worker")) return vk_worker_main(argc,argv);
  if(!strcmp(argv[1],"pack"))  return cmd_pack(argc,argv);
  if(!strcmp(argv[1],"info"))  return cmd_info(argc,argv);
  if(!strcmp(argv[1],"run"))   return cmd_run(argc,argv);
  if(!strcmp(argv[1],"synth")) return cmd_synth(argc,argv);
  if(!strcmp(argv[1],"bench")) return cmd_bench(argc,argv);
  if(!strcmp(argv[1],"chat"))  return cmd_chat(argc,argv);
  if(!strcmp(argv[1],"ppl"))   return cmd_ppl(argc,argv);
  if(!strcmp(argv[1],"vkinfo")){
    char rep[512]; g2b_error e=g2b_vk_probe(rep,sizeof rep);
    printf("%s\n",rep); return e?1:0;
  }
  if(!strcmp(argv[1],"verify")){
    if(argc<3){ usage(argv[0]); return 1; }
    char rep[512]; int r=g2bx_verify(argv[2],rep,sizeof rep);
    printf("%s\n",rep); return r?1:0;
  }
  usage(argv[0]); return 1;
}
