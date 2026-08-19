/* CLI: pack | info | run | synth | chat | bench — runtime G2BX v3.3 */
#include "g2b.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

#define RECENT_CAP 128
#define RECENT_USE 64

static void set_threads(int n){
#if defined(_OPENMP)
  if(n>0) omp_set_num_threads(n);
  int t = n>0?n:omp_get_num_threads();
  fprintf(stderr,"hilos: usando %d core(s)\n", t);
#else
  fprintf(stderr,"hilos: compilado sin OpenMP (usa -fopenmp para acelerar)\n");
#endif
}

typedef struct {
  i32 ids[RECENT_CAP];
  int n; /* total pushes; ring index = n % RECENT_CAP */
} RecentBuf;

static void recent_push(RecentBuf *r, i32 id){
  r->ids[r->n % RECENT_CAP] = id;
  r->n++;
}

/* Copy the last min(r->n, maxn, RECENT_CAP) tokens into out (oldest→newest). */
static int recent_snapshot(const RecentBuf *r, i32 *out, int maxn){
  if(!r || r->n<=0 || maxn<=0) return 0;
  int avail = r->n < RECENT_CAP ? r->n : RECENT_CAP;
  if(avail > maxn) avail = maxn;
  int start = r->n - avail;
  for(int i=0;i<avail;i++) out[i] = r->ids[(start + i) % RECENT_CAP];
  return avail;
}

static void usage(const char *a0){
  fprintf(stderr,
    "gguf2bin2 — GGUF -> G2BX (formato propio) -> inferencia Qwen3/Llama\n\n"
    "  %s pack   <model.gguf> <out.g2bx> [--q4]\n"
    "  %s info   <model.g2bx|gguf>\n"
    "  %s run    <model> [texto] [opts]\n"
    "      -n N          tokens a generar (def 64)\n"
    "      -t TEMP       temperatura (0=greedy, def 0.7)\n"
    "      --top-k K     top-k (def 40)\n"
    "      --top-p P     nucleus p (def 0.9)\n"
    "      --repeat-penalty R  (def 1.1)\n"
    "      --tokens a,b  ids directos\n"
    "      --bos         prepend BOS\n"
    "  %s synth  <out.g2bx>\n"
    "  %s bench  <model> [-n N]\n"
    "  %s chat   <model> [-n N] [-t TEMP] [--no-think]\n\n"
    "RAM / contexto (run, chat, bench):\n"
    "      -c N / --ctx N     contexto efectivo (KV cache) en runtime (def: el del modelo)\n"
    "      --q8-kv            KV cache cuantizada Q8_0 (~3.8x menos RAM que F32)\n"
    "      --f32-kv           fuerza KV cache F32 (desactiva el auto-Q8 >1GB)\n"
    "      --max-ram MB       presupuesto RAM: auto-Q8 + baja ctx hasta caber (ej: 2048)\n"
    "      --swap [PATH]      KV cache respaldada en disco (~D: como RAM); sin PATH usa D:\\\n"
    "      --threads N        numero de cores OpenMP (def: todos)\n\n"
    "Ejemplos:\n"
    "  %s run qwen.g2bx \"The capital of France is\" -n 20 -t 0\n"
    "  %s chat qwen.g2bx --no-think -t 0.7\n"
    "  %s bench qwen.g2bx --max-ram 2048\n",
    a0,a0,a0,a0,a0,a0,a0,a0,a0);
}

static int ends_with(const char *s, const char *suf){
  size_t n=strlen(s), m=strlen(suf);
  return n>=m && !strcmp(s+n-m,suf);
}
static int is_g2bx(const char *p){
  return ends_with(p,".g2bx")||ends_with(p,".G2BX")||ends_with(p,".gbin");
}
static int load_any(const char *path, Model *m){
  if(is_g2bx(path)) return model_load_g2bx(path,m);
  return model_load_gguf(path,m);
}
static int cmd_pack(int argc, char **argv){
  if(argc<4){ usage(argv[0]); return 1; }
  int downq4 = (argc>4 && !strcmp(argv[4],"--q4")) || (argc>5 && !strcmp(argv[5],"--q4"));
  return g2bx_pack_ex(argv[2],argv[3],downq4)?1:0;
}

static int cmd_info(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *p=argv[2];
  if(is_g2bx(p)) return g2bx_info(p)?1:0;
  GGUF g; if(gguf_load(p,&g)) return 1;
  char arch[64]="?"; gguf_meta_str(&g,"general.architecture",arch,sizeof arch);
  printf("GGUF ver=%u tensors=%llu arch=%s size=%llu\n",
    g.version,(unsigned long long)g.n_tensors,arch,(unsigned long long)g.size);
  const char *prefs[]={"qwen3","qwen2","llama",NULL};
  for(int i=0;prefs[i];i++){
    char k[80];
    snprintf(k,sizeof k,"%s.embedding_length",prefs[i]);
    i64 d=gguf_meta_i64(&g,k); if(!d) continue;
    snprintf(k,sizeof k,"%s.block_count",prefs[i]); i64 L=gguf_meta_i64(&g,k);
    snprintf(k,sizeof k,"%s.attention.head_count",prefs[i]); i64 h=gguf_meta_i64(&g,k);
    snprintf(k,sizeof k,"%s.attention.head_count_kv",prefs[i]); i64 kv=gguf_meta_i64(&g,k);
    snprintf(k,sizeof k,"%s.attention.key_length",prefs[i]); i64 hd=gguf_meta_i64(&g,k);
    snprintf(k,sizeof k,"%s.rope.freq_base",prefs[i]); f32 th=gguf_meta_f32(&g,k);
    printf("  [%s] dim=%lld layers=%lld heads=%lld kv=%lld head_dim=%lld rope_theta=%g\n",
      prefs[i],(long long)d,(long long)L,(long long)h,(long long)kv,(long long)hd,th);
  }
  int qkn=gguf_by_name(&g,"blk.0.attn_q_norm.weight")!=NULL;
  printf("  qk_norm=%s  output.weight=%s\n",
    qkn?"yes":"no", gguf_by_name(&g,"output.weight")?"yes":"tied");
  for(u64 i=0;i<g.n_tensors && i<8;i++)
    printf("  tensor %s type=%u\n",g.t[i].name,g.t[i].type);
  if(g.n_tensors>8)
    printf("  ... +%llu\n",(unsigned long long)(g.n_tensors-8));
  gguf_free(&g);
  return 0;
}

static i32 tokenize_prefix(Tokenizer *t, const char *text, i32 **out){
  i32 *ids; i32 n=tok_encode(t,text,&ids); *out=ids; return n;
}

/* Ruta por defecto para el swap: D: (disco con espacio) o el temp del sistema. */
static int drive_ok(const char *root){
#if defined(_WIN32)
  UINT r=GetDriveTypeA(root);
  return r!=DRIVE_NO_ROOT_DIR && r!=DRIVE_UNKNOWN;
#else
  (void)root; return 0;
#endif
}
static const char *default_swap(void){
#if defined(_WIN32)
  static char buf[300];
  if(drive_ok("D:\\")) snprintf(buf,sizeof buf,"D:\\gguf2bin2_kv.swap");
  else { if(!GetTempPathA(sizeof buf,buf)) strcpy(buf,"C:\\"); strncat(buf,"gguf2bin2_kv.swap",sizeof buf-1); }
  return buf;
#else
  return "/tmp/gguf2bin2_kv.swap";
#endif
}

/* Aplica opciones de RAM/contexto tras cargar el modelo y reporta el footprint. */
static int apply_ram_opts(Model *m, i32 ctx, u64 max_ram, int q8kv, int f32kv, const char *swap){
  int realloc=0;
  if(f32kv) m->flags &= (u8)~F_KV_Q8;
  else if(q8kv){ m->flags |= F_KV_Q8; realloc=1; }
  else if(max_ram==0 && ctx<=0 && model_kv_bytes(m,0) > ((u64)1<<30)){
    m->flags |= F_KV_Q8; realloc=1;
    fprintf(stderr,"ram: KV en F32 dispararia >1 GB -> activo KV Q8_0 automatico (--f32-kv para forzar)\n");
  }
  if(max_ram>0){
    u64 budget = (u64)max_ram << 20;
    if(model_auto_budget(m,budget)) return -1;
  } else if(ctx>0){
    if(model_set_ctx(m,ctx)) return -1;
  } else if(q8kv||realloc){
    if(model_set_ctx(m, m->ctx>0?m->ctx:m->c.seq_len)) return -1;
  }
  if(swap && *swap){
    const char *p = (!strcmp(swap,"@")) ? default_swap() : swap;
    if(model_enable_swap(m,p))
      fprintf(stderr,"swap: no se pudo usar %s (prueba --q8-kv en su lugar)\n", p);
  }
  model_ram_report(m);
  if(m->c.seq_len && ctx>0 && ctx!=m->ctx)
    fprintf(stderr,"model: contexto efectivo %d (max del modelo %d)\n", m->ctx, m->c.seq_len);
  return 0;
}

/* sampling con top-k, top-p, repeat penalty (penalty se aplica UNA sola vez) */
typedef struct { int id; float logit; } TokLogit;
static int cmp_logit_desc(const void *a, const void *b){
  float fa=((const TokLogit*)a)->logit, fb=((const TokLogit*)b)->logit;
  return (fa<fb)-(fa>fb);
}

static void apply_repeat_penalty(float *logits, int n, float penalty, const i32 *recent, int recent_n){
  if(penalty==1.f || !recent || recent_n<=0) return;
  /* Dedup: mark once so the same id is not penalized multiple times. */
  u8 *seen = calloc((size_t)n, 1);
  if(!seen){
    for(int i=0;i<recent_n;i++){
      int id=recent[i]; if(id<0||id>=n) continue;
      float v=logits[id];
      if(v>0) logits[id]=v/penalty; else logits[id]=v*penalty;
    }
    return;
  }
  for(int i=0;i<recent_n;i++){
    int id=recent[i]; if(id<0||id>=n||seen[id]) continue;
    seen[id]=1;
    float v=logits[id];
    if(v>0) logits[id]=v/penalty; else logits[id]=v*penalty;
  }
  free(seen);
}

static i32 sample_advanced(float *logits, int n, float temp, int top_k, float top_p,
                           float repeat_penalty, const i32 *recent, int recent_n){
  if(n<=0) return 0;
  apply_repeat_penalty(logits, n, repeat_penalty, recent, recent_n);

  if(temp<=0.f){
    int bi=0; float bv=logits[0];
    for(int i=1;i<n;i++) if(logits[i]>bv){ bv=logits[i]; bi=i; }
    return bi;
  }

  for(int i=0;i<n;i++) logits[i]/=temp;
  TokLogit *arr=malloc((size_t)n*sizeof(TokLogit));
  if(!arr){
    int bi=0; float bv=logits[0];
    for(int i=1;i<n;i++) if(logits[i]>bv){ bv=logits[i]; bi=i; }
    return bi;
  }
  for(int i=0;i<n;i++){ arr[i].id=i; arr[i].logit=logits[i]; }
  int keep=n;
  qsort(arr, (size_t)n, sizeof(TokLogit), cmp_logit_desc);
  if(top_k>0 && top_k<n) keep=top_k;
  float maxl=arr[0].logit;
  float sum=0.f;
  for(int i=0;i<keep;i++){ float e=expf(arr[i].logit-maxl); arr[i].logit=e; sum+=e; }
  for(int i=0;i<keep;i++) arr[i].logit/=sum;
  if(top_p<1.f){
    float cumsum=0.f; int last=keep;
    for(int i=0;i<keep;i++){ cumsum+=arr[i].logit; if(cumsum>=top_p){ last=i+1; break; } }
    keep=last;
  }
  if(keep<n && keep>0){
    sum=0.f; for(int i=0;i<keep;i++) sum+=arr[i].logit;
    if(sum>0.f) for(int i=0;i<keep;i++) arr[i].logit/=sum;
  }
  if(keep<=0){ int id=arr[0].id; free(arr); return id; }
  float r=(float)rand()/(float)RAND_MAX;
  float cum=0.f;
  for(int i=0;i<keep;i++){
    cum+=arr[i].logit;
    if(r<=cum){ int id=arr[i].id; free(arr); return id; }
  }
  int id=arr[keep-1].id; free(arr); return id;
}

static int cmd_run(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *path=argv[2];
  i32 n_tok=64; f32 temp=0.7f; int top_k=40; float top_p=0.9f; float rep_pen=1.1f; int use_bos=0;
  i32 ctx=0; int q8kv=0; int f32kv=0; u64 max_ram=0; int nthr=0; const char *swap=NULL;
  i32 prompt[1024]; i32 np=0;
  char text[8192]; text[0]=0;
  for(int i=3;i<argc;i++){
    if(!strcmp(argv[i],"-n")&&i+1<argc) n_tok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthr=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")||!strcmp(argv[i],"--ctx")) { if(i+1<argc) ctx=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--q8-kv")) q8kv=1;
    else if(!strcmp(argv[i],"--f32-kv")) f32kv=1;
    else if(!strcmp(argv[i],"--max-ram")&&i+1<argc) max_ram=(u64)atoll(argv[++i]);
    else if(!strcmp(argv[i],"--swap")){ swap = (i+1<argc && argv[i+1][0]!='-') ? argv[++i] : "@"; }
    else if(!strcmp(argv[i],"-t")&&i+1<argc) temp=(f32)atof(argv[++i]);
    else if(!strcmp(argv[i],"--top-k")&&i+1<argc) top_k=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--top-p")&&i+1<argc) top_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--repeat-penalty")&&i+1<argc) rep_pen=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--bos")) use_bos=1;
    else if(!strcmp(argv[i],"--tokens")&&i+1<argc){
      char *s=argv[++i], *tok;
      for(tok=strtok(s,","); tok&&np<1024; tok=strtok(NULL,",")) prompt[np++]=atoi(tok);
    } else {
      strncat(text,argv[i],sizeof(text)-1-strlen(text));
      strncat(text," ",sizeof(text)-1-strlen(text));
    }
  }
  Model m; if(load_any(path,&m)) return 1;
  if(apply_ram_opts(&m,ctx,max_ram,q8kv,f32kv,swap)){ model_free(&m); return 1; }
  if(nthr>0) set_threads(nthr);
  if(text[0] && m.tok){
    i32 *enc; i32 en=tokenize_prefix(m.tok,text,&enc);
    if(en>0){
      np=0;
      if(use_bos && m.tok->bos>=0) prompt[np++]=m.tok->bos;
      for(i32 i=0;i<en && np<1024;i++) prompt[np++]=enc[i];
    }
    free(enc);
  }
  if(!np) prompt[np++]=1;
  f32 *logits=calloc((size_t)m.c.vocab,sizeof(f32));
  if(!logits){ model_free(&m); return 1; }
  srand((unsigned)time(NULL));
  i32 pos=0;
  RecentBuf recent; memset(&recent,0,sizeof recent);
  i32 rep_tmp[RECENT_USE];

  for(i32 i=0;i<np;i++){
    if(pos >= m.ctx){
      fprintf(stderr,"\nrun: contexto lleno (ctx=%d); truncando prompt\n", m.ctx);
      break;
    }
    if(prompt[i]<0 || prompt[i]>=m.c.vocab){
      fprintf(stderr,"run: token id %d fuera de vocab (%d)\n", prompt[i], m.c.vocab);
      free(logits); model_free(&m); return 1;
    }
    model_forward(&m,prompt[i],pos,logits);
    if(m.tok && !(use_bos && i==0 && prompt[i]==m.tok->bos)){
      i32 one=prompt[i]; char *p=tok_decode(m.tok,&one,1); printf("%s",p); free(p);
    } else if(!m.tok) printf("%d ",prompt[i]);
    recent_push(&recent, prompt[i]);
    pos++;
  }
  for(i32 i=0;i<n_tok;i++){
    if(pos >= m.ctx){
      fprintf(stderr,"\nrun: contexto lleno (ctx=%d); stop\n", m.ctx);
      break;
    }
    float *tmp=(float*)malloc((size_t)m.c.vocab*sizeof(float));
    if(!tmp){ free(logits); model_free(&m); return 1; }
    memcpy(tmp,logits,(size_t)m.c.vocab*sizeof(float));
    int rp_n = recent_snapshot(&recent, rep_tmp, RECENT_USE);
    i32 next=sample_advanced(tmp,m.c.vocab,temp,top_k,top_p,rep_pen,rep_tmp,rp_n);
    free(tmp);
    if(m.tok){ i32 one=next; char *p=tok_decode(m.tok,&one,1); printf("%s",p); free(p); }
    else printf("%d ",next);
    fflush(stdout);
    recent_push(&recent, next);
    model_forward(&m,next,pos,logits); pos++;
  }
  printf("\n"); free(logits); model_free(&m); return 0;
}

static int cmd_synth(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  return exp_synth_qwen_tiny(argv[2])?1:0;
}
static i32 find_tok(Tokenizer *t, const char *s){
  for(i32 i=0;i<t->n;i++) if(t->tok[i]&&!strcmp(t->tok[i],s)) return i;
  return -1;
}

static int cmd_chat(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *path=argv[2];
  i32 n_tok=256; f32 temp=0.7f; int top_k=40; float top_p=0.9f; float rep_pen=1.05f; int no_think=0;
  i32 ctx=0; int q8kv=0; int f32kv=0; u64 max_ram=0; int nthr=0; const char *swap=NULL;
  for(int i=3;i<argc;i++){
    if(!strcmp(argv[i],"-n")&&i+1<argc) n_tok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthr=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")||!strcmp(argv[i],"--ctx")) { if(i+1<argc) ctx=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--q8-kv")) q8kv=1;
    else if(!strcmp(argv[i],"--f32-kv")) f32kv=1;
    else if(!strcmp(argv[i],"--max-ram")&&i+1<argc) max_ram=(u64)atoll(argv[++i]);
    else if(!strcmp(argv[i],"--swap")){ swap = (i+1<argc && argv[i+1][0]!='-') ? argv[++i] : "@"; }
    else if(!strcmp(argv[i],"-t")&&i+1<argc) temp=(f32)atof(argv[++i]);
    else if(!strcmp(argv[i],"--top-k")&&i+1<argc) top_k=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--top-p")&&i+1<argc) top_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--no-think")) no_think=1;
    else if(!strcmp(argv[i],"--repeat-penalty")&&i+1<argc) rep_pen=(float)atof(argv[++i]);
  }
  Model m; if(load_any(path,&m)) return 1;
  if(apply_ram_opts(&m,ctx,max_ram,q8kv,f32kv,swap)){ model_free(&m); return 1; }
  if(nthr>0) set_threads(nthr);
  if(!m.tok){ fprintf(stderr,"chat: el modelo no trae tokenizer. Re-empaqueta\n"); model_free(&m); return 1; }
  Tokenizer *tk=m.tok;
  i32 im_start=find_tok(tk,"<|im_start|>"), im_end=find_tok(tk,"<|im_end|>");
  i32 think_start=find_tok(tk,"<think>"), think_end=find_tok(tk,"</think>");
  i32 eos=(tk->eos>=0)?tk->eos:im_end;
  if(im_start<0||im_end<0){ fprintf(stderr,"chat: faltan tokens especiales\n"); model_free(&m); return 1; }
  f32 *logits=calloc((size_t)m.c.vocab,sizeof(f32));
  if(!logits){ model_free(&m); return 1; }
  srand((unsigned)time(NULL));
  i32 *conv=NULL; i32 cn=0, ccap=2048; conv=malloc((size_t)ccap*sizeof(i32));
  if(!conv){ free(logits); model_free(&m); return 1; }
#define PUSH(x) do{ if(cn>=ccap){ ccap*=2; conv=realloc(conv,(size_t)ccap*sizeof(i32)); if(!conv){ fprintf(stderr,"chat: OOM\n"); free(logits); model_free(&m); return 1; } } conv[cn++]=(x); }while(0)
  {
    i32 *st; i32 sn=tok_encode(tk,"system\nYou are a helpful assistant.",&st);
    PUSH(im_start); for(i32 i=0;i<sn;i++) PUSH(st[i]); PUSH(im_end);
    i32 *nl; i32 nnl=tok_encode(tk,"\n",&nl); for(i32 i=0;i<nnl;i++) PUSH(nl[i]); free(nl); free(st);
  }
  i32 pos=0, fed=0;
  char line[8192];
  printf("=== Chat Qwen3 %s (escribe 'salir') ===\n", no_think?"[no-think]":"[thinking]");
  for(;;){
    printf("\nTu> "); fflush(stdout);
    if(!fgets(line,sizeof line,stdin)) break;
    size_t ll=strlen(line); while(ll>0 && (line[ll-1]=='\n'||line[ll-1]=='\r')) line[--ll]=0;
    if(!strcmp(line,"salir")||!strcmp(line,"exit")) break;
    if(ll==0) continue;
    char ub[9000]; snprintf(ub,sizeof ub,"user\n%s",line);
    i32 *ut; i32 un=tok_encode(tk,ub,&ut);
    i32 *at; i32 an=tok_encode(tk,"assistant\n",&at);
    PUSH(im_start); for(i32 i=0;i<un;i++) PUSH(ut[i]); PUSH(im_end);
    i32 *nl2; i32 nnl2=tok_encode(tk,"\n",&nl2); for(i32 i=0;i<nnl2;i++) PUSH(nl2[i]); free(nl2);
    PUSH(im_start); for(i32 i=0;i<an;i++) PUSH(at[i]);
    if(no_think && think_start>=0 && think_end>=0){
      PUSH(think_start);
      i32 *nn1; i32 n1=tok_encode(tk,"\n\n",&nn1); for(i32 i=0;i<n1;i++) PUSH(nn1[i]); free(nn1);
      PUSH(think_end);
      i32 *nn2; i32 n2=tok_encode(tk,"\n\n",&nn2); for(i32 i=0;i<n2;i++) PUSH(nn2[i]); free(nn2);
    }
    free(ut); free(at);
    for(; fed<cn; fed++){
      if(pos >= m.ctx){
        fprintf(stderr,"chat: contexto lleno (ctx=%d)\n", m.ctx);
        break;
      }
      model_forward(&m,conv[fed],pos++,logits);
    }
    if(pos >= m.ctx){ printf("\n"); continue; }
    printf("Qwen3> "); fflush(stdout);
    RecentBuf recent; memset(&recent,0,sizeof recent);
    i32 rep_tmp[RECENT_USE];
    for(i32 step=0; step<n_tok; step++){
      if(pos >= m.ctx){
        fprintf(stderr,"\nchat: contexto lleno; stop\n");
        break;
      }
      float *tmp=malloc((size_t)m.c.vocab*sizeof(float));
      if(!tmp) break;
      memcpy(tmp,logits,(size_t)m.c.vocab*sizeof(float));
      int rp_n = recent_snapshot(&recent, rep_tmp, RECENT_USE);
      i32 nxt=sample_advanced(tmp,m.c.vocab,temp,top_k,top_p,rep_pen,rep_tmp,rp_n);
      free(tmp);
      if(nxt==eos || nxt==im_end) break;
      i32 one=nxt; char *piece=tok_decode(tk,&one,1); printf("%s",piece); fflush(stdout); free(piece);
      recent_push(&recent, nxt);
      PUSH(nxt);
      model_forward(&m,nxt,pos++,logits);
    }
    printf("\n");
    if(cn>0 && conv[cn-1]!=im_end) PUSH(im_end);
    i32 *nl3; i32 nnl3=tok_encode(tk,"\n",&nl3); for(i32 i=0;i<nnl3;i++) PUSH(nl3[i]); free(nl3);
  }
#undef PUSH
  free(conv); free(logits); model_free(&m); return 0;
}

static double now_sec(void){
#if defined(_WIN32)
  static double inv=0;
  LARGE_INTEGER c,f;
  if(inv==0){ QueryPerformanceFrequency(&f); inv=1.0/(double)f.QuadPart; }
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart*inv;
#else
  return (double)clock()/CLOCKS_PER_SEC;
#endif
}

static int cmd_bench(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  i32 n=32; i32 ctx=0; int q8kv=0; int f32kv=0; u64 max_ram=0; int nthr=0; const char *swap=NULL;
  for(int i=3;i<argc;i++){
    if(!strcmp(argv[i],"-n")&&i+1<argc) n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthr=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")||!strcmp(argv[i],"--ctx")) { if(i+1<argc) ctx=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--q8-kv")) q8kv=1;
    else if(!strcmp(argv[i],"--f32-kv")) f32kv=1;
    else if(!strcmp(argv[i],"--max-ram")&&i+1<argc) max_ram=(u64)atoll(argv[++i]);
    else if(!strcmp(argv[i],"--swap")){ swap = (i+1<argc && argv[i+1][0]!='-') ? argv[++i] : "@"; }
  }
  Model m; if(load_any(argv[2],&m)) return 1;
  if(apply_ram_opts(&m,ctx,max_ram,q8kv,f32kv,swap)){ model_free(&m); return 1; }
  if(nthr>0) set_threads(nthr);
  f32 *logits=calloc((size_t)m.c.vocab,sizeof(f32));
  if(!logits){ model_free(&m); return 1; }
  model_forward(&m,1,0,logits);
  double t0=now_sec();
  for(i32 i=0;i<n;i++)
    model_forward(&m, (i32)(i % (m.c.vocab>1?m.c.vocab:1)), (i32)(i % (m.ctx>0?m.ctx:m.c.seq_len)), logits);
  double sec=now_sec()-t0; if(sec<1e-9) sec=1e-9;
  fprintf(stderr,"bench: %d tokens in %.4fs -> %.1f tok/s  (dim=%d L=%d hd=%d ctx=%d)\n",
    n,sec,n/sec,m.c.dim,m.c.n_layers,m.c.head_dim,m.ctx);
  free(logits); model_free(&m); return 0;
}

int main(int argc, char **argv){
  if(argc<2){ usage(argv[0]); return 1; }
  if(!strcmp(argv[1],"pack"))  return cmd_pack(argc,argv);
  if(!strcmp(argv[1],"info"))  return cmd_info(argc,argv);
  if(!strcmp(argv[1],"run"))   return cmd_run(argc,argv);
  if(!strcmp(argv[1],"synth")) return cmd_synth(argc,argv);
  if(!strcmp(argv[1],"bench")) return cmd_bench(argc,argv);
  if(!strcmp(argv[1],"chat"))  return cmd_chat(argc,argv);
  usage(argv[0]); return 1;
}
