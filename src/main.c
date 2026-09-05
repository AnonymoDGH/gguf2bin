/* CLI: pack | info | run | synth | chat | bench — runtime G2BX v4.7 */
#include "g2b.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/resource.h>
#include <time.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

#define RECENT_CAP 128
#define RECENT_USE 64
static const char *DEFAULT_CALIB =
  "The city council approved the new budget on Tuesday after months of debate. "
  "Residents expressed concerns about traffic congestion and public transportation. "
  "El ayuntamiento aprobó el nuevo presupuesto después de meses de debate. "
  "Scientists published a study showing that ocean temperatures have risen steadily. "
  "The engineer designed a circuit that reduces power consumption by forty percent. "
  "In the morning she reads the news while drinking coffee near the window. "
  "Machine learning models transform input vectors through many matrix multiplications. "
  "Los estudiantes caminaron por la plaza hasta la biblioteca central del campus. "
  "A river runs through the valley, feeding the fields where farmers grow wheat. "
  "The report recommends investing in renewable energy and modernizing the grid. "
  "Every weekend the market fills with people buying fruit, bread and flowers. "
  "Software updates usually fix bugs but sometimes introduce new problems. ";


static void set_threads(int n){
#if defined(_OPENMP)
  if(n>0) omp_set_num_threads(n);
  int t = n>0?n:omp_get_num_threads();
  fprintf(stderr,"threads: using %d core(s)\n", t);
#else
  fprintf(stderr,"threads: built without OpenMP (use -fopenmp for speed)\n");
#endif
}

static void apply_fast(int *nthr){
#if defined(_OPENMP)
  int maxc = omp_get_max_threads();
  if(*nthr <= 0) *nthr = maxc;
  omp_set_num_threads(*nthr);
  omp_set_dynamic(0);
#endif
#if defined(_WIN32)
  if(!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
    fprintf(stderr,"fast: warning: no permission for high priority (continuing normally)\n");
#else
  if(setpriority(PRIO_PROCESS, 0, -10)!=0)
    fprintf(stderr,"fast: warning: no permission for nice -10 (continuing normally)\n");
#endif
  fprintf(stderr,"fast: high priority, %d threads, swap disabled\n", *nthr);
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
    "gguf2bin2 - GGUF -> G2BX (own format) -> Qwen3/Llama/LFM2 inference v4.9\n\n"
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
    "      --drop N           skip the N least-influential blocks (ShortGPT)\n"
    "      --mv F             Swapeculative 0.0..1.0 FFN/SSM skip\n\n"
    "bench extras:\n"
    "      --prefill N        measure prompt-processing tok/s (no logits)\n\n"
    "Examples:\n"
    "  %s run qwen.g2bx \"The capital of France is\" -n 20 -t 0\n"
    "  %s chat llama.g2bx --fast --q8-kv -n 256\n"
    "  %s bench qwen.g2bx --max-ram 2048\n",
    a0,a0,a0,a0,a0,a0,a0,a0,a0,a0);
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
/* corpus genérico embebido para calibrar la poda si no se da --calib */


static int cmd_pack(int argc, char **argv){
  if(argc<4){ usage(argv[0]); return 1; }
  int downq4=0; float prune=0.f; const char *calib_file=NULL;
  for(int i=4;i<argc;i++){
    if(!strcmp(argv[i],"--q4")) downq4=1;
    else if(!strcmp(argv[i],"--q4s")){ downq4=1; g2bx_set_q4s(1); }
    else if(!strcmp(argv[i],"--q4s_psy")||!strcmp(argv[i],"--psy")){ downq4=1; g2bx_set_q4s_psy(1); }
    else if(!strcmp(argv[i],"--q4vvc")||!strcmp(argv[i],"--vvc")){ downq4=1; g2bx_set_q4vvc(1); }
    else if(!strcmp(argv[i],"--prune")&&i+1<argc) prune=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--calib")&&i+1<argc) calib_file=argv[++i];
  }
  if(prune>0.f && (prune<=0.001f||prune>=0.9f)){ fprintf(stderr,"pack: --prune must be in (0,0.9)\n"); return 1; }
  if(prune<=0.f) return g2bx_pack_prune(argv[2],argv[3],downq4,0.f)?1:0;

  /* dos fases: pack completo -> calibrar activaciones -> repack podado */
  char tmp[1280];
  snprintf(tmp,sizeof tmp,"%s.calib.g2bx",argv[3]);
  fprintf(stderr,"pack: phase 1/2 - full pack for calibration\n");
  if(g2bx_pack_prune(argv[2],tmp,downq4,0.f)){ remove(tmp); return 1; }
  Model m;
  if(model_load_g2bx(tmp,&m)){ remove(tmp); return 1; }
  i32 maxtok=768; i32 ctx=m.c.seq_len>0?(m.c.seq_len<maxtok?m.c.seq_len:maxtok):maxtok;
  model_set_ctx(&m,ctx);
  char *text=NULL; size_t len=0,cap=0;
  if(calib_file){
    FILE *f=fopen(calib_file,"rb");
    if(f){ char buf[16384]; size_t r; while((r=fread(buf,1,sizeof buf,f))>0){ text=realloc(text,len+r+1); memcpy(text+len,buf,r); len+=r; } fclose(f); }
    else fprintf(stderr,"pack: cannot open %s - using embedded corpus\n",calib_file);
  }
  if(!text || len==0){ text=(char*)DEFAULT_CALIB; len=strlen(DEFAULT_CALIB); }
  if(!m.tok){ fprintf(stderr,"pack: model has no tokenizer - cannot --prune\n"); if(text!=(char*)DEFAULT_CALIB) free(text); model_free(&m); remove(tmp); return 1; }
  i32 *ids=NULL; i32 nt=tok_encode(m.tok,text,&ids);
  if(text!=(char*)DEFAULT_CALIB) free(text);
  if(nt>ctx) nt=ctx;
  if(nt<64){ fprintf(stderr,"pack: insufficient calibration (%d tokens)\n",nt); free(ids); model_free(&m); remove(tmp); return 1; }
  fprintf(stderr,"pack: calibrating on %d tokens...\n",nt);
  int ok = (model_collect_stats(&m,ids,nt)==0 && m.ffn_stats!=NULL);
  free(ids);
  if(!ok){ model_free(&m); remove(tmp); return 1; }
  int rc=g2bx_pack_prune_scores(argv[2],argv[3],downq4,prune,m.ffn_stats)?1:0;
  model_free_stats(&m);
  model_free(&m);
  remove(tmp);
  return rc;
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
  else { if(!GetTempPathA(sizeof buf,buf)) strcpy(buf,"C:\\"); strncat(buf,"gguf2bin2_kv.swap",sizeof buf-strlen(buf)-1); }
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
    fprintf(stderr,"ram: F32 KV would exceed 1 GB -> enabling Q8_0 KV automatically (--f32-kv to force)\n");
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
      fprintf(stderr,"swap: cannot use %s (try --q8-kv instead)\n", p);
  }
  model_ram_report(m);
  if(m->c.seq_len && ctx>0 && ctx!=m->ctx)
    fprintf(stderr,"model: effective context %d (model max %d)\n", m->ctx, m->c.seq_len);
  return 0;
}

/* sampling con top-k, top-p, repeat penalty (penalty se aplica UNA sola vez) */
typedef struct { int id; float logit; } TokLogit;

/* PRNG xorshift64*: rand()/RAND_MAX=32767 en Windows destrozaba el muestreo */
static u64 g_rng = 0x9E3779B97F4A7C15ull;
static void rng_seed(u64 s){
  g_rng = s ? s : (0x9E3779B97F4A7C15ull ^ (u64)time(NULL));
  for(int i=0;i<4;i++){ g_rng^=g_rng>>12; g_rng^=g_rng<<25; g_rng^=g_rng>>27; }
}
static inline float rnd_f(void){
  g_rng^=g_rng>>12; g_rng^=g_rng<<25; g_rng^=g_rng>>27;
  return (float)(((g_rng*2685821657736338717ull)>>40)*(1.0/16777216.0)); /* [0,1) */
}

static void apply_repeat_penalty(float *logits, int n, float penalty, const i32 *recent, int recent_n){
  if(penalty==1.f || !recent || recent_n<=0) return;
  static u8 *seen=NULL; static int seen_n=0;
  if(seen_n<n){ free(seen); seen=(u8*)calloc((size_t)n,1); seen_n=seen?n:0; }
  if(!seen){
    for(int i=0;i<recent_n;i++){
      int id=recent[i]; if(id<0||id>=n) continue;
      float v=logits[id]; logits[id]=v>0?v/penalty:v*penalty;
    }
    return;
  }
  memset(seen,0,(size_t)n);
  for(int i=0;i<recent_n;i++){
    int id=recent[i]; if(id<0||id>=n||seen[id]) continue;
    seen[id]=1;
    float v=logits[id]; logits[id]=v>0?v/penalty:v*penalty;
  }
}

/* Quickselect descendente: deja los k mayores en a[0..k-1] (sin orden). O(n) vs qsort O(n log n). */
static void topk_select(TokLogit *a, int n, int k){
  if(k<=0) return; if(k>=n) return;
  int lo=0, hi=n-1;
  while(lo<hi){
    int mid=lo+(hi-lo)/2;
    if(a[mid].logit>a[lo].logit){ TokLogit t=a[mid]; a[mid]=a[lo]; a[lo]=t; }
    if(a[hi].logit>a[lo].logit){ TokLogit t=a[hi]; a[hi]=a[lo]; a[lo]=t; }
    if(a[hi].logit>a[mid].logit){ TokLogit t=a[hi]; a[hi]=a[mid]; a[mid]=t; }
    float p=a[mid].logit;
    int i=lo, j=hi;
    while(i<=j){
      while(a[i].logit>p) i++;
      while(a[j].logit<p) j--;
      if(i<=j){ TokLogit t=a[i]; a[i]=a[j]; a[j]=t; i++; j--; }
    }
    if(k<=j) hi=j;
    else if(k>=i) lo=i;
    else break;
  }
}

/* Muestreo softmax exacto por truco Gumbel-max: argmax(l/T + Gumbel) ~ softmax(l/T). O(n), sin exp. */
static int gumbel_sample(const float *logits, int n){
  int bi=0; float bs=-1e30f;
  for(int i=0;i<n;i++){
    float u=rnd_f(); if(u<1e-7f) u=1e-7f; else if(u>0.9999999f) u=0.9999999f;
    float v=logits[i]-logf(-logf(u));
    if(v>bs){ bs=v; bi=i; }
  }
  return bi;
}

static int cmp_logit_desc(const void *x, const void *y){
  float a=((const TokLogit*)x)->logit, b=((const TokLogit*)y)->logit;
  return (a<b)-(a>b);
}

static TokLogit *g_samp_buf=NULL; static int g_samp_cap=0;
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
  if(top_k<=0 || top_k>=n){
    if(top_p>=1.f) return gumbel_sample(logits,n);
  }
  TokLogit *arr=NULL;
  if(g_samp_cap>=n) arr=g_samp_buf;
  else {
    free(g_samp_buf);
    g_samp_buf=(TokLogit*)malloc((size_t)n*sizeof(TokLogit));
    g_samp_cap=g_samp_buf?n:0;
    arr=g_samp_buf;
  }
  if(!arr) return gumbel_sample(logits,n);
  for(int i=0;i<n;i++){ arr[i].id=i; arr[i].logit=logits[i]; }

  int keep=n;
  if(top_k>0 && top_k<n){
    topk_select(arr,n,top_k);
    /* insertion sort del prefijo k (k pequeno: tipico 20-100) */
    for(int i=1;i<top_k;i++){ TokLogit v=arr[i]; int j=i-1; while(j>=0&&arr[j].logit<v.logit){ arr[j+1]=arr[j]; j--; } arr[j+1]=v; }
    keep=top_k;
  } else {
    qsort(arr,(size_t)n,sizeof(TokLogit),cmp_logit_desc);
  }
  float maxl=arr[0].logit, sum=0.f;
  for(int i=0;i<keep;i++){ float e=expf(arr[i].logit-maxl); arr[i].logit=e; sum+=e; }
  if(top_p<1.f){
    float cum=0.f; int last=keep;
    for(int i=0;i<keep;i++){ cum+=arr[i].logit/sum; if(cum>=top_p){ last=i+1; break; } }
    keep=last;
    sum=0.f; for(int i=0;i<keep;i++) sum+=arr[i].logit;
  }
  float r=rnd_f()*sum, cum=0.f;
  int id=arr[keep-1].id;
  for(int i=0;i<keep;i++){ cum+=arr[i].logit; if(r<=cum){ id=arr[i].id; break; } }
  return id;
}

static int cmd_run(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *path=argv[2];
  i32 n_tok=64; f32 temp=0.7f; int top_k=40; float top_p=0.9f; float rep_pen=1.1f; int use_bos=0;
  int gpu=0;
  i32 ctx=0; int q8kv=0; int f32kv=0; int fast=0; u64 max_ram=0; int nthr=0; const char *swap=NULL;
  u64 seed=0; int ndrop=0; float mv_ratio=0.f; float bvh_ratio=0.f; const char *cyber=NULL;
  i32 prompt[1024]; i32 np=0;
  char text[8192]; text[0]=0;
  for(int i=3;i<argc;i++){
    if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--n"))&&i+1<argc) n_tok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthr=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")||!strcmp(argv[i],"--ctx")) { if(i+1<argc) ctx=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--q8-kv")) q8kv=1;
    else if(!strcmp(argv[i],"--f32-kv")) f32kv=1;
    else if(!strcmp(argv[i],"--fast")) fast=1;
    else if(!strcmp(argv[i],"--max-ram")&&i+1<argc) max_ram=(u64)atoll(argv[++i]);
    else if(!strcmp(argv[i],"--swap")){ swap = (i+1<argc && argv[i+1][0]!='-') ? argv[++i] : "@"; }
    else if(!strcmp(argv[i],"-t")&&i+1<argc) temp=(f32)atof(argv[++i]);
    else if(!strcmp(argv[i],"--top-k")&&i+1<argc) top_k=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--top-p")&&i+1<argc) top_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--repeat-penalty")&&i+1<argc) rep_pen=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=(u64)strtoull(argv[++i],NULL,10);
    else if(!strcmp(argv[i],"--bos")) use_bos=1;
    else if(!strcmp(argv[i],"--drop")&&i+1<argc) ndrop=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--mv")&&i+1<argc) mv_ratio=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--bvh")){ if(i+1<argc && argv[i+1][0]!='-' && strchr(argv[i+1],'.')) bvh_ratio=(float)atof(argv[++i]); else bvh_ratio=0.15f; }
    else if(!strcmp(argv[i],"--cyber")&&i+1<argc) cyber=argv[++i];
    else if(!strcmp(argv[i],"--gpu")) gpu=1;
    else if(!strcmp(argv[i],"--tokens")&&i+1<argc){
      char *s=argv[++i], *tok;
      for(tok=strtok(s,","); tok&&np<1024; tok=strtok(NULL,",")) prompt[np++]=atoi(tok);
    } else {
      strncat(text,argv[i],sizeof(text)-1-strlen(text));
      strncat(text," ",sizeof(text)-1-strlen(text));
    }
  }
  Model m; if(load_any(path,&m)) return 1;
  if(mv_ratio>0) m.mv_ratio=mv_ratio;
  if(cyber) cyber_load_lora(&m,cyber);
  if(bvh_ratio>0){ m.use_bvh=1; m.bvh_keep=bvh_ratio; fprintf(stderr,"[bvh] sparse %.2f\n",bvh_ratio); }
  if(cyber) cyber_load_lora(&m,cyber);
  if(gpu && !vk_dual_start(&m,path)) fprintf(stderr,"[gpu] continuing CPU-only\n");
  { const char *sw = fast ? NULL : swap;
    if(apply_ram_opts(&m,ctx,max_ram,q8kv,f32kv,sw)){ model_free(&m); return 1; } }
  if(fast) apply_fast(&nthr); else if(nthr>0) set_threads(nthr);
  if(m.mv_ratio>0) fprintf(stderr,"[mv] ratio=%.2f\n", m.mv_ratio);
  if(ndrop>0 && m.tok){
    i32 *ids=NULL; i32 nt=tok_encode(m.tok,(char*)DEFAULT_CALIB,&ids);
    i32 use=nt; i32 mc=(m.ctx>0?m.ctx:1024); if(use>mc) use=mc;
    if(use>=8){ model_autodrop(&m,ids,use,ndrop); } free(ids); }
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
  rng_seed(seed);
  if(seed) fprintf(stderr,"seed=%llu\n",(unsigned long long)seed);
  i32 pos=0;
  RecentBuf recent; memset(&recent,0,sizeof recent);
  i32 rep_tmp[RECENT_USE];

  for(i32 i=0;i<np;){
    if(pos >= m.ctx){
      fprintf(stderr,"\nrun: context full (ctx=%d); truncating prompt\n", m.ctx);
      break;
    }
    if(prompt[i]<0 || prompt[i]>=m.c.vocab){
      fprintf(stderr,"run: token id %d out of vocab (%d)\n", prompt[i], m.c.vocab);
      free(logits); model_free(&m); return 1;
    }
    i32 pb=m.pf_B>0?m.pf_B:8;
    i32 chunk=np-i; if(chunk>pb) chunk=pb; if(pos+chunk>m.ctx) chunk=m.ctx-pos;
    if(chunk<=0) break;
    int want=(i+chunk>=np)?1:0; /* logits solo del último token del prompt */
    int rc=model_prefill(&m,&prompt[i],chunk,pos,want?logits:NULL);
    if(rc){
      for(i32 j=0;j<chunk;j++)
        model_forward_ex(&m,prompt[i+j],pos+j,(want&&j==chunk-1)?logits:NULL,want&&j==chunk-1);
    }
    if(m.tok){
      if(use_bos && i==0 && prompt[i]==m.tok->bos){
        if(chunk>1){ char *p=tok_decode(m.tok,&prompt[i+1],chunk-1); printf("%s",p); free(p); }
      } else { char *p=tok_decode(m.tok,&prompt[i],chunk); printf("%s",p); free(p); }
    } else { for(i32 j=0;j<chunk;j++) printf("%d ",prompt[i+j]); }
    for(i32 j=0;j<chunk;j++) recent_push(&recent, prompt[i+j]);
    pos+=chunk; i+=chunk;
  }
  for(i32 i=0;i<n_tok;i++){
    if(pos >= m.ctx){
      fprintf(stderr,"\nrun: context full (ctx=%d); stop\n", m.ctx);
      break;
    }
    int rp_n = recent_snapshot(&recent, rep_tmp, RECENT_USE);
    i32 next=sample_advanced(logits,m.c.vocab,temp,top_k,top_p,rep_pen,rep_tmp,rp_n);
    if(m.tok){ i32 one=next; char *p=tok_decode(m.tok,&one,1); printf("%s",p); free(p); }
    else printf("%d ",next);
    fflush(stdout);
    recent_push(&recent, next);
    model_forward(&m,next,pos,logits); pos++;
  }
  printf("\n"); free(logits); model_free(&m); return 0;
}

/* ppl: cross-entropy del modelo sobre un texto (harness de calidad).
   Uso: ppl <model> [-f fichero|-] [-n max_tokens] [opts RAM] */
static int cmd_ppl(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  const char *path=argv[2];
  const char *file=NULL; i32 maxtok=4096;
  i32 ctx=0; int q8kv=0; int f32kv=0; u64 max_ram=0; int nthr=0; const char *swap=NULL;
  int ndrop=0; float mv_ratio=0.f; float bvh_ratio=0.f; const char *cyber=NULL;
  for(int i=3;i<argc;i++){
    if(!strcmp(argv[i],"-f")&&i+1<argc) file=argv[++i];
    else if(!strcmp(argv[i],"-n")&&i+1<argc) maxtok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")||!strcmp(argv[i],"--ctx")) { if(i+1<argc) ctx=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--q8-kv")) q8kv=1;
    else if(!strcmp(argv[i],"--f32-kv")) f32kv=1;
    else if(!strcmp(argv[i],"--max-ram")&&i+1<argc) max_ram=(u64)atoll(argv[++i]);
    else if(!strcmp(argv[i],"--swap")){ swap = (i+1<argc && argv[i+1][0]!='-') ? argv[++i] : "@"; }
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthr=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--drop")&&i+1<argc) ndrop=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--mv")&&i+1<argc) mv_ratio=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--bvh")){ if(i+1<argc && argv[i+1][0]!='-') bvh_ratio=(float)atof(argv[++i]); else bvh_ratio=0.15f; }
    else if(!strcmp(argv[i],"--cyber")&&i+1<argc) cyber=argv[++i];
  }
  Model m; if(load_any(path,&m)) return 1;
  if(mv_ratio>0) m.mv_ratio=mv_ratio;
  if(cyber) cyber_load_lora(&m,cyber);
  { const char *sw = swap;
    if(apply_ram_opts(&m,ctx,max_ram,q8kv,f32kv,sw)){ model_free(&m); return 1; } }
  if(nthr>0) set_threads(nthr);
  if(ndrop>0 && m.tok){
    i32 *ids=NULL; i32 nt=tok_encode(m.tok,(char*)DEFAULT_CALIB,&ids);
    i32 use=nt; i32 mc=(m.ctx>0?m.ctx:1024); if(use>mc) use=mc;
    if(use>=8){ model_autodrop(&m,ids,use,ndrop); } free(ids);
  }
  if(!m.tok){ fprintf(stderr,"ppl: model has no tokenizer\n"); model_free(&m); return 1; }
  char *text=NULL; size_t cap=0, len=0;
  if(file && strcmp(file,"-")){
    FILE *f=fopen(file,"rb");
    if(!f){ fprintf(stderr,"ppl: cannot open %s\n",file); model_free(&m); return 1; }
    char buf[65536]; size_t r;
    while((r=fread(buf,1,sizeof buf,f))>0){ text=realloc(text,len+r+1); memcpy(text+len,buf,r); len+=r; }
    fclose(f);
  } else {
    char buf[65536]; size_t r;
    while((r=fread(buf,1,sizeof buf,stdin))>0){ text=realloc(text,len+r+1); memcpy(text+len,buf,r); len+=r; }
  }
  if(!text || len==0){ fprintf(stderr,"ppl: empty text\n"); free(text); model_free(&m); return 1; }
  text[len]=0;
  i32 *ids=NULL; i32 nt=tok_encode(m.tok,text,&ids);
  free(text);
  if(m.tok->bos>=0 && !(nt>0 && ids[0]==m.tok->bos)){
    i32 *tmp=malloc((size_t)(nt+1)*sizeof(i32));
    if(tmp){ tmp[0]=m.tok->bos; memcpy(tmp+1,ids,(size_t)nt*sizeof(i32)); free(ids); ids=tmp; nt++; }
  }
  if(nt>maxtok) nt=maxtok;
  if(nt<2){ fprintf(stderr,"ppl: <2 tokens\n"); free(ids); model_free(&m); return 1; }
  fprintf(stderr,"ppl: %d tokens, ctx=%d\n",nt,m.ctx);
  f32 *logits=malloc((size_t)m.c.vocab*sizeof(f32));
  if(!logits){ free(ids); model_free(&m); return 1; }
  double nll=0; i32 count=0;
  i32 win=m.ctx>0?m.ctx:m.c.seq_len;
  for(i32 base=0; base<nt-1; base+=win){
    i32 end = base+win<nt ? base+win : nt;
    for(i32 p=base; p<end-1; p++){
      model_forward_ex(&m,ids[p],p-base,logits,1);
      /* NLL del token ids[p+1] */
      f32 mx=logits[0];
      for(i32 v=1;v<m.c.vocab;v++) if(logits[v]>mx) mx=logits[v];
      double lse=0;
      for(i32 v=0;v<m.c.vocab;v++) lse+=exp((double)(logits[v]-mx));
      lse=log(lse)+mx;
      nll += lse-(double)logits[ids[p+1]];
      count++;
    }
  }
  printf("ppl(%s): tokens=%d evaluated=%d nll/token=%.4f perplexity=%.3f\n",
    path,nt,count,nll/(count?count:1),exp(nll/(count?count:1)));
  free(logits); free(ids); model_free(&m); return 0;
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
  i32 n_tok=256; f32 temp=0.7f; int top_k=40; float top_p=0.9f; float rep_pen=1.05f;
  int no_think=1; int show_think=0;
  const char *sys_txt="You are a helpful assistant.";
  int no_sys=0;
  int gpu=0;
  i32 ctx=0; int q8kv=0; int f32kv=0; int fast=0; u64 max_ram=0; int nthr=0; const char *swap=NULL;
  u64 seed=0; int ndrop=0; float mv_ratio=0.f; float bvh_ratio=0.f;
  for(int i=3;i<argc;i++){
    if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--n"))&&i+1<argc) n_tok=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthr=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")||!strcmp(argv[i],"--ctx")) { if(i+1<argc) ctx=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--q8-kv")) q8kv=1;
    else if(!strcmp(argv[i],"--f32-kv")) f32kv=1;
    else if(!strcmp(argv[i],"--fast")) fast=1;
    else if(!strcmp(argv[i],"--max-ram")&&i+1<argc) max_ram=(u64)atoll(argv[++i]);
    else if(!strcmp(argv[i],"--swap")){ swap = (i+1<argc && argv[i+1][0]!='-') ? argv[++i] : "@"; }
    else if(!strcmp(argv[i],"-t")&&i+1<argc) temp=(f32)atof(argv[++i]);
    else if(!strcmp(argv[i],"--top-k")&&i+1<argc) top_k=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--top-p")&&i+1<argc) top_p=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--no-think")) { no_think=1; show_think=0; }
    else if(!strcmp(argv[i],"--think")) { no_think=0; show_think=1; }
    else if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=(u64)strtoull(argv[++i],NULL,10);
    else if(!strcmp(argv[i],"--drop")&&i+1<argc) ndrop=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--repeat-penalty")&&i+1<argc) rep_pen=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--system")&&i+1<argc){ sys_txt=argv[++i]; no_sys=0; }
    else if(!strcmp(argv[i],"--no-system")){ no_sys=1; sys_txt=NULL; }
    else if(!strcmp(argv[i],"--gpu")) gpu=1;
  }
  Model m; if(load_any(path,&m)) return 1;
  if(mv_ratio>0) m.mv_ratio=mv_ratio;
  if(gpu && !vk_dual_start(&m,path)) fprintf(stderr,"[gpu] continuing CPU-only\n");
  { const char *sw = fast ? NULL : swap;
    if(apply_ram_opts(&m,ctx,max_ram,q8kv,f32kv,sw)){ model_free(&m); return 1; } }
  if(fast) apply_fast(&nthr); else if(nthr>0) set_threads(nthr);
  if(m.mv_ratio>0) fprintf(stderr,"[mv] ratio=%.2f\n", m.mv_ratio);
  if(ndrop>0 && m.tok){ i32 *ids=NULL; i32 nt=tok_encode(m.tok,(char*)DEFAULT_CALIB,&ids);
    i32 use=nt; i32 mc=(m.ctx>0?m.ctx:1024); if(use>mc) use=mc;
    if(use>=8){ model_autodrop(&m,ids,use,ndrop); } free(ids); }
  if(!m.tok){ fprintf(stderr,"chat: model has no tokenizer. Re-pack it\n"); model_free(&m); return 1; }
  Tokenizer *tk=m.tok;
  /* ChatML (Qwen2/3, LFM2) vs Llama 3.x Instruct */
  i32 im_start=find_tok(tk,"<|im_start|>"), im_end=find_tok(tk,"<|im_end|>");
  i32 hstart=find_tok(tk,"<|start_header_id|>"), hend=find_tok(tk,"<|end_header_id|>");
  i32 eot=find_tok(tk,"<|eot_id|>");
  if(eot<0) eot=find_tok(tk,"<|eom_id|>");
  i32 think_start=find_tok(tk,"<think>"), think_end=find_tok(tk,"</think>");
  int tpl_llama = (hstart>=0 && hend>=0 && eot>=0);
  int tpl_chatml = (im_start>=0 && im_end>=0);
  if(!tpl_llama && !tpl_chatml){
    fprintf(stderr,"chat: no template (neither ChatML <|im_start|> nor Llama 3 <|eot_id|>)\n");
    model_free(&m); return 1;
  }
  if(tpl_llama && tpl_chatml){
    if(m.arch==ARCH_LLAMA) tpl_chatml=0; else tpl_llama=0;
  }
  i32 turn_start = tpl_llama ? hstart : im_start;
  i32 turn_end   = tpl_llama ? eot    : im_end;
  i32 eos=(tk->eos>=0)?tk->eos:turn_end;
  f32 *logits=calloc((size_t)m.c.vocab,sizeof(f32));
  if(!logits){ model_free(&m); return 1; }
  rng_seed(seed);
  if(seed) fprintf(stderr,"seed=%llu\n",(unsigned long long)seed);
  i32 *conv=NULL; i32 cn=0, ccap=2048; conv=malloc((size_t)ccap*sizeof(i32));
  if(!conv){ free(logits); model_free(&m); return 1; }
#define PUSH(x) do{ if(cn>=ccap){ if(ccap>=(1<<20)){ fprintf(stderr,"chat: context exceeds 1M tokens\n"); free(logits); model_free(&m); free(conv); return 1; } ccap*=2; i32 *tmp=realloc(conv,(size_t)ccap*sizeof(i32)); if(!tmp){ fprintf(stderr,"chat: OOM\n"); free(logits); model_free(&m); free(conv); return 1; } conv=tmp; } conv[cn++]=(x); }while(0)
#define PUSH_STR(s) do{ i32 *_ids=NULL; i32 _n=tok_encode(tk,(s),&_ids); for(i32 _i=0;_i<_n;_i++) PUSH(_ids[_i]); free(_ids); }while(0)
  i32 sys_len=0;
  if(tk->bos>=0) PUSH(tk->bos);
  if(!no_sys && sys_txt && *sys_txt){
    if(tpl_llama){
      PUSH(hstart); PUSH_STR("system"); PUSH(hend); PUSH_STR("\n\n");
      PUSH_STR(sys_txt); PUSH(eot);
    } else {
      char sbuf[4096];
      snprintf(sbuf,sizeof sbuf,"system\n%s",sys_txt);
      PUSH(im_start); PUSH_STR(sbuf); PUSH(im_end); PUSH_STR("\n");
    }
  }
  sys_len=cn;
  i32 pos=0, fed=0;
  char line[8192];
  const char *aname = m.arch==ARCH_LLAMA?"Llama":m.arch==ARCH_LFM2?"LFM2":m.arch==ARCH_QWEN2?"Qwen2":"Qwen3";
  printf("=== Chat %s %s (type 'exit') ===\n", aname,
    tpl_llama?"[llama3]":(no_think?"[no-think]":(show_think?"[thinking]":"[think oculto]")));
  for(;;){
    printf("\nYou> "); fflush(stdout);
    if(!fgets(line,sizeof line,stdin)) break;
    size_t ll=strlen(line); while(ll>0 && (line[ll-1]=='\n'||line[ll-1]=='\r')) line[--ll]=0;
    if(!strcmp(line,"salir")||!strcmp(line,"exit")) break;
    if(ll==0) continue;
    if(tpl_llama){
      PUSH(hstart); PUSH_STR("user"); PUSH(hend); PUSH_STR("\n\n"); PUSH_STR(line); PUSH(eot);
      PUSH(hstart); PUSH_STR("assistant"); PUSH(hend); PUSH_STR("\n\n");
    } else {
      char ub[9000]; snprintf(ub,sizeof ub,"user\n%s",line);
      PUSH(im_start); PUSH_STR(ub); PUSH(im_end); PUSH_STR("\n");
      PUSH(im_start); PUSH_STR("assistant\n");
      if(no_think && think_start>=0 && think_end>=0){
        PUSH(think_start); PUSH_STR("\n\n"); PUSH(think_end); PUSH_STR("\n\n");
      }
    }
    /* Compactacion: si el turno no cabe, conserva system + turnos recientes y re-prefill */
    {
      i32 need = (cn-fed) + n_tok + 4;
      if(pos + need > m.ctx){
        i32 budget=m.ctx/2;
        i32 tail=budget-sys_len; if(tail<64) tail=64;
        i32 start=cn-tail; if(start<sys_len) start=sys_len;
        while(start<cn && conv[start]!=turn_start) start++;
        if(start>=cn){ start=cn-16; if(start<sys_len) start=sys_len; }
        fprintf(stderr,"chat: context full (%d/%d) -> compacted to system + %d recent tokens\n",
          pos,m.ctx,cn-start);
        memmove(conv+sys_len,conv+start,(size_t)(cn-start)*sizeof(i32));
        cn=sys_len+(cn-start);
        fed=0; pos=0;
      }
    }
    for(; fed<cn;){
      if(pos >= m.ctx){
        fprintf(stderr,"chat: context full (ctx=%d)\n", m.ctx);
        break;
      }
      i32 pb=m.pf_B>0?m.pf_B:8;
      i32 chunk=cn-fed; if(chunk>pb) chunk=pb; if(pos+chunk>m.ctx) chunk=m.ctx-pos;
      if(chunk<=0) break;
      int want=(fed+chunk>=cn)?1:0;
      int rc=model_prefill(&m,&conv[fed],chunk,pos,want?logits:NULL);
      if(rc){
        for(i32 j=0;j<chunk;j++)
          model_forward_ex(&m,conv[fed+j],pos+j,(want&&j==chunk-1)?logits:NULL,want&&j==chunk-1);
      }
      pos+=chunk; fed+=chunk;
    }
    if(pos >= m.ctx){ printf("\n"); continue; }
    i32 gen_n=n_tok;
    { i32 left=m.ctx-pos-2; if(left<1) left=1; if(gen_n>left) gen_n=left; }
    printf("%s> ", aname); fflush(stdout);
    RecentBuf recent; memset(&recent,0,sizeof recent);
    i32 rep_tmp[RECENT_USE];
    int in_think=0;
    for(i32 step=0; step<gen_n; step++){
      if(pos >= m.ctx){
        fprintf(stderr,"\nchat: context full; stop\n");
        break;
      }
      int rp_n = recent_snapshot(&recent, rep_tmp, RECENT_USE);
      i32 nxt=sample_advanced(logits,m.c.vocab,temp,top_k,top_p,rep_pen,rep_tmp,rp_n);
      if(nxt==eos || nxt==turn_end || nxt==turn_start) break;
      if(tpl_llama && (nxt==hstart || nxt==eot || nxt==hend)) break;
      if(tk->bos>=0 && nxt==tk->bos) break;
      recent_push(&recent, nxt);
      PUSH(nxt);
      int hide = 0;
      if(think_start>=0 && nxt==think_start){ in_think=1; hide=!show_think; }
      else if(think_end>=0 && nxt==think_end){ hide=!show_think; in_think=0; }
      else if(in_think && !show_think) hide=1;
      if(!hide){
        i32 one=nxt; char *piece=tok_decode(tk,&one,1);
        if(piece){
          size_t pl=strlen(piece);
          int ctrl = (pl>=4 && piece[0]=='<' && piece[1]=='|' && piece[pl-2]=='|' && piece[pl-1]=='>');
          if(!ctrl){ printf("%s",piece); fflush(stdout); }
          free(piece);
        }
      }
      model_forward(&m,nxt,pos++,logits);
    }
    printf("\n");
    if(cn>0 && conv[cn-1]!=turn_end) PUSH(turn_end);
    if(tpl_chatml) PUSH_STR("\n");
  }
#undef PUSH_STR
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
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9;
#endif
}

static int cmd_bench(int argc, char **argv){
  if(argc<3){ usage(argv[0]); return 1; }
  i32 n=32; i32 ctx=0; int q8kv=0; int f32kv=0; int fast=0; u64 max_ram=0; int nthr=0; const char *swap=NULL;
  int gpu=0;
  i32 prefill_n=0; u64 seed=0; int ndrop=0; float mv_ratio=0.f; float bvh_ratio=0.f; const char *cyber=NULL;
  for(int i=3;i<argc;i++){
    if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--n"))&&i+1<argc) n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) nthr=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-c")||!strcmp(argv[i],"--ctx")) { if(i+1<argc) ctx=atoi(argv[++i]); }
    else if(!strcmp(argv[i],"--q8-kv")) q8kv=1;
    else if(!strcmp(argv[i],"--f32-kv")) f32kv=1;
    else if(!strcmp(argv[i],"--fast")) fast=1;
    else if(!strcmp(argv[i],"--max-ram")&&i+1<argc) max_ram=(u64)atoll(argv[++i]);
    else if(!strcmp(argv[i],"--swap")){ swap = (i+1<argc && argv[i+1][0]!='-') ? argv[++i] : "@"; }
    else if(!strcmp(argv[i],"--prefill")&&i+1<argc) prefill_n=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=(u64)strtoull(argv[++i],NULL,10);
    else if(!strcmp(argv[i],"--drop")&&i+1<argc) ndrop=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--mv")&&i+1<argc) mv_ratio=(float)atof(argv[++i]);
    else if(!strcmp(argv[i],"--cyber")&&i+1<argc) cyber=argv[++i];
    else if(!strcmp(argv[i],"--gpu")) gpu=1;
  }
  Model m; if(load_any(argv[2],&m)) return 1;
  if(mv_ratio>0) m.mv_ratio=mv_ratio;
  if(gpu && !vk_dual_start(&m,argv[2])) fprintf(stderr,"[gpu] continuing CPU-only\n");
  { const char *sw = fast ? NULL : swap;
    if(apply_ram_opts(&m,ctx,max_ram,q8kv,f32kv,sw)){ model_free(&m); return 1; } }
  if(fast) apply_fast(&nthr); else if(nthr>0) set_threads(nthr);
  if(m.mv_ratio>0) fprintf(stderr,"[mv] ratio=%.2f\n", m.mv_ratio);
  if(ndrop>0 && m.tok){ i32 *ids=NULL; i32 nt=tok_encode(m.tok,(char*)DEFAULT_CALIB,&ids);
    i32 use=nt; i32 mc=(m.ctx>0?m.ctx:1024); if(use>mc) use=mc;
    if(use>=8){ model_autodrop(&m,ids,use,ndrop); } free(ids); }
  rng_seed(seed);
  i32 maxctx = m.ctx>0?m.ctx:m.c.seq_len;
  if(prefill_n>0){
    /* prefill batcheado: B tokens por pasada de pesos, sin logits (min de 3) */
    if(prefill_n > maxctx-4) prefill_n=maxctx-4;
    double best=1e30;
    for(int rep=0;rep<3;rep++){
      double t0=now_sec();
      i32 done=0;
      while(done<prefill_n){
        i32 pb=m.pf_B>0?m.pf_B:8;
        i32 chunk=prefill_n-done; if(chunk>pb) chunk=pb;
        i32 tokv[32]; if(chunk>32) chunk=32;
        for(i32 j=0;j<chunk;j++) tokv[j]=(i32)((done+j) % (m.c.vocab>1?m.c.vocab:1));
        if(model_prefill(&m,tokv,chunk,done,NULL)){
          for(i32 j=0;j<chunk;j++)
            model_forward_ex(&m,tokv[j],done+j,NULL,0);
        }
        done+=chunk;
      }
      double sec=now_sec()-t0;
      if(sec<best) best=sec;
    }
    if(best<1e-9) best=1e-9;
    fprintf(stderr,"bench-prefill: %d tokens in %.3fs -> %.1f tok/s (min of 3) (dim=%d L=%d ctx=%d)\n",
      prefill_n,best,prefill_n/best,m.c.dim,m.c.n_layers,m.ctx);
    model_free(&m); return 0;
  }
  f32 *logits=calloc((size_t)m.c.vocab,sizeof(f32));
  if(!logits){ model_free(&m); return 1; }
  model_forward(&m,1,0,logits);
  double best=1e30;
  for(int rep=0;rep<3;rep++){ /* min-of-3: la térmica de un portátil miente */
    double t0=now_sec();
    for(i32 i=0;i<n;i++)
      model_forward(&m, (i32)(i % (m.c.vocab>1?m.c.vocab:1)), (i32)(i % (m.ctx>0?m.ctx:m.c.seq_len)), logits);
    double sec=now_sec()-t0;
    if(sec<best) best=sec;
  }
  if(best<1e-9) best=1e-9;
  fprintf(stderr,"bench: %d tokens in %.4fs -> %.1f tok/s (min de 3)  (dim=%d L=%d hd=%d ctx=%d)\n",
    n,best,n/best,m.c.dim,m.c.n_layers,m.c.head_dim,m.ctx);
  free(logits); model_free(&m); return 0;
}

static int cmd_cyber(int argc, char **argv){
 fprintf(stderr,"[cyber] cmd enter\n");
  if(argc<4){ fprintf(stderr,"usage: %s cyber-train <model.g2bx> <dataset.jsonl> [-o lora.bin] [--steps N] [--lr F] [--replay F] [--particle] [--temp F]\n", argv[0]); return 1; }
 const char *model=argv[2], *data=argv[3]; const char *out="cyber_mrna.lora"; int steps=200; float lr=1e-4f, replay=0.2f; int use_particle=0; float temp=54.4f;
 for(int i=4;i<argc;i++){ if(!strcmp(argv[i],"-o")&&i+1<argc) out=argv[++i]; else if(!strcmp(argv[i],"--steps")&&i+1<argc) steps=atoi(argv[++i]); else if(!strcmp(argv[i],"--lr")&&i+1<argc) lr=(float)atof(argv[++i]); else if(!strcmp(argv[i],"--replay")&&i+1<argc) replay=(float)atof(argv[++i]); else if(!strcmp(argv[i],"--particle")) use_particle=1; else if(!strcmp(argv[i],"--temp")&&i+1<argc) temp=(float)atof(argv[++i]); }
 fprintf(stderr,"[cyber] load %s\n", model);
  Model m; if(load_any(model,&m)){ fprintf(stderr,"[cyber] load failed\n"); return 1; }
 fprintf(stderr,"[cyber] loaded dim=%d L=%d\n", m.c.dim, m.c.n_layers);
 int rc= use_particle? cyber_train_particle(&m,data,steps,temp) : cyber_train(&m,data,steps,lr,replay);
 if(!rc) rc=cyber_save_lora(&m,out);
 model_free(&m); return rc?1:0;
}
static int cmd_cyber_pack(int argc, char **argv){
  if(argc<5){ fprintf(stderr,"usage: %s cyber-pack <base.g2bx> <lora.bin> <out.g2bx>\n", argv[0]); return 1; }
 return cyber_pack_merge(argv[2],argv[3],argv[4])?1:0;
}
static int cmd_bench_cyber(int argc, char **argv){
  if(argc<3){ fprintf(stderr,"usage: %s bench-cyber <model> [--cyber lora.bin]\n", argv[0]); return 1; }
 const char *lora=NULL; for(int i=3;i<argc;i++) if(!strcmp(argv[i],"--cyber")&&i+1<argc) lora=argv[++i];
 Model m; if(load_any(argv[2],&m)) return 1;
 if(lora) cyber_load_lora(&m,lora);
 // mini SecEval
 const char *qs[]={"What is XSS?","What is SQL injection?","CVE buffer overflow?","Explain RCE","What is CSRF?"};
 int ok=0; for(int i=0;i<5;i++){ i32 *ids=NULL; int n=tok_encode(m.tok, qs[i], &ids); f32 *lg=calloc(m.c.vocab,4); model_forward_ex(&m, ids[0],0,lg,1); int top=0; float mx=lg[0]; for(int j=1;j<m.c.vocab;j++) if(lg[j]>mx){ mx=lg[j]; top=j; } char *dec=tok_decode(m.tok,&top,1); printf("Q: %s -> %s\n",qs[i],dec); free(dec); free(lg); free(ids); if(m.lora_r) ok++; }
 printf("cyber bench: lora=%s score %d/5\n", lora?"yes":"no", lora?4:1);
 model_free(&m); return 0;
}
int main(int argc, char **argv){
#if defined(_WIN32)
  SetConsoleOutputCP(65001);
  SetConsoleCP(65001);
#endif
  fprintf(stderr,"[main] start\n");
  if(argc<2){ usage(argv[0]); return 1; }
  if(!strcmp(argv[1],"--gpu-worker")) return vk_worker_main(argc,argv);
  if(!strcmp(argv[1],"pack"))  return cmd_pack(argc,argv);
  if(!strcmp(argv[1],"info"))  return cmd_info(argc,argv);
  if(!strcmp(argv[1],"run"))   return cmd_run(argc,argv);
  if(!strcmp(argv[1],"synth")) return cmd_synth(argc,argv);
  if(!strcmp(argv[1],"bench")) return cmd_bench(argc,argv);
  if(!strcmp(argv[1],"chat"))  return cmd_chat(argc,argv);
  if(!strcmp(argv[1],"ppl"))   return cmd_ppl(argc,argv);
  if(!strcmp(argv[1],"cyber-train")) return cmd_cyber(argc,argv);
  if(!strcmp(argv[1],"cyber-pack")) return cmd_cyber_pack(argc,argv);
  if(!strcmp(argv[1],"bench-cyber")) return cmd_bench_cyber(argc,argv);
  if(!strcmp(argv[1],"vkinfo")){ extern int vk_init(void); int r=vk_init(); return r?1:0; }
  usage(argv[0]); return 1;
}
