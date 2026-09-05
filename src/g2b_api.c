/* g2b_api.c — implementación de la API pública (include/gguf2bin.h).
 *
 * Wrapper fino sobre los internals (src/internal/g2b.h): la lógica de cómputo
 * NO se duplica aquí, solo ciclo de vida de sesión, sampler por sesión,
 * plantillas de chat y medición. Regla: ningún cambio aquí altera bits de
 * salida (los harnesses qkcheck/prefilltest/kvtest son los guardianes).
 */
#include "internal/g2b.h"
#include "internal/g2bx_io.h"
#include "internal/sampler.h"
#include "gguf2bin.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/resource.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

/* ── sesión ─────────────────────────────────────────────────────────── */
struct g2b_session {
  Model m;
  g2b_config cfg;
  char *src_path;
  Sampler samp; /* RNG + scratch de muestreo (Fase 2: módulo único) */
  /* chat: historial + plantilla + cursores (antes: locales de cmd_chat) */
  i32 *conv;
  i32 cn, ccap, pos, fed, sys_len;
  int tpl_llama, tpl_chatml;
  i32 turn_start, turn_end, eos;
  i32 hstart, hend, eot, im_start, im_end, think_start, think_end;
  int no_think;
  f32 *logits;
};

/* corpus genérico embebido para calibrar la poda (movido de main.c) */
static const char *G2B_DEFAULT_CALIB =
  "The city council approved the new budget on Tuesday after months of debate. "
  "Los estudiantes caminaron por la plaza hasta la biblioteca central del campus. "
  "A river runs through the valley, feeding the fields where farmers grow wheat. "
  "The report recommends investing in renewable energy and modernizing the grid. "
  "Every weekend the market fills with people buying fruit, bread and flowers. "
  "Software updates usually fix bugs but sometimes introduce new problems. ";

/* ── versión / errores ──────────────────────────────────────────────── */
const char *g2b_version(void){ return G2B_VERSION; }
const char *g2b_strerror(g2b_error e){
  switch(e){
    case G2B_OK: return "ok";
    case G2B_ERR_IO: return "io error (archivo no existe o no legible)";
    case G2B_ERR_FORMAT: return "formato inválido (magic/versión/estructura)";
    case G2B_ERR_GEOMETRY: return "geometría inconsistente";
    case G2B_ERR_OOM: return "sin memoria";
    case G2B_ERR_CONTEXT: return "token/posición fuera de rango";
    case G2B_ERR_UNSUPPORTED: return "no soportado";
    case G2B_ERR_CANCELLED: return "cancelado por el usuario";
    default: return "error desconocido";
  }
}

/* ── helpers (movidos de main.c) ────────────────────────────────────── */
static int ends_with(const char *s, const char *suf){
  size_t n=strlen(s), m=strlen(suf);
  return n>=m && !strcmp(s+n-m,suf);
}
static int is_g2bx(const char *p){
  return ends_with(p,".g2bx")||ends_with(p,".G2BX")||ends_with(p,".gbin");
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
static void set_threads(int n){
#if defined(_OPENMP)
  if(n>0) omp_set_num_threads(n);
  int t = n>0?n:omp_get_num_threads();
  fprintf(stderr,"threads: using %d core(s)\n", t);
#else
  (void)n;
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
static i32 find_tok(Tokenizer *t, const char *s){
  for(i32 i=0;i<t->n;i++) if(t->tok[i]&&!strcmp(t->tok[i],s)) return i;
  return -1;
}

/* ── RAM opts (movido de main.c) ────────────────────────────────────── */
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

/* ── ciclo de vida ──────────────────────────────────────────────────── */
g2b_error g2b_open(const char *path, const g2b_config *cfg, g2b_session **out){
  if(!path || !out) return G2B_ERR_IO;
  { FILE *t=fopen(path,"rb"); if(!t) return G2B_ERR_IO; fclose(t); }
  g2b_session *s=calloc(1,sizeof *s);
  if(!s) return G2B_ERR_OOM;
  sampler_init(&s->samp);
  if(cfg) s->cfg=*cfg;
  s->src_path=strdup(path);
  if(!s->src_path){ free(s); return G2B_ERR_OOM; }
  int rc = is_g2bx(path) ? model_load_g2bx(path,&s->m) : model_load_gguf(path,&s->m);
  if(rc){ free(s->src_path); free(s); return G2B_ERR_FORMAT; }
  /* geometría mínima (la validación profunda ya la hizo la carga) */
  if(s->m.c.n_heads<=0 || s->m.c.dim<=0 || s->m.c.vocab<=0 || s->m.c.n_layers<=0){
    model_free(&s->m); free(s->src_path); free(s); return G2B_ERR_GEOMETRY;
  }
  if(s->cfg.mv_ratio>0) s->m.mv_ratio=s->cfg.mv_ratio;
  if(s->cfg.bvh_keep>0){
    s->m.use_bvh=1; s->m.bvh_keep=s->cfg.bvh_keep;
    fprintf(stderr,"[bvh] sparse %.2f\n",s->cfg.bvh_keep);
  }
  if(s->cfg.lora_path && cyber_load_lora(&s->m,s->cfg.lora_path)){
    model_free(&s->m); free(s->src_path); free(s); return G2B_ERR_FORMAT;
  }
  if(s->cfg.gpu && !vk_dual_start(&s->m,path))
    fprintf(stderr,"[gpu] continuing CPU-only\n");
  { const char *sw = s->cfg.fast ? NULL : s->cfg.swap_path;
    int q8 = s->cfg.q8_kv>0, f32 = s->cfg.q8_kv==0;
    if(apply_ram_opts(&s->m,s->cfg.ctx,(u64)s->cfg.max_ram_bytes,q8,f32,sw)){
      model_free(&s->m); free(s->src_path); free(s); return G2B_ERR_OOM;
    } }
  { int nthr=s->cfg.threads;
    if(s->cfg.fast) apply_fast(&nthr); else if(nthr>0) set_threads(nthr); }
  if(s->m.mv_ratio>0) fprintf(stderr,"[mv] ratio=%.2f\n", s->m.mv_ratio);
  sampler_seed(&s->samp, s->cfg.seed);
  *out=s;
  return G2B_OK;
}
void g2b_close(g2b_session *s){
  if(!s) return;
  model_free(&s->m);
  free(s->src_path); sampler_free(&s->samp);
  free(s->conv); free(s->logits);
  free(s);
}
g2b_error g2b_set_ctx(g2b_session *s, int ctx){
  if(!s) return G2B_ERR_IO;
  return model_set_ctx(&s->m,ctx)?G2B_ERR_OOM:G2B_OK;
}
int g2b_ctx_used(const g2b_session *s){ return s ? s->pos : 0; }
void g2b_ram_report(const g2b_session *s, char *buf, int buflen){
  if(!buf || buflen<=0) return;
  buf[0]=0;
  if(!s) return;
  const Model *m=&s->m;
  u64 rt=model_est_ram((Model*)m);
  int q8=(m->flags&F_KV_Q8)&&!m->no_kv_q8;
  snprintf(buf,(size_t)buflen,
    "ram: pesos=%llu MB (%s; paginas reclamables) | runtime=%llu MB "
    "= KV(%s,ctx=%d)%s + buffers + tokenizer",
    (unsigned long long)(m->data_size>>20), m->use_mmap?"mmap":"memcpy",
    (unsigned long long)(rt>>20), q8?"Q8_0":"F32", m->ctx,
    m->use_swap?" (file-backed)":"");
}
static void fill_info_from_cfg(g2b_model_info *o, const ModelCfg *c, int arch,
                               int bos, int eos, u64 wbytes, u64 est, int hastok){
  static const char *an[]={"llama","qwen2","qwen3","lfm2","qwen35"};
  snprintf(o->arch,sizeof o->arch,"%s",arch>=0&&arch<5?an[arch]:"?");
  o->dim=c->dim; o->n_layers=c->n_layers; o->n_heads=c->n_heads;
  o->n_kv_heads=c->n_kv_heads; o->head_dim=c->head_dim;
  o->vocab=c->vocab; o->ctx_max=c->seq_len; o->ctx_eff=c->seq_len;
  o->bos_id=bos; o->eos_id=eos;
  o->weight_bytes=wbytes; o->est_ram_bytes=est; o->has_tokenizer=hastok;
}
g2b_error g2b_model_info_of(const g2b_session *s, g2b_model_info *out){
  if(!s || !out) return G2B_ERR_IO;
  const Model *m=&s->m;
  int bos=m->tok&&m->tok->bos>=0?m->tok->bos:-1;
  int eos=m->tok&&m->tok->eos>=0?m->tok->eos:-1;
  u64 wsum=0;
  for(u32 i=0;i<m->n_slots;i++) wsum+=m->slots[i].nbytes;
  fill_info_from_cfg(out,&m->c,m->arch,bos,eos,wsum,
    model_est_ram((Model*)m),m->tok!=NULL);
  out->ctx_eff=m->ctx>0?m->ctx:m->c.seq_len;
  return G2B_OK;
}
/* Metadata sin cargar pesos: escaneo del header G2BX o metadata GGUF. */
g2b_error g2b_info(const char *path, g2b_model_info *out){
  if(!path || !out) return G2B_ERR_IO;
  memset(out,0,sizeof *out);
  { FILE *t=fopen(path,"rb"); if(!t) return G2B_ERR_IO; fclose(t); }
  if(is_g2bx(path)){
    FILE *f=fopen(path,"rb");
    if(!f) return G2B_ERR_IO;
    G2bxHeader h;
    int hrc=g2bx_read_header(f,&h);
    fclose(f);
    if(hrc) return G2B_ERR_FORMAT;
    ModelCfg c=h.cfg;
    if(c.n_kv_heads<=0) c.n_kv_heads=c.n_heads;
    if(c.head_dim<=0 && c.n_heads>0) c.head_dim=c.dim/c.n_heads;
    u64 tokbytes=h.has_tok
      ? (u64)(h.tok_nv+h.tok_nm)*48u + ((u64)1u<<19)*(8u+4u)*2u : 0;
    int ctx=c.seq_len>0?c.seq_len:2048;
    u64 est=model_est_ram_cfg(&c,c.fa_interval,(h.flags&F_KV_Q8)?1:0,ctx,16,tokbytes);
    fill_info_from_cfg(out,&c,h.arch,h.tok_bos,h.tok_eos,h.weight_bytes,est,h.has_tok);
    g2bx_header_free(&h);
    return G2B_OK;
  }
  /* GGUF: metadata (mmap, sin cargar tensores a RAM) */
  GGUF g; memset(&g,0,sizeof g);
  if(gguf_load(path,&g)) return G2B_ERR_FORMAT;
  char aname[64]="?";
  gguf_meta_str(&g,"general.architecture",aname,sizeof aname);
  ModelCfg c; memset(&c,0,sizeof c);
  { char k[96];
    snprintf(k,sizeof k,"%s.embedding_length",aname); c.dim=(i32)gguf_meta_i64(&g,k);
    snprintf(k,sizeof k,"%s.block_count",aname); c.n_layers=(i32)gguf_meta_i64(&g,k);
    snprintf(k,sizeof k,"%s.attention.head_count",aname); c.n_heads=(i32)gguf_meta_i64(&g,k);
    snprintf(k,sizeof k,"%s.attention.head_count_kv",aname); c.n_kv_heads=(i32)gguf_meta_i64(&g,k);
    if(!c.n_kv_heads) c.n_kv_heads=c.n_heads;
    snprintf(k,sizeof k,"%s.attention.key_length",aname); c.head_dim=(i32)gguf_meta_i64(&g,k);
    if(!c.head_dim && c.n_heads>0) c.head_dim=c.dim/c.n_heads;
    snprintf(k,sizeof k,"%s.context_length",aname); c.seq_len=(i32)gguf_meta_i64(&g,k);
    if(!c.seq_len) c.seq_len=2048;
    GTensor *e=gguf_by_name(&g,"token_embd.weight");
    if(e && e->n_dims>=2){ u64 a=e->dims[0], b=e->dims[1]; c.vocab=(i32)(a>b?a:b); }
  }
  u64 wsum=0;
  for(u64 i=0;i<g.n_tensors;i++){
    GTensor *t=&g.t[i];
    u64 ne=1;
    for(u32 d=0;d<t->n_dims;d++){
      if(t->dims[d] && ne>UINT64_MAX/t->dims[d]){ ne=0; break; }
      ne*=t->dims[d];
    }
    wsum+=ggml_type_size(t->type,ne);
  }
  int bos=(int)gguf_meta_i64(&g,"tokenizer.ggml.bos_token_id");
  int eos=(int)gguf_meta_i64(&g,"tokenizer.ggml.eos_token_id");
  int hastok=gguf_meta_arr_len(&g,"tokenizer.ggml.tokens")>0;
  u64 tokbytes=hastok ? (u64)(c.vocab>0?c.vocab:150000)*48u + ((u64)1u<<19)*(8u+4u)*2u : 0;
  u64 est=model_est_ram_cfg(&c,0,0,c.seq_len,16,tokbytes);
  snprintf(out->arch,sizeof out->arch,"%.15s",aname);
  out->dim=c.dim; out->n_layers=c.n_layers; out->n_heads=c.n_heads;
  out->n_kv_heads=c.n_kv_heads; out->head_dim=c.head_dim;
  out->vocab=c.vocab; out->ctx_max=c.seq_len; out->ctx_eff=c.seq_len;
  out->bos_id=bos; out->eos_id=eos;
  out->weight_bytes=wsum; out->est_ram_bytes=est; out->has_tokenizer=hastok;
  gguf_free(&g);
  return G2B_OK;
}

/* ── tokenizer ────────────────────────────────────────────────────── */
int g2b_encode(g2b_session *s, const char *text, int32_t **out_ids){
  if(!s || !text || !out_ids) return -1;
  *out_ids=NULL;
  if(!s->m.tok) return -1;
  return tok_encode(s->m.tok,(char*)text,out_ids);
}
char *g2b_decode(g2b_session *s, const int32_t *ids, int n){
  if(!s || !ids || n<=0) return NULL;
  if(!s->m.tok) return NULL;
  return tok_decode(s->m.tok,ids,n);
}

/* ── generación de bajo nivel ─────────────────────────────────────── */
#define G2B_REP_WIN 64
g2b_error g2b_generate(g2b_session *s, const int32_t *toks, int n_toks,
                       const g2b_gen_params *p){
  if(!s || !p) return G2B_ERR_IO;
  if(p->seed) sampler_seed(&s->samp,p->seed);
  Model *m=&s->m;
  int rep_win=p->repeat_window>0?p->repeat_window:G2B_REP_WIN;
  if(rep_win>128) rep_win=128;
  i32 recent_ring[128]; int recent_n=0;
  i32 rep_tmp[128];
  f32 *logits=calloc((size_t)(m->c.vocab>0?m->c.vocab:1),sizeof(f32));
  if(!logits) return G2B_ERR_OOM;
  i32 pos=g2b_ctx_used(s);
  /* prefill por chunks (logits solo del último token) */
  for(i32 i=0;i<n_toks;){
    if(p->is_cancelled && p->is_cancelled(p->ud)){ free(logits); return G2B_ERR_CANCELLED; }
    if(pos >= m->ctx) break; /* contexto lleno: trunca como la CLI */
    if(toks[i]<0 || toks[i]>=m->c.vocab){ free(logits); return G2B_ERR_CONTEXT; }
    i32 pb=m->pf_B>0?m->pf_B:8;
    i32 chunk=n_toks-i; if(chunk>pb) chunk=pb; if(pos+chunk>m->ctx) chunk=m->ctx-pos;
    if(chunk<=0) break;
    int want=(i+chunk>=n_toks)?1:0;
    int rc=model_prefill(m,&toks[i],chunk,pos,want?logits:NULL);
    if(rc){
      for(i32 j=0;j<chunk;j++)
        model_forward_ex(m,toks[i+j],pos+j,(want&&j==chunk-1)?logits:NULL,want&&j==chunk-1);
    }
    for(i32 j=0;j<chunk;j++){ recent_ring[recent_n%128]=toks[i+j]; recent_n++; }
    pos+=chunk; i+=chunk;
  }
  s->pos=pos;
  double t0=now_sec();
  for(int i=0;i<p->max_tokens;i++){
    if(p->is_cancelled && p->is_cancelled(p->ud)){ free(logits); s->pos=pos; return G2B_ERR_CANCELLED; }
    if(pos >= m->ctx) break;
    int a=recent_n>rep_win?recent_n-rep_win:0, rp_n=0;
    for(int k=a;k<recent_n && rp_n<128;k++) rep_tmp[rp_n++]=recent_ring[k%128];
    i32 next=sampler_sample(&s->samp,logits,m->c.vocab,p->temp,p->top_k,p->top_p,
                             p->repeat_penalty,rep_tmp,rp_n);
    if(p->on_token){
      if(m->tok){
        char *pc=tok_decode(m->tok,&next,1);
        if(pc){ p->on_token(pc,p->ud); free(pc); }
      } else {
        /* sin tokenizer: la CLI histórica imprime ids (modelos synth) */
        char tmp[16]; snprintf(tmp,sizeof tmp,"%d ",(int)next);
        p->on_token(tmp,p->ud);
      }
    }
    recent_ring[recent_n%128]=next; recent_n++;
    model_forward(m,next,pos,logits); pos++;
    if(p->on_progress){
      double el=now_sec()-t0; if(el<1e-9) el=1e-9;
      p->on_progress(i+1,p->max_tokens,(float)((i+1)/el),p->ud);
    }
  }
  s->pos=pos;
  free(logits);
  return G2B_OK;
}

/* ── chat ─────────────────────────────────────────────────────────── */
static int conv_push(g2b_session *s, i32 x){
  if(s->cn>=s->ccap){
    if(s->ccap>=(1<<20)) return -1;
    s->ccap*=2;
    i32 *tmp=realloc(s->conv,(size_t)s->ccap*sizeof(i32));
    if(!tmp) return -1;
    s->conv=tmp;
  }
  s->conv[s->cn++]=x;
  return 0;
}
static int conv_push_str(g2b_session *s, const char *txt){
  i32 *ids=NULL; i32 n=tok_encode(s->m.tok,(char*)txt,&ids);
  int rc=0;
  for(i32 i=0;i<n && !rc;i++) rc=conv_push(s,ids[i]);
  free(ids);
  return rc;
}
g2b_error g2b_chat_begin(g2b_session *s, const char *system_utf8, int no_think){
  if(!s) return G2B_ERR_IO;
  Model *m=&s->m;
  if(!m->tok) return G2B_ERR_UNSUPPORTED;
  Tokenizer *tk=m->tok;
  s->no_think=no_think;
  s->hstart=find_tok(tk,"<|start_header_id|>"); s->hend=find_tok(tk,"<|end_header_id|>");
  s->eot=find_tok(tk,"<|eot_id|>");
  if(s->eot<0) s->eot=find_tok(tk,"<|eom_id|>");
  s->im_start=find_tok(tk,"<|im_start|>"); s->im_end=find_tok(tk,"<|im_end|>");
  s->think_start=find_tok(tk,"<think>"); s->think_end=find_tok(tk,"</think>");
  s->tpl_llama=(s->hstart>=0 && s->hend>=0 && s->eot>=0);
  s->tpl_chatml=(s->im_start>=0 && s->im_end>=0);
  if(!s->tpl_llama && !s->tpl_chatml) return G2B_ERR_UNSUPPORTED;
  if(s->tpl_llama && s->tpl_chatml){
    if(m->arch==ARCH_LLAMA) s->tpl_chatml=0; else s->tpl_llama=0;
  }
  s->turn_start = s->tpl_llama ? s->hstart : s->im_start;
  s->turn_end   = s->tpl_llama ? s->eot    : s->im_end;
  s->eos=(tk->eos>=0)?tk->eos:s->turn_end;
  free(s->conv); s->conv=NULL; s->cn=0; s->ccap=2048; s->pos=0; s->fed=0; s->sys_len=0;
  s->conv=malloc((size_t)s->ccap*sizeof(i32));
  if(!s->conv) return G2B_ERR_OOM;
  free(s->logits); s->logits=NULL;
  s->logits=calloc((size_t)(m->c.vocab>0?m->c.vocab:1),sizeof(f32));
  if(!s->logits){ free(s->conv); s->conv=NULL; return G2B_ERR_OOM; }
  if(tk->bos>=0 && conv_push(s,tk->bos)) return G2B_ERR_OOM;
  if(system_utf8 && *system_utf8){
    if(s->tpl_llama){
      if(conv_push(s,s->hstart)||conv_push_str(s,"system")||conv_push(s,s->hend)
         ||conv_push_str(s,"\n\n")||conv_push_str(s,system_utf8)||conv_push(s,s->eot))
        return G2B_ERR_OOM;
    } else {
      char sbuf[4096];
      snprintf(sbuf,sizeof sbuf,"system\n%s",system_utf8);
      if(conv_push(s,s->im_start)||conv_push_str(s,sbuf)||conv_push(s,s->im_end)
         ||conv_push_str(s,"\n"))
        return G2B_ERR_OOM;
    }
  }
  s->sys_len=s->cn;
  return G2B_OK;
}
g2b_error g2b_chat_reset(g2b_session *s){
  if(!s || !s->conv) return G2B_ERR_IO;
  s->cn=s->sys_len; s->fed=0; s->pos=0;
  return G2B_OK;
}
g2b_error g2b_chat_turn(g2b_session *s, const char *user_utf8,
                        const g2b_gen_params *p){
  if(!s || !user_utf8 || !p) return G2B_ERR_IO;
  if(!s->conv || !s->logits) return G2B_ERR_CONTEXT;
  if(p->seed) sampler_seed(&s->samp,p->seed);
  Model *m=&s->m;
  Tokenizer *tk=m->tok;
  /* formatea el turno */
  if(s->tpl_llama){
    if(conv_push(s,s->hstart)||conv_push_str(s,"user")||conv_push(s,s->hend)
       ||conv_push_str(s,"\n\n")||conv_push_str(s,user_utf8)||conv_push(s,s->eot)
       ||conv_push(s,s->hstart)||conv_push_str(s,"assistant")||conv_push(s,s->hend)
       ||conv_push_str(s,"\n\n"))
      return G2B_ERR_OOM;
  } else {
    char ub[9000]; snprintf(ub,sizeof ub,"user\n%s",user_utf8);
    if(conv_push(s,s->im_start)||conv_push_str(s,ub)||conv_push(s,s->im_end)
       ||conv_push_str(s,"\n")||conv_push(s,s->im_start)||conv_push_str(s,"assistant\n"))
      return G2B_ERR_OOM;
    if(s->no_think && s->think_start>=0 && s->think_end>=0){
      if(conv_push(s,s->think_start)||conv_push_str(s,"\n\n")
         ||conv_push(s,s->think_end)||conv_push_str(s,"\n\n"))
        return G2B_ERR_OOM;
    }
  }
  /* compactación: conserva system + turnos recientes y re-prefill */
  {
    i32 need = (s->cn-s->fed) + p->max_tokens + 4;
    if(s->pos + need > m->ctx){
      i32 budget=m->ctx/2;
      i32 tail=budget-s->sys_len; if(tail<64) tail=64;
      i32 start=s->cn-tail; if(start<s->sys_len) start=s->sys_len;
      while(start<s->cn && s->conv[start]!=s->turn_start) start++;
      if(start>=s->cn){ start=s->cn-16; if(start<s->sys_len) start=s->sys_len; }
      fprintf(stderr,"chat: context full (%d/%d) -> compacted to system + %d recent tokens\n",
        s->pos,m->ctx,s->cn-start);
      memmove(s->conv+s->sys_len,s->conv+start,(size_t)(s->cn-start)*sizeof(i32));
      s->cn=s->sys_len+(s->cn-start);
      s->fed=0; s->pos=0;
    }
  }
  /* prefill de lo nuevo */
  for(; s->fed<s->cn;){
    if(p->is_cancelled && p->is_cancelled(p->ud)) return G2B_ERR_CANCELLED;
    if(s->pos >= m->ctx) break;
    i32 pb=m->pf_B>0?m->pf_B:8;
    i32 chunk=s->cn-s->fed; if(chunk>pb) chunk=pb;
    if(s->pos+chunk>m->ctx) chunk=m->ctx-s->pos;
    if(chunk<=0) break;
    int want=(s->fed+chunk>=s->cn)?1:0;
    int rc=model_prefill(m,&s->conv[s->fed],chunk,s->pos,want?s->logits:NULL);
    if(rc){
      for(i32 j=0;j<chunk;j++)
        model_forward_ex(m,s->conv[s->fed+j],s->pos+j,
          (want&&j==chunk-1)?s->logits:NULL,want&&j==chunk-1);
    }
    s->pos+=chunk; s->fed+=chunk;
  }
  if(s->pos >= m->ctx) return G2B_OK;
  /* decode con paradas de turno */
  i32 rep_ring[128]; int rep_n=0;
  i32 rep_tmp[128];
  int rep_win=p->repeat_window>0?p->repeat_window:64;
  if(rep_win>128) rep_win=128;
  int in_think=0;
  i32 gen_n=p->max_tokens;
  { i32 left=m->ctx-s->pos-2; if(left<1) left=1; if(gen_n>left) gen_n=left; }
  double t0=now_sec();
  for(i32 step=0; step<gen_n; step++){
    if(p->is_cancelled && p->is_cancelled(p->ud)) return G2B_ERR_CANCELLED;
    if(s->pos >= m->ctx) break;
    int a=rep_n>rep_win?rep_n-rep_win:0, rp_n=0;
    for(int k=a;k<rep_n && rp_n<128;k++) rep_tmp[rp_n++]=rep_ring[k%128];
    i32 nxt=sampler_sample(&s->samp,s->logits,m->c.vocab,p->temp,p->top_k,p->top_p,
                            p->repeat_penalty,rep_tmp,rp_n);
    if(nxt==s->eos || nxt==s->turn_end || nxt==s->turn_start) break;
    if(s->tpl_llama && (nxt==s->hstart || nxt==s->eot || nxt==s->hend)) break;
    if(tk->bos>=0 && nxt==tk->bos) break;
    rep_ring[rep_n%128]=nxt; rep_n++;
    if(conv_push(s,nxt)) break;
    int hide = 0;
    if(s->think_start>=0 && nxt==s->think_start){ in_think=1; hide=!p->show_think; }
    else if(s->think_end>=0 && nxt==s->think_end){ hide=!p->show_think; in_think=0; }
    else if(in_think && !p->show_think) hide=1;
    if(!hide && p->on_token){
      char *piece=tok_decode(tk,&nxt,1);
      if(piece){
        size_t pl=strlen(piece);
        int ctrl = (pl>=4 && piece[0]=='<' && piece[1]=='|' && piece[pl-2]=='|' && piece[pl-1]=='>');
        if(!ctrl) p->on_token(piece,p->ud);
        free(piece);
      }
    }
    model_forward(m,nxt,s->pos++,s->logits);
    if(p->on_progress){
      double el=now_sec()-t0; if(el<1e-9) el=1e-9;
      p->on_progress(step+1,gen_n,(float)((step+1)/el),p->ud);
    }
  }
  if(s->cn>0 && s->conv[s->cn-1]!=s->turn_end && conv_push(s,s->turn_end))
    return G2B_ERR_OOM;
  if(s->tpl_chatml && conv_push_str(s,"\n")) return G2B_ERR_OOM;
  return G2B_OK;
}
/* ── autodrop ShortGPT ────────────────────────────────────────────── */
g2b_error g2b_autodrop(g2b_session *s, int ndrop){
  if(!s || ndrop<=0) return G2B_OK;
  Model *m=&s->m;
  if(!m->tok) return G2B_ERR_UNSUPPORTED;
  i32 *ids=NULL; i32 nt=tok_encode(m->tok,(char*)G2B_DEFAULT_CALIB,&ids);
  i32 use=nt; i32 mc=(m->ctx>0?m->ctx:1024); if(use>mc) use=mc;
  if(use>=8) model_autodrop(m,ids,use,ndrop);
  free(ids);
  return G2B_OK;
}

/* ── ppl: cross-entropy del modelo sobre un texto (movido de main.c) ── */
g2b_error g2b_ppl(g2b_session *s, const char *text, int max_tokens,
                  g2b_ppl_result *out){
  if(!s || !text || !out) return G2B_ERR_IO;
  Model *m=&s->m;
  if(!m->tok) return G2B_ERR_UNSUPPORTED;
  i32 *ids=NULL; i32 nt=tok_encode(m->tok,(char*)text,&ids);
  if(m->tok->bos>=0 && !(nt>0 && ids[0]==m->tok->bos)){
    i32 *tmp=malloc((size_t)(nt+1)*sizeof(i32));
    if(tmp){ tmp[0]=m->tok->bos; memcpy(tmp+1,ids,(size_t)nt*sizeof(i32)); free(ids); ids=tmp; nt++; }
  }
  if(nt>max_tokens) nt=max_tokens;
  if(nt<2){ free(ids); return G2B_ERR_CONTEXT; }
  fprintf(stderr,"ppl: %d tokens, ctx=%d\n",nt,m->ctx);
  f32 *logits=malloc((size_t)m->c.vocab*sizeof(f32));
  if(!logits){ free(ids); return G2B_ERR_OOM; }
  double nll=0; i32 count=0;
  i32 win=m->ctx>0?m->ctx:m->c.seq_len;
  for(i32 base=0; base<nt-1; base+=win){
    i32 end = base+win<nt ? base+win : nt;
    for(i32 p=base; p<end-1; p++){
      model_forward_ex(m,ids[p],p-base,logits,1);
      f32 mx=logits[0];
      for(i32 v=1;v<m->c.vocab;v++) if(logits[v]>mx) mx=logits[v];
      double lse=0;
      for(i32 v=0;v<m->c.vocab;v++) lse+=exp((double)(logits[v]-mx));
      lse=log(lse)+mx;
      nll += lse-(double)logits[ids[p+1]];
      count++;
    }
  }
  free(logits); free(ids);
  out->tokens=nt; out->evaluated=count;
  out->nll_per_token=nll/(count?count:1);
  out->perplexity=exp(out->nll_per_token);
  return G2B_OK;
}

/* ── pack ─────────────────────────────────────────────────────────── */
g2b_error g2b_pack(const char *gguf, const char *out, const g2b_pack_opts *o){
  if(!gguf || !out) return G2B_ERR_IO;
  int downq4=0; float prune=0.f; const char *calib=NULL; int oq=0;
  if(o){ downq4=o->downquant; prune=o->prune; calib=o->calib_path; oq=o->out_quant; }
  if(oq==1) g2bx_set_q4s(1);
  else if(oq==2) g2bx_set_q4s_psy(1);
  else if(oq==3) g2bx_set_q4vvc(1);
  if(prune>0.f && (prune<=0.001f||prune>=0.9f)) return G2B_ERR_CONTEXT;
  if(prune<=0.f) return g2bx_pack_prune(gguf,out,downq4,0.f)?G2B_ERR_FORMAT:G2B_OK;
  /* dos fases: pack completo -> calibrar -> repack podado */
  char tmp[1280];
  snprintf(tmp,sizeof tmp,"%s.calib.g2bx",out);
  fprintf(stderr,"pack: phase 1/2 - full pack for calibration\n");
  if(g2bx_pack_prune(gguf,tmp,downq4,0.f)){ remove(tmp); return G2B_ERR_FORMAT; }
  Model m; memset(&m,0,sizeof m);
  if(model_load_g2bx(tmp,&m)){ remove(tmp); return G2B_ERR_FORMAT; }
  i32 maxtok=768; i32 ctx=m.c.seq_len>0?(m.c.seq_len<maxtok?m.c.seq_len:maxtok):maxtok;
  model_set_ctx(&m,ctx);
  char *text=NULL; size_t len=0;
  if(calib){
    if(!os_read_file(calib,&text,&len)){ if(!len){ free(text); text=NULL; len=0; } }
    else fprintf(stderr,"pack: cannot open %s - using embedded corpus\n",calib);
  }
  if(!text || len==0){ text=(char*)G2B_DEFAULT_CALIB; len=strlen(G2B_DEFAULT_CALIB); }
  g2b_error rc=G2B_OK;
  if(!m.tok){ fprintf(stderr,"pack: model has no tokenizer - cannot --prune\n"); rc=G2B_ERR_FORMAT; }
  else {
    i32 *ids=NULL; i32 nt=tok_encode(m.tok,text,&ids);
    if(nt>ctx) nt=ctx;
    if(nt<64){ fprintf(stderr,"pack: insufficient calibration (%d tokens)\n",nt); free(ids); rc=G2B_ERR_FORMAT; }
    else {
      fprintf(stderr,"pack: calibrating on %d tokens...\n",nt);
      int ok = (model_collect_stats(&m,ids,nt)==0 && m.ffn_stats!=NULL);
      free(ids);
      if(!ok) rc=G2B_ERR_FORMAT;
      else if(g2bx_pack_prune_scores(gguf,out,downq4,prune,m.ffn_stats)) rc=G2B_ERR_FORMAT;
      model_free_stats(&m);
    }
  }
  if(text!=(char*)G2B_DEFAULT_CALIB) free(text);
  model_free(&m);
  remove(tmp);
  return rc;
}

/* ── bench (min-of-3, movido de main.c) ───────────────────────────── */
g2b_error g2b_bench(g2b_session *s, int n_tokens, int prefill_tokens,
                    float *decode_tps, float *prefill_tps){
  if(!s) return G2B_ERR_IO;
  Model *m=&s->m;
  i32 maxctx = m->ctx>0?m->ctx:m->c.seq_len;
  if(prefill_tokens>0){
    i32 pn=prefill_tokens;
    if(pn > maxctx-4) pn=maxctx-4;
    double best=1e30;
    for(int rep=0;rep<3;rep++){
      double t0=now_sec();
      i32 done=0;
      while(done<pn){
        i32 pb=m->pf_B>0?m->pf_B:8;
        i32 chunk=pn-done; if(chunk>pb) chunk=pb;
        i32 tokv[32]; if(chunk>32) chunk=32;
        for(i32 j=0;j<chunk;j++) tokv[j]=(i32)((done+j) % (m->c.vocab>1?m->c.vocab:1));
        if(model_prefill(m,tokv,chunk,done,NULL)){
          for(i32 j=0;j<chunk;j++)
            model_forward_ex(m,tokv[j],done+j,NULL,0);
        }
        done+=chunk;
      }
      double sec=now_sec()-t0;
      if(sec<best) best=sec;
    }
    if(best<1e-9) best=1e-9;
    if(prefill_tps) *prefill_tps=(float)(pn/best);
    if(decode_tps) *decode_tps=0;
    return G2B_OK;
  }
  f32 *logits=calloc((size_t)(m->c.vocab>0?m->c.vocab:1),sizeof(f32));
  if(!logits) return G2B_ERR_OOM;
  model_forward(m,1,0,logits);
  double best=1e30;
  for(int rep=0;rep<3;rep++){
    double t0=now_sec();
    for(i32 i=0;i<n_tokens;i++)
      model_forward(m, (i32)(i % (m->c.vocab>1?m->c.vocab:1)), (i32)(i % (m->ctx>0?m->ctx:m->c.seq_len)), logits);
    double sec=now_sec()-t0;
    if(sec<best) best=sec;
  }
  if(best<1e-9) best=1e-9;
  if(decode_tps) *decode_tps=(float)(n_tokens/best);
  if(prefill_tps) *prefill_tps=0;
  free(logits);
  return G2B_OK;
}

/* ── sonda Vulkan ─────────────────────────────────────────────────── */
g2b_error g2b_vk_probe(char *report, int buflen){
  if(!report || buflen<=0) return G2B_ERR_IO;
  report[0]=0;
  int r=vk_init();
  if(r){
    snprintf(report,(size_t)buflen,"vulkan: init fallido (rc=%d)",r);
    return G2B_ERR_UNSUPPORTED;
  }
  snprintf(report,(size_t)buflen,"vulkan: %s vram=%llu MB",
    vk_device_name(),(unsigned long long)(vk_vram()>>20));
  return G2B_OK;
}
