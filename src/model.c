/* model.c — carga/libera G2BX, geometría, RAM, calibración, synth (split de l5, Fase 3) */
#include "internal/g2b.h"
#include "internal/g2bx_io.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/stat.h> /* load_gguf: cache por mtime */

u8 *slot_ptr(Model *m, Slot *s){
  if(!m||!s||!m->data) return NULL;
  if((u64)s->off>(u64)m->data_size || (u64)s->nbytes>(u64)m->data_size-(u64)s->off) return NULL;
  return m->data+s->off;
}

Slot *slot_get(Model *m, u8 role, i32 layer){
  if(layer<0) return m->ix_global[role];
  if(layer>=m->c.n_layers) return NULL;
  return m->ix_layer[layer][role];
}

static void build_index(Model *m){
  m->ix_global=calloc(R_COUNT,sizeof(Slot*));
  m->ix_layer=calloc((size_t)m->c.n_layers,sizeof(Slot**));
  for(i32 L=0;L<m->c.n_layers;L++) m->ix_layer[L]=calloc(R_COUNT,sizeof(Slot*));
  for(u32 i=0;i<m->n_slots;i++){
    Slot *s=&m->slots[i];
    if(s->role>=R_COUNT) continue;
    if(s->layer==0xFFFF) m->ix_global[s->role]=s;
    else if(s->layer<m->c.n_layers) m->ix_layer[s->layer][s->role]=s;
  }
}

/* KV cache en Q8_0 (34 B / 32 elems) vs F32 (128 B / 32 elems): ~3.76x menos RAM */
/* Ajusta el contexto efectivo (y el modo de KV cache) del runtime. */
int model_set_ctx(Model *m, i32 ctx){
  if(!m || m->c.n_layers<=0 || m->c.n_heads<=0) return -1;
  if(ctx<=0) ctx=256;
  if(m->c.seq_len>0 && ctx>m->c.seq_len) ctx=m->c.seq_len;
  free_rt(m);
  m->ctx=ctx;
  if(alloc_rt(m,ctx)){ free_rt(m); m->ctx=0; return -1; }
  return 0;
}

/* Estimación de RAM residente del runtime (excluye pesos: mmap→page cache evictable). */
static u64 est_for(Model *m, i32 ctx, int q8){
  ModelCfg *c=&m->c;
  i32 nkv=c->n_kv_heads*c->head_dim;
  i32 nl=kv_nlayers(m);
  /* K y V: dos cachés */
  u64 kv = q8 ? 2u*(size_t)nl*ctx*kv_q8_rowsize(nkv)
              : 2u*(size_t)nl*ctx*(size_t)nkv*4u;
  i32 dim=c->dim, hid=c->hidden_dim;
  i32 maxn=dim>hid?dim:hid; if(c->vocab>maxn) maxn=c->vocab;
  u64 nbuf=(u64)dim*3u+(u64)hid*2u+(u64)c->n_heads*c->head_dim+(u64)nkv*2u+(u64)c->n_heads*ctx+(u64)maxn;
  u64 buf=nbuf*4u;
  /* prefill batcheado */
  if(m->pf_B>0)
    buf += (u64)m->pf_B*(u64)(dim+dim+hid*2+c->n_heads*c->head_dim+nkv*2+c->n_heads*ctx)*4u;
  u64 tok=0; Tokenizer *t=m->tok;
  if(t) tok=(u64)(t->n+(size_t)t->nmerges)*48u + ((size_t)1u<<19)*(8u+4u)*2u;
  return kv+buf+tok;
}
u64 model_est_ram(Model *m){
  if(!m || m->c.n_layers<=0) return 0;
  i32 ctx=m->ctx>0?m->ctx:m->c.seq_len;
  return est_for(m, ctx, kv_is_q8(m));
}
/* Estimación de RAM a partir de solo la cfg (g2b_info: header escaneado, sin
 * cargar el modelo). Misma fórmula que est_for; mantener ambas sincronizadas. */
u64 model_est_ram_cfg(const ModelCfg *c, int fa_interval, int kv_q8, int ctx,
                      int pf_B, u64 tok_bytes){
  if(!c || c->n_layers<=0) return 0;
  i32 nkv=c->n_kv_heads*c->head_dim;
  i32 nl = fa_interval>0 ? (c->n_layers+fa_interval-1)/fa_interval : c->n_layers;
  size_t qr=((size_t)nkv+31u)/32u*34u;
  u64 kv = kv_q8 ? 2u*(u64)nl*(u64)ctx*qr
                 : 2u*(u64)nl*(u64)ctx*(u64)nkv*4u;
  i32 dim=c->dim, hid=c->hidden_dim;
  i32 maxn=dim>hid?dim:hid; if(c->vocab>maxn) maxn=c->vocab;
  u64 nbuf=(u64)dim*3u+(u64)hid*2u+(u64)c->n_heads*c->head_dim+(u64)nkv*2u+(u64)c->n_heads*(u64)ctx+(u64)maxn;
  u64 buf=nbuf*4u;
  if(pf_B>0)
    buf += (u64)pf_B*(u64)(dim+dim+hid*2+c->n_heads*c->head_dim+nkv*2+c->n_heads*ctx)*4u;
  return kv+buf+tok_bytes;
}
/* Tamaño de la KV cache (K+V) para un modo dado. */
u64 model_kv_bytes(Model *m, int q8){
  if(!m || m->c.n_layers<=0) return 0;
  i32 nkv=m->c.n_kv_heads*m->c.head_dim;
  i32 ctx=m->c.seq_len;
  if(q8) return 2u*(size_t)m->c.n_layers*ctx*kv_q8_rowsize(nkv);
  return  2u*(size_t)m->c.n_layers*ctx*(size_t)nkv*4u;
}
void model_ram_report(Model *m){
  if(!m) return;
  u64 rt=model_est_ram(m);
  u64 kv=model_kv_bytes(m,kv_is_q8(m));
  fprintf(stderr,
    "ram: pesos=%llu MB (%s; paginas reclamables) | runtime=%llu MB "
    "= KV(%s,ctx=%d)%s + buffers + tokenizer\n",
    (unsigned long long)(m->data_size>>20), m->use_mmap?"mmap":"memcpy",
    (unsigned long long)(rt>>20), kv_is_q8(m)?"Q8_0":"F32", m->ctx,
    m->use_swap?" (file-backed)":"");
  if(kv > ((u64)1<<30) && !kv_is_q8(m))
    fprintf(stderr,"ram: warning: F32 KV uses ~%llu MB; try --q8-kv or --swap <path>\n",
      (unsigned long long)(kv>>20));
}

/* Encaja el modelo en un presupuesto de RAM: primero KV→Q8_0, luego baja ctx. */
int model_auto_budget(Model *m, u64 max_ram){
  if(!m || !max_ram || m->c.n_layers<=0) return -1;
  int q8=kv_is_q8(m);
  i32 ctx=m->c.seq_len>0?m->c.seq_len:2048;
  if(!q8 && est_for(m,ctx,0)>max_ram){ q8=1; m->flags|=F_KV_Q8; }
  while(ctx>256 && est_for(m,ctx,q8)>max_ram) ctx/=2;
  if(est_for(m,ctx,q8)>max_ram)
    fprintf(stderr,"model: warning: even with Q8 KV and ctx=%d it does not fit in %llu MB (uses %llu MB)\n",
      ctx,(unsigned long long)(max_ram>>20),(unsigned long long)(est_for(m,ctx,q8)>>20));
  return model_set_ctx(m,ctx);
}

static void model_unmap(Model *m){
  if(!m || !m->data) return;
  if(m->use_mmap){
    os_unmap(&m->wmap);
    m->data=NULL; m->own_data=0; m->use_mmap=0;
  } else if(m->own_data){
    free(m->data); m->data=NULL; m->own_data=0;
  }
}

/* Load G2BX: prefer mmap of weight blob when possible; fallback to malloc+fread. */
static int load_header_body(FILE *f, Model *m, const char *path){
  G2bxHeader h;
  int hrc=g2bx_read_header(f,&h);
  if(hrc==-2){ fprintf(stderr,"model: unsupported G2BX version %u\n",h.ver); return -1; }
  if(hrc==-3){ fprintf(stderr,"model: checksum mismatch (file corrupt or truncated)\n"); return -1; }
  if(hrc){ fprintf(stderr,"model: not G2BX\n"); return -1; }
  m->arch=h.arch; m->flags=h.flags; m->c=h.cfg;
  m->n_slots=h.n_slots; m->slots=h.slots; /* adopta el array */
  i64 header_end=(i64)h.data_start;
  memset(&h,0,sizeof h);

  u64 max_end=0;
  for(u32 i=0;i<m->n_slots;i++){
    u64 off=m->slots[i].off, nb=m->slots[i].nbytes;
    if(nb && off>UINT64_MAX-nb){ fprintf(stderr,"model: slot %u overflow\n",i); return -1; }
    u64 e=off+nb;
    if(e>max_end) max_end=e;
  }
  u64 aligned_end=ALIGN64(max_end);
  m->data_size=(size_t)aligned_end;

  if(header_end < 0) return -1;

  int used_mmap = 0;
  { /* blob de pesos por mmap (page cache evictable); fallback malloc+fread */
    OsMap om; os_map_init(&om);
    if(os_map_ro(path,&om)==0){
      if((size_t)header_end + m->data_size <= om.size){
        m->wmap=om;
        m->data=(u8*)om.view + (size_t)header_end;
        m->own_data=0; m->use_mmap=1; used_mmap=1;
        /* tokenizer starts after weight blob */
        os_fseek(f, header_end + (i64)m->data_size, SEEK_SET);
      } else os_unmap(&om);
    }
  }

  if(!used_mmap){
    m->use_mmap=0;
    m->data=malloc(m->data_size);
    m->own_data=1;
    if(!m->data || fread(m->data,1,m->data_size,f)!=m->data_size){
      fprintf(stderr,"model: truncated\n"); return -1;
    }
  }

  if(m->c.seq_len<=0) m->c.seq_len=2048;

  /* validación de geometría: sin esto, GQA y KV Q8 leen fuera de rango en silencio */
  if(m->c.n_heads<=0 || m->c.dim<=0 || m->c.vocab<=0 || m->c.n_layers<=0
     || m->c.n_layers>1024 || m->c.hidden_dim<=0){
    fprintf(stderr,"model: invalid geometry (heads=%d dim=%d vocab=%d L=%d hid=%d)\n",
      m->c.n_heads,m->c.dim,m->c.vocab,m->c.n_layers,m->c.hidden_dim);
    return -1;
  }
  if(m->c.n_kv_heads<=0) m->c.n_kv_heads=m->c.n_heads;
  if(m->c.n_kv_heads<=0 || m->c.n_heads % m->c.n_kv_heads){
    fprintf(stderr,"model: n_heads=%d not divisible by n_kv_heads=%d - invalid GQA\n",
      m->c.n_heads,m->c.n_kv_heads);
    return -1;
  }
  if(m->c.head_dim<=0) m->c.head_dim=m->c.dim/m->c.n_heads;
  if(m->c.head_dim<=0){ fprintf(stderr,"model: invalid head_dim\n"); return -1; }
  if(m->c.head_dim % 32){
    m->no_kv_q8=1; /* el slice por head no cae en bloque Q8 */
    fprintf(stderr,"model: head_dim=%d not a multiple of 32 - forcing F32 KV cache\n",m->c.head_dim);
  }
  if(m->arch==ARCH_QWEN35 && m->c.fa_interval>0){
    /* los arrays beta_v/g_v/dtb/av del forward son f32[HY_NV_MAX] en stack */
    if(m->c.ssm_dt_rank>HY_NV_MAX || m->c.ssm_dt_rank<=0){
      fprintf(stderr,"model: qwen35 ssm_dt_rank=%d fuera de rango [1..%d]\n",
        m->c.ssm_dt_rank,HY_NV_MAX);
      return -1;
    }
    if(m->c.ssm_inner % m->c.ssm_dt_rank){
      fprintf(stderr,"model: qwen35 ssm_inner=%d no divisible por dt_rank=%d\n",
        m->c.ssm_inner,m->c.ssm_dt_rank);
      return -1;
    }
  }

  build_index(m);
  if(model_set_ctx(m, m->c.seq_len)){
    fprintf(stderr,"model: OOM in runtime buffers\n"); return -1;
  }
  m->tok=NULL;
  Tokenizer *tk=malloc(sizeof(Tokenizer));
  if(tk && tok_read_section(f,tk)==0) m->tok=tk;
  else { free(tk); m->tok=NULL; }
  return 0;
}

int model_load_g2bx(const char *path, Model *m){
  memset(m,0,sizeof *m);
  os_map_init(&m->wmap);
  os_map_init(&m->swapmap);
  FILE *f=fopen(path,"rb");
  if(!f){ fprintf(stderr,"model: cannot open %s\n",path); return -1; }
  int rc=load_header_body(f,m,path);
  fclose(f);
  if(rc){ model_free(m); return -1; }
  m->src_path=strdup(path);
  if(!m->src_path){ model_free(m); return -1; }
  fprintf(stderr,"model: G2BX arch=%u dim=%d L=%d hd=%d vocab=%d flags=0x%02x mmap=%s\n",
    m->arch,m->c.dim,m->c.n_layers,m->c.head_dim,m->c.vocab,m->flags,
    m->use_mmap?"yes":"no");
  return 0;
}

int model_load_gguf(const char *path, Model *m){
  char cache[1024];
  snprintf(cache,sizeof cache,"%s.g2bx",path);
  struct stat ss, sc;
  int fresh=(stat(path,&ss)==0 && stat(cache,&sc)==0 && sc.st_mtime>=ss.st_mtime);
  if(!fresh){ if(g2bx_pack(path,cache)) return -1; }
  return model_load_g2bx(cache,m);
}

void model_free(Model *m){
  if(!m) return;
  if(m->ix_layer){
    for(i32 L=0;L<m->c.n_layers;L++) free(m->ix_layer[L]);
    free(m->ix_layer);
  }
  free(m->ix_global);
  free(m->slots);
  model_unmap(m);
  free_rt(m);
  if(m->lora_r){
   for(int L=0; L<m->c.n_layers; L++){ free(m->loraA_q[L]); free(m->loraB_q[L]); free(m->loraA_v[L]); free(m->loraB_v[L]); free(m->loraA_gate[L]); free(m->loraB_gate[L]); free(m->loraM_q[L]); free(m->loraM_v[L]); free(m->loraM_gate[L]); free(m->galore_m[L]); free(m->galore_v[L]); }
   free(m->loraA_q); free(m->loraB_q); free(m->loraA_v); free(m->loraB_v); free(m->loraA_gate); free(m->loraB_gate); free(m->loraM_q); free(m->loraM_v); free(m->loraM_gate); free(m->galore_m); free(m->galore_v);
  }
  free(m->swap_path); m->swap_path=NULL;
  free(m->src_path); m->src_path=NULL;
  if(m->tok){ tok_free(m->tok); free(m->tok); m->tok=NULL; }
  memset(m,0,sizeof *m);
  os_map_init(&m->wmap);
  os_map_init(&m->swapmap);
}

/* ── Calibración para poda estructurada ── */
int model_collect_stats(Model *m, const i32 *toks, i32 n){
  if(!m || !toks || n<=0) return -1;
  if(!m->ffn_stats)
    m->ffn_stats=calloc((size_t)m->c.n_layers*(size_t)m->c.hidden_dim,sizeof(f32));
  if(!m->ffn_stats) return -1;
  m->collect_stats=1;
  i32 pos=0;
  while(pos<n){
    i32 pb=m->pf_B>0?m->pf_B:8;
    i32 chunk=n-pos; if(chunk>pb) chunk=pb;
    if(model_prefill(m,toks+pos,chunk,pos,NULL))
      for(i32 j=0;j<chunk;j++)
        model_forward_ex(m,toks[pos+j],pos+j,NULL,0);
    pos+=chunk;
  }
  m->collect_stats=0;
  return 0;
}
void model_free_stats(Model *m){
  free(m->ffn_stats); m->ffn_stats=NULL; m->collect_stats=0;
}

/* ── ShortGPT: Block Influence = 1 - cos(entrada, salida) por bloque ── */
int model_autodrop(Model *m, const i32 *toks, i32 n, int ndrop){
  if(!m || !toks || n<8 || ndrop<=0) return -1;
  i32 nl=m->c.n_layers, dim=m->c.dim;
  free(m->skip_layer);
  m->skip_layer=calloc((size_t)nl,1);
  m->bi_pre=(f32*)malloc((size_t)nl*(size_t)dim*sizeof(f32));
  m->bi_dot=calloc((size_t)nl,sizeof(f32));
  m->bi_n2=calloc((size_t)nl,sizeof(f32));
  m->bi_n2p=calloc((size_t)nl,sizeof(f32));
  if(!m->skip_layer||!m->bi_pre||!m->bi_dot||!m->bi_n2||!m->bi_n2p) return -1;
  m->collect_bi=1;
  for(i32 p=0;p<n;p++) model_forward_ex(m,toks[p],p,NULL,0);
  m->collect_bi=0;
  typedef struct { f32 bi; i32 L; } BIS;
  BIS *bis=malloc((size_t)nl*sizeof(BIS));
  for(i32 L=0;L<nl;L++){
    f32 cosv = (m->bi_n2p[L]>0 && m->bi_n2[L]>0)?
      m->bi_dot[L]/sqrtf(m->bi_n2p[L]*m->bi_n2[L]) : 1.f;
    bis[L].bi=1.f-cosv; bis[L].L=L;
  }
  for(int a=0;a+1<nl;a++){ int b=a;
    for(int c2=a+1;c2<nl;c2++) if(bis[c2].bi<bis[b].bi) b=c2;
    if(b!=a){ BIS t=bis[a]; bis[a]=bis[b]; bis[b]=t; } }
  fprintf(stderr,"drop: Block Influence:");
  for(i32 L=0;L<nl;L++) fprintf(stderr," %d:%.3f",bis[L].L,(double)bis[L].bi);
  fprintf(stderr,"\n");
  for(int k=0;k<ndrop && k<nl;k++){
    m->skip_layer[bis[k].L]=1;
    fprintf(stderr,"drop: skipping block %d (BI=%.4f)\n",bis[k].L,(double)bis[k].bi);
  }
  free(bis);
  free(m->bi_pre); free(m->bi_dot); free(m->bi_n2); free(m->bi_n2p);
  m->bi_pre=m->bi_dot=m->bi_n2=m->bi_n2p=NULL;
  return 0;
}

static void pack_q4_0_const(u8 *dst, i32 n, f32 scale){
  i32 nb=n/32; u16 sh;
  union { f32 f; u32 u; } u; u.f=scale;
  u32 sign=(u.u>>16)&0x8000;
  i32 exp=((u.u>>23)&0xff)-127+15;
  u32 man=(u.u>>13)&0x3ff;
  if(exp<=0) sh=(u16)sign;
  else if(exp>=31) sh=(u16)(sign|0x7c00);
  else sh=(u16)(sign|(exp<<10)|man);
  for(i32 b=0;b<nb;b++){ memcpy(dst,&sh,2); dst+=2; memset(dst,0x88,16); dst+=16; }
}

int exp_synth_qwen_tiny(const char *out_path){
  ModelCfg c={.dim=64,.hidden_dim=128,.n_layers=2,.n_heads=4,.n_kv_heads=2,
              .vocab=128,.seq_len=64,.head_dim=16,.eps=1e-6f,.rope_theta=1000000.f};
  u8 arch=ARCH_QWEN3, flags=F_QK_NORM|F_TIE_EMBD;
  enum { MAXS=32 }; Slot slots[MAXS]; u32 ns=0; u8 *blobs[MAXS]; u32 bsz[MAXS];
#define ADD(r,ly,ty,ne) do{ \
  slots[ns].role=(u8)(r); slots[ns].layer=(u16)(ly); slots[ns].type=(u8)(ty); \
  u32 nb=(u32)ggml_type_size((ty),(ne)); slots[ns].nbytes=nb; \
  blobs[ns]=calloc(1,nb); bsz[ns]=nb; \
  if((ty)==T_Q4_0) pack_q4_0_const(blobs[ns],(i32)(ne),0.02f); \
  else if((ty)==T_F32){ for(u32 i=0;i<(ne);i++) ((f32*)blobs[ns])[i]=1.f; } \
  ns++; \
} while(0)
  ADD(R_TOK_EMBD,0xFFFF,T_Q4_0,(u64)c.vocab*c.dim);
  ADD(R_OUT_NORM,0xFFFF,T_F32,(u64)c.dim);
  for(i32 L=0;L<c.n_layers;L++){
    ADD(R_ATTN_NORM,L,T_F32,(u64)c.dim);
    ADD(R_ATTN_Q,L,T_Q4_0,(u64)c.n_heads*c.head_dim*c.dim);
    ADD(R_ATTN_K,L,T_Q4_0,(u64)c.n_kv_heads*c.head_dim*c.dim);
    ADD(R_ATTN_V,L,T_Q4_0,(u64)c.n_kv_heads*c.head_dim*c.dim);
    ADD(R_ATTN_O,L,T_Q4_0,(u64)c.dim*c.n_heads*c.head_dim);
    ADD(R_ATTN_Q_NORM,L,T_F32,(u64)c.head_dim);
    ADD(R_ATTN_K_NORM,L,T_F32,(u64)c.head_dim);
    ADD(R_FFN_NORM,L,T_F32,(u64)c.dim);
    ADD(R_FFN_GATE,L,T_Q4_0,(u64)c.hidden_dim*c.dim);
    ADD(R_FFN_UP,L,T_Q4_0,(u64)c.hidden_dim*c.dim);
    ADD(R_FFN_DOWN,L,T_Q4_0,(u64)c.dim*c.hidden_dim);
  }
#undef ADD
  u64 cur=g2bx_layout_slots(slots,ns);
  FILE *o=fopen(out_path,"w+b"); if(!o) return -1;
  int wrc=0;
  if(g2bx_write_header(o,arch,flags,&c,slots,ns)) wrc=-1;
  else if(g2bx_write_blob(o,slots,ns,blobs,bsz,cur)) wrc=-1;
  else if(g2bx_write_footer(o)) wrc=-1;
  fclose(o);
  for(u32 i=0;i<ns;i++) free(blobs[i]);
  if(wrc){ fprintf(stderr,"synth: write error on %s\n",out_path); return -1; }
  fprintf(stderr,"synth qwen-tiny -> %s (%llu B, %u slots)\n",
    out_path,(unsigned long long)cur,ns);
  return 0;
}
