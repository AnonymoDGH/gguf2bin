/* L5 — load G2BX + forward (Llama/Qwen2/Qwen3) — FIXED v3.3 */
#include "g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define G2BX_MAGIC "G2BX"
#define ALIGN64(x) (((x)+63ull)&~63ull)

u8 *slot_ptr(Model *m, Slot *s){ return s? m->data+s->off : NULL; }

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
static size_t kv_q8_rowsize(i32 nkv){ return ((size_t)nkv + 31u)/32u * 34u; }

static void q8_quant_row(const f32 *src, u8 *dst, i32 n){
  for(i32 off=0; off<n; off+=32){
    i32 m=(n-off)<32?(n-off):32;
    f32 amax=0; for(i32 j=0;j<m;j++){ f32 a=fabsf(src[off+j]); if(a>amax) amax=a; }
    f32 d=amax/127.f; if(!(d>0.f)) d=1e-10f;
    u16 sd=f32_to_half(d); memcpy(dst,&sd,2);
    for(i32 j=0;j<m;j++){ i32 q=(i32)lroundf(src[off+j]/d); if(q>127)q=127; else if(q<-127)q=-127; dst[2+j]=(u8)(i8)q; }
    for(i32 j=m;j<32;j++) dst[2+j]=0;
    dst+=34;
  }
}
static void q8_dequant_row(const u8 *src, f32 *out, i32 n){
  for(i32 off=0; off<n; off+=32){
    f32 s=half_to_float(*(const u16*)src); src+=2;
    i32 m=(n-off)<32?(n-off):32;
    for(i32 j=0;j<m;j++) out[off+j]=s*(i8)src[j];
    src+=32;
  }
}

#if defined(_WIN32)
/* Respaldar la KV cache en un archivo (p.ej. D:) -> páginas frías se vuelcan a disco. */
static int kv_swap_alloc(Model *m, size_t bytes){
  if(!m->swap_path || bytes==0) return -1;
  HANDLE f=CreateFileA(m->swap_path,GENERIC_READ|GENERIC_WRITE,0,NULL,
                       CREATE_ALWAYS,FILE_ATTRIBUTE_TEMPORARY,NULL);
  if(f==INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz; sz.QuadPart=(LONGLONG)bytes;
  if(!SetFilePointerEx(f,sz,NULL,FILE_BEGIN)||!SetEndOfFile(f)){ CloseHandle(f); return -1; }
  HANDLE mm=CreateFileMappingA(f,NULL,PAGE_READWRITE,(DWORD)((u64)bytes>>32),(DWORD)(bytes&0xffffffffu),NULL);
  if(!mm){ CloseHandle(f); return -1; }
  void *v=MapViewOfFile(mm,FILE_MAP_ALL_ACCESS,0,0,(SIZE_T)bytes);
  if(!v){ CloseHandle(mm); CloseHandle(f); return -1; }
  m->swap_f=(void*)f; m->swap_m=(void*)mm; m->swap_view=v; m->swap_size=bytes; m->use_swap=1;
  return 0;
}
static void kv_swap_free(Model *m){
  if(!m->use_swap || !m->swap_view) return;
  UnmapViewOfFile(m->swap_view);
  if(m->swap_m) CloseHandle((HANDLE)m->swap_m);
  if(m->swap_f) CloseHandle((HANDLE)m->swap_f);
  m->swap_view=NULL; m->swap_m=NULL; m->swap_f=NULL; m->swap_size=0; m->use_swap=0;
  if(m->swap_path) DeleteFileA(m->swap_path);
}
#else
static int kv_swap_alloc(Model *m, size_t bytes){
  if(!m->swap_path || bytes==0) return -1;
  int fd=open(m->swap_path,O_RDWR|O_CREAT|O_TRUNC,0600);
  if(fd<0) return -1;
  if(ftruncate(fd,(off_t)bytes)){ close(fd); return -1; }
  void *v=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(v==MAP_FAILED){ close(fd); return -1; }
  m->swap_fd=fd; m->swap_view=v; m->swap_size=bytes; m->use_swap=1;
  return 0;
}
static void kv_swap_free(Model *m){
  if(!m->use_swap || !m->swap_view) return;
  if(m->swap_size) munmap(m->swap_view,m->swap_size);
  if(m->swap_fd>=0) close(m->swap_fd);
  m->swap_view=NULL; m->swap_fd=-1; m->swap_size=0; m->use_swap=0;
  if(m->swap_path) unlink(m->swap_path);
}
#endif

static void free_rt(Model *m){
  free(m->buf); free(m->kvrow);
  m->buf=NULL; m->kvrow=NULL;
  if(m->use_swap && m->swap_view){ kv_swap_free(m); return; }
  free(m->kcache); free(m->vcache); free(m->kcq); free(m->vcq);
  m->kcache=NULL; m->vcache=NULL; m->kcq=NULL; m->vcq=NULL;
}
static int alloc_rt(Model *m, i32 ctx){
  ModelCfg *c=&m->c;
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd;
  int q8=(m->flags&F_KV_Q8)?1:0;
  i32 maxn=dim>hid?dim:hid; if(c->vocab>maxn) maxn=c->vocab;
  i32 nbuf=dim*3 + hid*2 + nq + nkv*2 + c->n_heads*ctx + maxn;
  m->buf=calloc((size_t)nbuf,sizeof(f32));
  m->kvrow=malloc((size_t)nkv*sizeof(f32));
  if(!m->buf || !m->kvrow) return -1;
  size_t half = 0, usize = 0;
  if(q8){ size_t qr=kv_q8_rowsize(nkv); half=(size_t)c->n_layers*(size_t)ctx*qr; }
  else   half=(size_t)c->n_layers*(size_t)ctx*(size_t)nkv*sizeof(f32);
  usize=2*half;
  if(m->swap_path && kv_swap_alloc(m,usize)==0){
    /* kcache y vcache son dos mitades de una vista contigua */
    u8 *base=(u8*)m->swap_view;
    if(q8){ m->kcq=base; m->vcq=base+half; m->kcache=NULL; m->vcache=NULL; }
    else  { m->kcache=(f32*)base; m->vcache=(f32*)(base+half); m->kcq=NULL; m->vcq=NULL; }
    return 0;
  }
  if(q8){
    m->kcq=malloc(half); m->vcq=malloc(half);
    if(!m->kcq || !m->vcq){ free_rt(m); return -1; }
  } else {
    m->kcache=malloc(half); m->vcache=malloc(half);
    if(!m->kcache || !m->vcache){ free_rt(m); return -1; }
  }
  return 0;
}

/* Respaldar la KV cache en un archivo (p.ej. D:) para que las páginas frías
   se vuelquen a disco y la RAM física quede para lo caliente. */
int model_enable_swap(Model *m, const char *path){
  if(!m || !path || !*path) return -1;
  free(m->swap_path); m->swap_path=NULL;
  m->swap_path=strdup(path);
  if(m->ctx>0) {
    if(model_set_ctx(m,m->ctx)){ free(m->swap_path); m->swap_path=NULL; return -1; }
  }
  fprintf(stderr,"swap: KV cache respaldada en %s\n", path);
  return 0;
}

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
  /* K y V: dos cachés */
  u64 kv = q8 ? 2u*(size_t)c->n_layers*ctx*kv_q8_rowsize(nkv)
              : 2u*(size_t)c->n_layers*ctx*(size_t)nkv*4u;
  i32 dim=c->dim, hid=c->hidden_dim;
  i32 maxn=dim>hid?dim:hid; if(c->vocab>maxn) maxn=c->vocab;
  u64 nbuf=(u64)dim*3u+(u64)hid*2u+(u64)c->n_heads*c->head_dim+(u64)nkv*2u+(u64)c->n_heads*ctx+(u64)maxn;
  u64 buf=nbuf*4u;
  u64 tok=0; Tokenizer *t=m->tok;
  if(t) tok=(u64)(t->n+(size_t)t->nmerges)*48u + ((size_t)1u<<19)*(8u+4u)*2u;
  return kv+buf+tok;
}
u64 model_est_ram(Model *m){
  if(!m || m->c.n_layers<=0) return 0;
  i32 ctx=m->ctx>0?m->ctx:m->c.seq_len;
  return est_for(m, ctx, (m->flags&F_KV_Q8)?1:0);
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
  u64 kv=model_kv_bytes(m,(m->flags&F_KV_Q8)?1:0);
  fprintf(stderr,
    "ram: pesos=%llu MB (%s; paginas reclamables) | runtime=%llu MB "
    "= KV(%s,ctx=%d)%s + buffers + tokenizer\n",
    (unsigned long long)(m->data_size>>20), m->use_mmap?"mmap":"memcpy",
    (unsigned long long)(rt>>20), (m->flags&F_KV_Q8)?"Q8_0":"F32", m->ctx,
    m->use_swap?" (file-backed)":"");
  if(kv > ((u64)1<<30) && !(m->flags&F_KV_Q8))
    fprintf(stderr,"ram: aviso — KV en F32 usa ~%llu MB; pruebe --q8-kv o --swap D:\\kv.swap\n",
      (unsigned long long)(kv>>20));
}

/* Encaja el modelo en un presupuesto de RAM: primero KV→Q8_0, luego baja ctx. */
int model_auto_budget(Model *m, u64 max_ram){
  if(!m || !max_ram || m->c.n_layers<=0) return -1;
  int q8=(m->flags&F_KV_Q8)?1:0;
  i32 ctx=m->c.seq_len>0?m->c.seq_len:2048;
  if(!q8 && est_for(m,ctx,0)>max_ram){ q8=1; m->flags|=F_KV_Q8; }
  while(ctx>256 && est_for(m,ctx,q8)>max_ram) ctx/=2;
  if(est_for(m,ctx,q8)>max_ram)
    fprintf(stderr,"model: aviso — ni con KV Q8 y ctx=%d cabe en %llu MB (usa %llu MB)\n",
      ctx,(unsigned long long)(max_ram>>20),(unsigned long long)(est_for(m,ctx,q8)>>20));
  return model_set_ctx(m,ctx);
}

static size_t kv_pos_offset(Model *m, i32 layer, i32 pos){
  return ((size_t)layer*m->ctx + (size_t)pos) * (size_t)m->c.n_kv_heads * (size_t)m->c.head_dim;
}
static void kv_store(Model *m, i32 layer, i32 pos, const f32 *k, const f32 *v){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim;
  if(m->flags&F_KV_Q8){
    size_t qr=kv_q8_rowsize(nkv);
    u8 *kb=m->kcq+((size_t)layer*m->ctx+(size_t)pos)*qr;
    u8 *vb=m->vcq+((size_t)layer*m->ctx+(size_t)pos)*qr;
    q8_quant_row(k,kb,nkv); q8_quant_row(v,vb,nkv);
  } else {
    f32 *kc=m->kcache+kv_pos_offset(m,layer,pos);
    f32 *vc=m->vcache+kv_pos_offset(m,layer,pos);
    memcpy(kc,k,(size_t)nkv*4); memcpy(vc,v,(size_t)nkv*4);
  }
}
static void kv_key_row(Model *m, i32 layer, i32 pos, f32 *out){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim;
  if(m->flags&F_KV_Q8){
    size_t qr=kv_q8_rowsize(nkv);
    q8_dequant_row(m->kcq+((size_t)layer*m->ctx+(size_t)pos)*qr, out, nkv);
  } else {
    memcpy(out, m->kcache+kv_pos_offset(m,layer,pos), (size_t)nkv*4);
  }
}
static void kv_val_row(Model *m, i32 layer, i32 pos, f32 *out){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim;
  if(m->flags&F_KV_Q8){
    size_t qr=kv_q8_rowsize(nkv);
    q8_dequant_row(m->vcq+((size_t)layer*m->ctx+(size_t)pos)*qr, out, nkv);
  } else {
    memcpy(out, m->vcache+kv_pos_offset(m,layer,pos), (size_t)nkv*4);
  }
}

static void model_unmap(Model *m){
  if(!m || !m->data) return;
  if(m->use_mmap){
#if defined(_WIN32)
    if(m->map_view){ UnmapViewOfFile(m->map_view); m->map_view=NULL; }
    if(m->map_handle){ CloseHandle((HANDLE)m->map_handle); m->map_handle=NULL; }
    if(m->file_handle){ CloseHandle((HANDLE)m->file_handle); m->file_handle=NULL; }
#else
    if(m->map_view && m->map_size) munmap(m->map_view, m->map_size);
    m->map_view=NULL; m->map_size=0;
    if(m->fd>=0){ close(m->fd); m->fd=-1; }
#endif
    m->data=NULL; m->own_data=0; m->use_mmap=0;
  } else if(m->own_data){
    free(m->data); m->data=NULL; m->own_data=0;
  }
}

/* Load G2BX: prefer mmap of weight blob when possible; fallback to malloc+fread. */
static int load_header_body(FILE *f, Model *m, const char *path){
  char magic[4]; u16 ver;
  if(fread(magic,1,4,f)!=4||memcmp(magic,G2BX_MAGIC,4)
     ||fread(&ver,2,1,f)!=1||fread(&m->arch,1,1,f)!=1||fread(&m->flags,1,1,f)!=1
     ||fread(&m->c,sizeof m->c,1,f)!=1||fread(&m->n_slots,4,1,f)!=1){
    fprintf(stderr,"model: no es G2BX\n"); return -1;
  }
  m->slots=malloc(m->n_slots*sizeof(Slot));
  if(!m->slots || fread(m->slots,sizeof(Slot),m->n_slots,f)!=m->n_slots) return -1;

  u64 max_end=0;
  for(u32 i=0;i<m->n_slots;i++){
    u64 e=m->slots[i].off+m->slots[i].nbytes;
    if(e>max_end) max_end=e;
  }
  u64 aligned_end=ALIGN64(max_end);
  m->data_size=(size_t)aligned_end;

  long header_end = ftell(f);
  if(header_end < 0) return -1;

  int used_mmap = 0;
#if defined(_WIN32)
  {
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, NULL);
    if(hf != INVALID_HANDLE_VALUE){
      LARGE_INTEGER fsz;
      if(GetFileSizeEx(hf, &fsz)){
        HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
        if(hm){
          void *view = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
          if(view){
            m->file_handle = (void*)hf;
            m->map_handle  = (void*)hm;
            m->map_view    = view;
            m->map_size    = (size_t)fsz.QuadPart;
            if((size_t)header_end + m->data_size <= m->map_size){
              m->data = (u8*)view + (size_t)header_end;
              m->own_data = 0;
              m->use_mmap = 1;
              used_mmap = 1;
              /* tokenizer starts after weight blob */
              fseek(f, header_end + (long)m->data_size, SEEK_SET);
            } else {
              UnmapViewOfFile(view); CloseHandle(hm); CloseHandle(hf);
              m->file_handle=m->map_handle=m->map_view=NULL; m->map_size=0;
            }
          } else { CloseHandle(hm); CloseHandle(hf); }
        } else CloseHandle(hf);
      } else CloseHandle(hf);
    }
  }
#else
  {
    int fd = open(path, O_RDONLY);
    if(fd >= 0){
      struct stat st;
      if(fstat(fd, &st)==0){
        void *view = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if(view != MAP_FAILED){
          m->fd = fd;
          m->map_view = view;
          m->map_size = (size_t)st.st_size;
          if((size_t)header_end + m->data_size <= m->map_size){
            m->data = (u8*)view + (size_t)header_end;
            m->own_data = 0;
            m->use_mmap = 1;
            used_mmap = 1;
            fseek(f, header_end + (long)m->data_size, SEEK_SET);
          } else {
            munmap(view, (size_t)st.st_size); close(fd);
            m->fd=-1; m->map_view=NULL; m->map_size=0;
          }
        } else close(fd);
      } else close(fd);
    }
  }
#endif

  if(!used_mmap){
    m->use_mmap=0;
    m->data=malloc(m->data_size);
    m->own_data=1;
    if(!m->data || fread(m->data,1,m->data_size,f)!=m->data_size){
      fprintf(stderr,"model: truncado\n"); return -1;
    }
  }

  if(m->c.seq_len<=0) m->c.seq_len=2048;

  build_index(m);
  if(model_set_ctx(m, m->c.seq_len)){
    fprintf(stderr,"model: OOM en buffers de runtime\n"); return -1;
  }
  m->tok=NULL;
  Tokenizer *tk=malloc(sizeof(Tokenizer));
  if(tk && tok_read_section(f,tk)==0) m->tok=tk;
  else { free(tk); m->tok=NULL; }
  return 0;
}

int model_load_g2bx(const char *path, Model *m){
  memset(m,0,sizeof *m);
#if !defined(_WIN32)
  m->fd = -1;
  m->swap_fd = -1;
#endif
  FILE *f=fopen(path,"rb");
  if(!f){ fprintf(stderr,"model: no abro %s\n",path); return -1; }
  int rc=load_header_body(f,m,path);
  fclose(f);
  if(rc){ model_free(m); return -1; }
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
  free(m->swap_path); m->swap_path=NULL;
  if(m->tok){ tok_free(m->tok); free(m->tok); m->tok=NULL; }
  memset(m,0,sizeof *m);
#if !defined(_WIN32)
  m->fd = -1;
  m->swap_fd = -1;
#endif
}

static void load_vec_f32(Model *m, Slot *s, f32 *dst, i32 n){
  if(!s){ memset(dst,0,(size_t)n*4); return; }
  if(s->type==T_F32){ memcpy(dst,slot_ptr(m,s),(size_t)n*4); return; }
  gguf_dequant(s->type, slot_ptr(m,s), dst, (u64)n);
}

static u64 row_stride(u32 type, i32 n){
  return (n/ggml_block_size(type))*ggml_type_bytes(type);
}

static void apply_rope(Model *m, f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta){
  /* Llama: interleaved pairs. Qwen2/Qwen3: NEOX half-split. */
  if(m->arch == ARCH_LLAMA) rope_th_llama(x, len, pos, head_dim, theta);
  else rope_th_neox(x, len, pos, head_dim, theta);
}

static int require_slot(Slot *s, const char *what, i32 layer){
  if(s) return 0;
  if(layer>=0) fprintf(stderr,"fwd: falta slot %s L%d\n", what, layer);
  else fprintf(stderr,"fwd: falta slot %s\n", what);
  return -1;
}

void model_forward(Model *m, i32 token, i32 pos, f32 *logits){
  ModelCfg *c=&m->c;
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd, ctx=m->ctx;
  i32 group = c->n_kv_heads>0 ? c->n_heads/c->n_kv_heads : 1;
  if(group<1) group=1;
  if(ctx<=0) ctx=c->seq_len;

  if(pos < 0 || pos >= ctx){
    static int warned;
    if(!warned){ fprintf(stderr,"fwd: pos=%d fuera de [0,%d); abort forward\n", pos, ctx); warned=1; }
    if(logits) memset(logits, 0, (size_t)c->vocab * sizeof(f32));
    return;
  }
  if(token < 0 || token >= c->vocab){
    static int warned;
    if(!warned){ fprintf(stderr,"fwd: token=%d fuera de vocab %d\n", token, c->vocab); warned=1; }
    if(logits) memset(logits, 0, (size_t)c->vocab * sizeof(f32));
    return;
  }
  if(!m->buf || (!m->kcache && !m->kcq)){
    fprintf(stderr,"fwd: runtime no inicializado\n"); return;
  }

  f32 *x=m->buf, *xb=x+dim, *xb2=xb+dim, *hb=xb2+dim, *hb2=hb+hid;
  f32 *q=hb2+hid, *k=q+nq, *v=k+nkv, *att=v+nkv, *row=att+(size_t)c->n_heads*ctx;

  Slot *emb=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(emb,"tok_embd",-1)) return;
  u8 *ep=slot_ptr(m,emb)+(size_t)token*row_stride(emb->type,dim);
  gguf_dequant(emb->type, ep, x, (u64)dim);

  for(i32 L=0;L<c->n_layers;L++){
    Slot *an=slot_get(m,R_ATTN_NORM,L);
    load_vec_f32(m,an,row,dim); rmsnorm(xb,x,row,dim,c->eps);

    Slot *wq=slot_get(m,R_ATTN_Q,L);
    Slot *wk=slot_get(m,R_ATTN_K,L);
    Slot *wv=slot_get(m,R_ATTN_V,L);
    if(require_slot(wq,"attn_q",L)||require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return;
    matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq,row);
    matmul_q(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,row);
    matmul_q(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,row);

    if(m->flags & F_QK_NORM){
      Slot *qn=slot_get(m,R_ATTN_Q_NORM,L);
      Slot *kn=slot_get(m,R_ATTN_K_NORM,L);
      f32 *qw=row;
      if(qn){ load_vec_f32(m,qn,qw,hd); qk_rmsnorm(q,qw,c->n_heads,hd,c->eps); }
      if(kn){ load_vec_f32(m,kn,qw,hd); qk_rmsnorm(k,qw,c->n_kv_heads,hd,c->eps); }
    }

    apply_rope(m, q, nq, pos, hd, c->rope_theta);
    apply_rope(m, k, nkv, pos, hd, c->rope_theta);

    kv_store(m, L, pos, k, v);

    f32 scale=1.f/sqrtf((f32)hd);
    /* Atención: dequant de la fila K/V una sola vez por posición (también F32). */
    for(i32 t=0;t<=pos;t++){
      kv_key_row(m,L,t,m->kvrow);
      for(i32 h=0;h<c->n_heads;h++){
        i32 kvh=h/group; f32 *qh=q+h*hd;
        f32 *kt=m->kvrow+kvh*hd;
        f32 s=0; for(i32 j=0;j<hd;j++) s+=qh[j]*kt[j];
        att[(size_t)h*ctx+t]=s*scale;
      }
    }
    for(i32 h=0;h<c->n_heads;h++) softmax(att+(size_t)h*ctx,pos+1);
    for(i32 j=0;j<nq;j++) q[j]=0;
    for(i32 t=0;t<=pos;t++){
      kv_val_row(m,L,t,m->kvrow);
      for(i32 h=0;h<c->n_heads;h++){
        i32 kvh=h/group; f32 a=att[(size_t)h*ctx+t];
        f32 *vh=m->kvrow+kvh*hd; f32 *qh=q+h*hd;
        for(i32 j=0;j<hd;j++) qh[j]+=a*vh[j];
      }
    }

    Slot *wo=slot_get(m,R_ATTN_O,L);
    if(require_slot(wo,"attn_o",L)) return;
    matmul_q(xb,q,slot_ptr(m,wo),wo->type,nq,dim,row);
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];

    Slot *fn=slot_get(m,R_FFN_NORM,L);
    load_vec_f32(m,fn,row,dim); rmsnorm(xb,x,row,dim,c->eps);

    Slot *wg=slot_get(m,R_FFN_GATE,L);
    Slot *wu=slot_get(m,R_FFN_UP,L);
    Slot *wd=slot_get(m,R_FFN_DOWN,L);
    if(require_slot(wg,"ffn_gate",L)||require_slot(wu,"ffn_up",L)||require_slot(wd,"ffn_down",L)) return;
    matmul_q(hb, xb,slot_ptr(m,wg),wg->type,dim,hid,row);
    matmul_q(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,row);
    silu(hb,hid);
    for(i32 i=0;i<hid;i++) hb[i]*=hb2[i];
    matmul_q(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,row);
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];
  }

  Slot *on=slot_get(m,R_OUT_NORM,-1);
  load_vec_f32(m,on,row,dim); rmsnorm(x,x,row,dim,c->eps);
  Slot *out=slot_get(m,R_OUTPUT,-1);
  if(!out) out=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(out,"output",-1)) return;
  matmul_q(logits,x,slot_ptr(m,out),out->type,dim,c->vocab,row);
}

i32 model_sample(f32 *logits, i32 n, f32 temp){
  if(n<=0) return 0;
  if(temp<=0.f){
    i32 bi=0; f32 bv=logits[0];
    for(i32 i=1;i<n;i++) if(logits[i]>bv){ bv=logits[i]; bi=i; }
    return bi;
  }
  for(i32 i=0;i<n;i++) logits[i]/=temp;
  softmax(logits,n);
  f32 r=(f32)rand()/(f32)RAND_MAX, c=0;
  for(i32 i=0;i<n;i++){ c+=logits[i]; if(r<=c) return i; }
  return n-1;
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
  u64 cur=0;
  for(u32 i=0;i<ns;i++){ slots[i].off=cur; cur=(cur+slots[i].nbytes+63ull)&~63ull; }
  FILE *o=fopen(out_path,"wb"); if(!o) return -1;
  u16 ver=1;
  fwrite("G2BX",1,4,o); fwrite(&ver,2,1,o); fwrite(&arch,1,1,o); fwrite(&flags,1,1,o);
  fwrite(&c,sizeof c,1,o); fwrite(&ns,4,1,o); fwrite(slots,sizeof(Slot),ns,o);
  u8 z[64]={0}; u64 w=0;
  for(u32 i=0;i<ns;i++){
    while(w<slots[i].off){ u64 p=slots[i].off-w; if(p>64)p=64; fwrite(z,1,(size_t)p,o); w+=p; }
    fwrite(blobs[i],1,bsz[i],o); w+=bsz[i];
  }
  while(w<cur){ u64 p=cur-w; if(p>64)p=64; fwrite(z,1,(size_t)p,o); w+=p; }
  fclose(o);
  for(u32 i=0;i<ns;i++) free(blobs[i]);
  fprintf(stderr,"synth qwen-tiny -> %s (%llu B, %u slots)\n",
    out_path,(unsigned long long)cur,ns);
  return 0;
}
