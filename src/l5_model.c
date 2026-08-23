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
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define G2BX_MAGIC "G2BX"
#define ALIGN64(x) (((x)+63ull)&~63ull)
#if defined(_WIN32)
#define fseek64 _fseeki64
#else
#define fseek64(f, off, whence) fseeko(f, (off_t)(off), whence)
#endif

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
#if !defined(__AVX2__)
static void q8_dequant_row(const u8 *src, f32 *out, i32 n){
  for(i32 off=0; off<n; off+=32){
    f32 s=half_to_float(*(const u16*)src); src+=2;
    i32 m=(n-off)<32?(n-off):32;
    for(i32 j=0;j<m;j++) out[off+j]=s*(i8)src[j];
    src+=32;
  }
}
#endif

#if defined(__AVX2__)
void q8_dequant_row_avx2(const u8 *src, f32 *out, i32 n){
  for(i32 off=0; off<n; off+=32){
    f32 sf=half_to_float(*(const u16*)src); src+=2;
    __m256 vs=_mm256_set1_ps(sf);
    __m256i q8=_mm256_loadu_si256((const __m256i*)src); src+=32;
    __m128i lo=_mm256_castsi256_si128(q8), hi=_mm256_extracti128_si256(q8,1);
    i32 m=(n-off)<32?(n-off):32;
    if(m==32){
      _mm256_storeu_ps(out+off,   _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs));
      _mm256_storeu_ps(out+off+8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs));
      _mm256_storeu_ps(out+off+16,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs));
      _mm256_storeu_ps(out+off+24,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs));
    } else {
      /* tail: dequant completo y copia parcial */
      f32 tmp[32];
      _mm256_storeu_ps(tmp,   _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs));
      _mm256_storeu_ps(tmp+8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs));
      _mm256_storeu_ps(tmp+16,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs));
      _mm256_storeu_ps(tmp+24,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs));
      memcpy(out+off,tmp,(size_t)m*4);
    }
  }
}
#endif

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
  free(m->buf);
  m->buf=NULL;
  free(m->pf_pool);
  m->pf_pool=NULL;
  m->pf_x=m->pf_xb=m->pf_hb=m->pf_hb2=m->pf_q=m->pf_k=m->pf_v=m->pf_att=NULL;
  m->pf_B=0;
  free(m->ffn_stats); m->ffn_stats=NULL;
  free(m->conv_state); m->conv_state=NULL;
  if(m->use_swap && m->swap_view){ kv_swap_free(m); return; }
  free(m->kcache); free(m->vcache); free(m->kcq); free(m->vcq);
  m->kcache=NULL; m->vcache=NULL; m->kcq=NULL; m->vcq=NULL;
}
#define G2BX_PF_B 8
static int alloc_rt(Model *m, i32 ctx){
  ModelCfg *c=&m->c;
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd;
  int q8=((m->flags&F_KV_Q8) && !m->no_kv_q8)?1:0;
  i32 maxn=dim>hid?dim:hid; if(c->vocab>maxn) maxn=c->vocab;
  i32 nbuf=dim*3 + hid*2 + nq + nkv*2 + c->n_heads*ctx + maxn;
  m->buf=malloc((size_t)nbuf*sizeof(f32)); /* no calloc: se sobreescribe antes de leer */
  if(!m->buf) return -1;
  size_t half = 0, usize = 0;
  if(q8){ size_t qr=kv_q8_rowsize(nkv); half=(size_t)c->n_layers*(size_t)ctx*qr; }
  else   half=(size_t)c->n_layers*(size_t)ctx*(size_t)nkv*sizeof(f32);
  usize=2*half;
  if(m->swap_path && kv_swap_alloc(m,usize)==0){
    u8 *base=(u8*)m->swap_view;
    if(q8){ m->kcq=base; m->vcq=base+half; m->kcache=NULL; m->vcache=NULL; }
    else  { m->kcache=(f32*)base; m->vcache=(f32*)(base+half); m->kcq=NULL; m->vcq=NULL; }
  } else {
    if(q8){
      m->kcq=malloc(half); m->vcq=malloc(half);
      if(!m->kcq || !m->vcq){ free_rt(m); return -1; }
    } else {
      m->kcache=malloc(half); m->vcache=malloc(half);
      if(!m->kcache || !m->vcache){ free_rt(m); return -1; }
    }
  }
  /* buffers del prefill batcheado */
  i32 B=G2BX_PF_B;
  size_t nx=(size_t)B*(size_t)(dim + dim + hid*2 + nq + nkv*2)*sizeof(f32)
           +(size_t)B*(size_t)c->n_heads*(size_t)ctx*sizeof(f32);
  f32 *pool=malloc(nx);
  if(!pool){ /* sin prefill batcheado, pero el runtime secuencial funciona */ m->pf_B=0; return 0; }
  m->pf_pool=pool;
  m->pf_B=B;
  m->pf_x=pool; pool+=(size_t)B*dim;
  m->pf_xb=pool; pool+=(size_t)B*dim;
  m->pf_hb=pool; pool+=(size_t)B*hid;
  m->pf_hb2=pool; pool+=(size_t)B*hid;
  m->pf_q=pool; pool+=(size_t)B*nq;
  m->pf_k=pool; pool+=(size_t)B*nkv;
  m->pf_v=pool; pool+=(size_t)B*nkv;
  m->pf_att=pool;
#ifdef DBG_RT
  fprintf(stderr,"alloc_rt: buf=%p pool=%p B=%d nx=%.1fMB ctx=%d\n",(void*)m->buf,(void*)m->pf_pool,B,nx/1048576.0,ctx);
#endif
  if(m->arch==ARCH_LFM2)
    m->conv_state=calloc((size_t)c->n_layers*2u*(size_t)dim,sizeof(f32));
  return 0;
}

/* Respaldar la KV cache en un archivo (p.ej. D:) para que las páginas frías
   se vuelquen a disco y la RAM física quede para lo caliente. */
int model_enable_swap(Model *m, const char *path){
  if(!m || !path || !*path) return -1;
  free(m->swap_path); m->swap_path=NULL;
  m->swap_path=strdup(path);
  if(!m->swap_path){ fprintf(stderr,"swap: OOM\n"); return -1; }
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
static int kv_is_q8(const Model *m){ return (m->flags&F_KV_Q8) && !m->no_kv_q8; }
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
    fprintf(stderr,"ram: aviso — KV en F32 usa ~%llu MB; pruebe --q8-kv o --swap D:\\kv.swap\n",
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
    fprintf(stderr,"model: aviso — ni con KV Q8 y ctx=%d cabe en %llu MB (usa %llu MB)\n",
      ctx,(unsigned long long)(max_ram>>20),(unsigned long long)(est_for(m,ctx,q8)>>20));
  return model_set_ctx(m,ctx);
}

static size_t kv_pos_offset(Model *m, i32 layer, i32 pos){
  return ((size_t)layer*m->ctx + (size_t)pos) * (size_t)m->c.n_kv_heads * (size_t)m->c.head_dim;
}
static void kv_store(Model *m, i32 layer, i32 pos, const f32 *k, const f32 *v){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim;
  if(kv_is_q8(m)){
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
/* Dequant solo la fila de UN head (hd elementos), no la fila KV completa.
   head_dim múltiplo de 32 → el segmento cae en límite de bloque Q8
   (garantizado en carga: si no, no_kv_q8 fuerza KV F32). */
static void kv_key_row_h(Model *m, i32 layer, i32 pos, i32 kvh, f32 *out){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim, hd=m->c.head_dim;
  i32 eo=kvh*hd;
  if(kv_is_q8(m)){
    size_t qr=kv_q8_rowsize(nkv);
    const u8 *src=m->kcq+((size_t)layer*m->ctx+(size_t)pos)*qr + (size_t)(eo/32)*34u;
#if defined(__AVX2__)
    q8_dequant_row_avx2(src, out, hd);
#else
    q8_dequant_row(src, out, hd);
#endif
  } else {
    memcpy(out, m->kcache+kv_pos_offset(m,layer,pos)+(size_t)eo, (size_t)hd*4);
  }
}
static void kv_val_row_h(Model *m, i32 layer, i32 pos, i32 kvh, f32 *out){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim, hd=m->c.head_dim;
  i32 eo=kvh*hd;
  if(kv_is_q8(m)){
    size_t qr=kv_q8_rowsize(nkv);
    const u8 *src=m->vcq+((size_t)layer*m->ctx+(size_t)pos)*qr + (size_t)(eo/32)*34u;
#if defined(__AVX2__)
    q8_dequant_row_avx2(src, out, hd);
#else
    q8_dequant_row(src, out, hd);
#endif
  } else {
    memcpy(out, m->vcache+kv_pos_offset(m,layer,pos)+(size_t)eo, (size_t)hd*4);
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

  i64 header_end;
# if defined(_WIN32)
  header_end = _ftelli64(f);
# else
  header_end = (i64)ftello(f);
# endif
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
              fseek64(f, header_end + (i64)m->data_size, SEEK_SET);
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
            fseek64(f, header_end + (i64)m->data_size, SEEK_SET);
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

  /* validación de geometría: sin esto, GQA y KV Q8 leen fuera de rango en silencio */
  if(m->c.n_kv_heads<=0) m->c.n_kv_heads=m->c.n_heads;
  if(m->c.n_heads % m->c.n_kv_heads){
    fprintf(stderr,"model: n_heads=%d no divisible por n_kv_heads=%d — GQA inválida\n",
      m->c.n_heads,m->c.n_kv_heads);
    return -1;
  }
  if(m->c.head_dim<=0 && m->c.n_heads) m->c.head_dim=m->c.dim/m->c.n_heads;
  if(m->c.head_dim % 32){
    m->no_kv_q8=1; /* el slice por head no cae en bloque Q8 */
    fprintf(stderr,"model: head_dim=%d no múltiplo de 32 — KV cache forzada a F32\n",m->c.head_dim);
  }

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

/* dot y fma sobre head_dim: AVX2 (4 acumuladores) con fallback escalar.
   Igual matemática que la atención previa, ahora reutilizable por head. */
static inline f32 dot_hd(const f32 *a, const f32 *b, i32 n){
#if defined(__AVX2__)
  __m256 s0=_mm256_setzero_ps(), s1=_mm256_setzero_ps(), s2=_mm256_setzero_ps(), s3=_mm256_setzero_ps();
  i32 j=0;
  for(;j+31<n;j+=32){
    s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+j),   _mm256_loadu_ps(b+j),   s0);
    s1=_mm256_fmadd_ps(_mm256_loadu_ps(a+j+8), _mm256_loadu_ps(b+j+8), s1);
    s2=_mm256_fmadd_ps(_mm256_loadu_ps(a+j+16),_mm256_loadu_ps(b+j+16),s2);
    s3=_mm256_fmadd_ps(_mm256_loadu_ps(a+j+24),_mm256_loadu_ps(b+j+24),s3);
  }
  __m256 ss=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
  __m128 lo=_mm256_castps256_ps128(ss), hi=_mm256_extractf128_ps(ss,1);
  lo=_mm_add_ps(lo,hi); lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo)); lo=_mm_add_ss(lo,_mm_shuffle_ps(lo,lo,1));
  f32 s=_mm_cvtss_f32(lo);
  for(;j<n;j++) s+=a[j]*b[j];
  return s;
#else
  f32 s=0; for(i32 j=0;j<n;j++) s+=a[j]*b[j]; return s;
#endif
}
static inline void fma_hd(f32 *dst, const f32 *src, f32 scale, i32 n){
#if defined(__AVX2__)
  __m256 va=_mm256_set1_ps(scale);
  i32 j=0;
  for(;j+31<n;j+=32){
    _mm256_storeu_ps(dst+j,   _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j),   _mm256_loadu_ps(dst+j)));
    _mm256_storeu_ps(dst+j+8, _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+8), _mm256_loadu_ps(dst+j+8)));
    _mm256_storeu_ps(dst+j+16,_mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+16),_mm256_loadu_ps(dst+j+16)));
    _mm256_storeu_ps(dst+j+24,_mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+24),_mm256_loadu_ps(dst+j+24)));
  }
  for(;j<n;j++) dst[j]+=scale*src[j];
#else
  for(i32 j=0;j<n;j++) dst[j]+=scale*src[j];
#endif
}

void model_forward(Model *m, i32 token, i32 pos, f32 *logits){
  model_forward_ex(m,token,pos,logits,1);
}

/* ── LFM2 (LiquidAI): híbrido shortconv + atención ── */
static void forward_lfm2(Model *m, i32 token, i32 pos, f32 *logits, int want_logits){
  static i8 dbg=-1; if(dbg==-1){ dbg=getenv("G2BX_DBG")?1:0; }
  ModelCfg *c=&m->c;
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd, ctx=m->ctx;
  i32 group=c->n_kv_heads>0?c->n_heads/c->n_kv_heads:1;
  if(group<1) group=1;
  if(ctx<=0) ctx=c->seq_len;
  if(pos<0||pos>=ctx||token<0||token>=c->vocab||!m->buf) return;

  f32 *x=m->buf, *xb=x+dim, *xb2=xb+dim, *hb=xb2+dim, *hb2=hb+hid;
  f32 *q=hb2+hid, *k=q+nq, *v=k+nkv, *att=v+nkv, *row=att+(size_t)c->n_heads*ctx;

  Slot *emb=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(emb,"tok_embd",-1)) return;
  gguf_dequant(emb->type,slot_ptr(m,emb)+(size_t)token*row_stride(emb->type,dim),x,(u64)dim);
  /* OJO: LFM2 no normaliza tras el embedding; token_embd_norm es la norma FINAL */

  if(dbg&&pos<3){ f32 s=0; for(i32 i=0;i<dim;i++) s+=x[i]*x[i]; fprintf(stderr,"[dbg] pos=%d post-embdnorm |x|=%g\n",pos,sqrtf(s)); }

  for(i32 L=0;L<c->n_layers;L++){
      if(m->skip_layer && m->skip_layer[L]) continue; /* ShortGPT: bloque redundante */
    Slot *an=slot_get(m,R_ATTN_NORM,L);
    load_vec_f32(m,an,row,dim); rmsnorm(xb,x,row,dim,c->eps);
    if(m->collect_bi){ f32 *pre=m->bi_pre+(size_t)L*dim; f32 s=0;
      for(i32 i=0;i<dim;i++){ pre[i]=x[i]; s+=x[i]*x[i]; } m->bi_n2p[L]+=s; }
    if(dbg&&pos<1&&L<3){ f32 s=0,w=0,w2=0,xr=0; for(i32 i=0;i<dim;i++){ s+=xb[i]*xb[i]; w+=row[i]; w2+=row[i]*row[i]; xr+=x[i]*x[i]; }
      fprintf(stderr,"[dbg] L%d: |x_resid|=%g rms_w=%g |xb|=%g\n",L,sqrtf(xr),sqrtf(w2/dim),sqrtf(s)); }

    Slot *wq=slot_get(m,R_ATTN_Q,L);
    if(wq){
      Slot *wk=slot_get(m,R_ATTN_K,L);
      Slot *wv=slot_get(m,R_ATTN_V,L);
      if(require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return;
      matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq,row);
      matmul_q(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,row);
      matmul_q(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,row);
      if(dbg&&pos<1&&L==2){
        f32 sq0=0,sv=0,svx=0; for(i32 j=0;j<hd;j++) sq0+=q[j]*q[j];
        for(i32 j=0;j<nkv;j++){ sv+=v[j]*v[j]; svx+=fabsf(v[j]); }
        fprintf(stderr,"[dbg] pre-qknorm: |q_h0|=%g rms|v|=%g mean|v|=%g xb=%g %g %g %g\n",
          sqrtf(sq0),sqrtf(sv/nkv),svx/nkv,xb[0],xb[1],xb[2],xb[3]);
      }
      { Slot *qn=slot_get(m,R_ATTN_Q_NORM,L);
        Slot *kn=slot_get(m,R_ATTN_K_NORM,L);
        if(qn){ load_vec_f32(m,qn,row,hd); qk_rmsnorm(q,row,c->n_heads,hd,c->eps); }
        if(kn){ load_vec_f32(m,kn,row,hd); qk_rmsnorm(k,row,c->n_kv_heads,hd,c->eps); } }
      apply_rope(m,q,nq,pos,hd,c->rope_theta);
      apply_rope(m,k,nkv,pos,hd,c->rope_theta);
      kv_store(m,L,pos,k,v);
      f32 scale=1.f/sqrtf((f32)hd);
      {
        i32 ngrp=(c->n_heads+group-1)/group;
#if defined(_OPENMP)
        #pragma omp parallel if(ngrp>=2 && pos>=16)
#endif
        {
          f32 *kr=(f32*)malloc((size_t)hd*sizeof(f32));
          f32 *vr=(f32*)malloc((size_t)hd*sizeof(f32));
#if defined(_OPENMP)
          #pragma omp for schedule(static)
#endif
          for(i32 g2=0;g2<ngrp;g2++){
            i32 h0=g2*group, h1=h0+group; if(h1>c->n_heads) h1=c->n_heads;
            for(i32 t=0;t<=pos;t++){
              kv_key_row_h(m,L,t,h0/group,kr);
              for(i32 h=h0;h<h1;h++)
                att[(size_t)h*ctx+t]=dot_hd(q+(size_t)h*hd,kr,hd)*scale;
            }
            for(i32 h=h0;h<h1;h++) softmax(att+(size_t)h*ctx,pos+1);
            for(i32 h=h0;h<h1;h++){ f32*qh=q+(size_t)h*hd; for(i32 j=0;j<hd;j++) qh[j]=0.f; }
            for(i32 t=0;t<=pos;t++){
              kv_val_row_h(m,L,t,h0/group,vr);
              for(i32 h=h0;h<h1;h++)
                fma_hd(q+(size_t)h*hd,vr,att[(size_t)h*ctx+t],hd);
            }
          }
          free(kr); free(vr);
        }
      }
      Slot *wo=slot_get(m,R_ATTN_O,L);
      if(require_slot(wo,"attn_o",L)) return;
      matmul_q(xb,q,slot_ptr(m,wo),wo->type,nq,dim,row);
      { f32 s=0,sq=0; for(i32 i=0;i<dim;i++) s+=xb[i]*xb[i]; for(i32 j=0;j<hd;j++) sq+=q[j]*q[j];
        if(dbg&&pos<1&&L==2) fprintf(stderr,"[dbg] att-out |o|=%g |q_head0|=%g att0=%g v0=%g\n",sqrtf(s),sq,att[0],v[0]); }
      for(i32 i=0;i<dim;i++) x[i]+=xb[i];
    } else {
      /* ── bloque shortconv: B,C,x = in_proj(h); g=B*x; conv causal k=3; y=C*conv ── */
      Slot *ci=slot_get(m,R_CONV_IN,L);
      Slot *co=slot_get(m,R_CONV_OUT,L);
      Slot *cw=slot_get(m,R_CONV_W,L);
      if(require_slot(ci,"shortconv.in_proj",L)||require_slot(co,"shortconv.out_proj",L)||require_slot(cw,"shortconv.conv",L)) return;
      matmul_q(hb,xb,slot_ptr(m,ci),ci->type,dim,3*dim,row); /* B|C|x en hb[0..3dim) */
      f32 *Bp=hb, *Cp=hb+dim, *Xp=hb+2*dim;
      for(i32 i=0;i<dim;i++) xb2[i]=Bp[i]*Xp[i];
      f32 *cs=m->conv_state+(size_t)L*2*dim;
      load_vec_f32(m,cw,row,(u64)3*dim);
      const f32 *w=row; /* GGUF dims [k,ch]: elemento (k,ch) en ch*3+k */
      for(i32 ch=0;ch<dim;ch++){
        f32 y=w[ch*3+0]*cs[ch] + w[ch*3+1]*cs[dim+ch] + w[ch*3+2]*xb2[ch];
        cs[ch]=cs[dim+ch]; cs[dim+ch]=xb2[ch];
        xb2[ch]=y*Cp[ch];
      }
      matmul_q(xb,xb2,slot_ptr(m,co),co->type,dim,dim,row);
      if(dbg&&pos<1&&L==0){ f32 s=0; for(i32 i=0;i<dim;i++) s+=xb[i]*xb[i];
        fprintf(stderr,"[dbg] conv-branch-out |o|=%g o[0..4]=%g %g %g %g %g\n",sqrtf(s),xb[0],xb[1],xb[2],xb[3],xb[4]); }
      for(i32 i=0;i<dim;i++) x[i]+=xb[i];
    }

    /* FFN común */
    Slot *fn=slot_get(m,R_FFN_NORM,L);
    load_vec_f32(m,fn,row,dim); rmsnorm(xb,x,row,dim,c->eps);
    Slot *wg=slot_get(m,R_FFN_GATE,L);
    Slot *wu=slot_get(m,R_FFN_UP,L);
    Slot *wd=slot_get(m,R_FFN_DOWN,L);
    if(require_slot(wg,"ffn_gate",L)||require_slot(wu,"ffn_up",L)||require_slot(wd,"ffn_down",L)) return;
    if(wg->type==wu->type && slot_ptr(m,wu)==slot_ptr(m,wg)+wg->nbytes)
      matmul_q(hb,xb,slot_ptr(m,wg),wg->type,dim,hid*2,row);
    else {
      matmul_q(hb ,xb,slot_ptr(m,wg),wg->type,dim,hid,row);
      matmul_q(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,row);
    }
    silu_mul(hb, hb2, hid);
    if(m->collect_stats && m->ffn_stats){
      f32 *st=m->ffn_stats+(size_t)L*hid;
      for(i32 i=0;i<hid;i++) st[i]+=fabsf(hb[i]);
    }
    matmul_q(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,row);
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];
    if(m->collect_bi){ const f32 *pre=m->bi_pre+(size_t)L*dim; f32 dot=0,s=0;
      for(i32 i=0;i<dim;i++){ dot+=pre[i]*x[i]; s+=x[i]*x[i]; }
      m->bi_dot[L]+=dot; m->bi_n2[L]+=s; }
  }

  { /* LFM2 no tiene output_norm: reutiliza token_embd_norm al final */
    Slot *on=slot_get(m,R_OUT_NORM,-1);
    if(!on) on=slot_get(m,R_EMBD_NORM,-1);
    load_vec_f32(m,on,row,dim); rmsnorm(x,x,row,dim,c->eps); }
  if(dbg&&pos<4){ f32 s=0,mx=-1e30f,mn=1e30f; for(i32 i=0;i<dim;i++){ s+=x[i]*x[i]; if(x[i]>mx)mx=x[i]; if(x[i]<mn)mn=x[i]; }
    fprintf(stderr,"[dbg] final |x|=%g max=%g min=%g\n",sqrtf(s),mx,mn); }
  if(!want_logits || !logits) return;
  matmul_q(logits,x,slot_ptr(m,emb),emb->type,dim,c->vocab,row); /* head atado */
  if(dbg&&pos<4){ f32 mx=-1e30f,mn=1e30f; i32 b1=0,b2=0,b3=0;
    for(i32 i=0;i<c->vocab;i++){ if(logits[i]>mx){mx=logits[i];b3=b2;b2=b1;b1=i;} if(logits[i]<mn)mn=logits[i]; }
    fprintf(stderr,"[dbg] logits min=%g max=%g top=%d,%d,%d (%g %g %g)\n",mn,mx,b1,b2,b3,logits[b1],logits[b2],logits[b3]); }
}

static i32 max_row_scratch(Model *m){ return m->c.vocab>m->c.dim?m->c.vocab:m->c.dim; }

void model_forward_ex(Model *m, i32 token, i32 pos, f32 *logits, int want_logits){
  if(m->arch==ARCH_LFM2){ forward_lfm2(m,token,pos,logits,want_logits); return; }
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
      if(m->skip_layer && m->skip_layer[L]) continue; /* ShortGPT: bloque redundante */
    Slot *an=slot_get(m,R_ATTN_NORM,L);
    load_vec_f32(m,an,row,dim); rmsnorm(xb,x,row,dim,c->eps);
    if(m->collect_bi){ f32 *pre=m->bi_pre+(size_t)L*dim; f32 s=0;
      for(i32 i=0;i<dim;i++){ pre[i]=x[i]; s+=x[i]*x[i]; } m->bi_n2p[L]+=s; }

    Slot *wq=slot_get(m,R_ATTN_Q,L);
    Slot *wk=slot_get(m,R_ATTN_K,L);
    Slot *wv=slot_get(m,R_ATTN_V,L);
    if(require_slot(wq,"attn_q",L)||require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return;
    /* Fusion Q+K+V: solo si los slots son contiguos y del mismo tipo
       (los offsets densos lo permiten; con padding ALIGN64 no). */
    if(wq->type==wk->type && wk->type==wv->type
       && slot_ptr(m,wk)==slot_ptr(m,wq)+wq->nbytes
       && slot_ptr(m,wv)==slot_ptr(m,wk)+wk->nbytes)
      matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq+nkv+nkv,row);
    else {
      matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq,row);
      matmul_q(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,row);
      matmul_q(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,row);
    }

    /* Sesgos de atencion (exclusivo Qwen2.5): Q += b_q, K += b_k, V += b_v */
    { Slot *qb=slot_get(m,R_ATTN_Q_BIAS,L); if(qb){ load_vec_f32(m,qb,row,nq); for(i32 i=0;i<nq;i++) q[i]+=row[i]; }
      Slot *kb=slot_get(m,R_ATTN_K_BIAS,L); if(kb){ load_vec_f32(m,kb,row,nkv); for(i32 i=0;i<nkv;i++) k[i]+=row[i]; }
      Slot *vb=slot_get(m,R_ATTN_V_BIAS,L); if(vb){ load_vec_f32(m,vb,row,nkv); for(i32 i=0;i<nkv;i++) v[i]+=row[i]; } }

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
    /* Atención GQA-major: para cada grupo kv, cada fila K/V se dequantiza UNA vez
       y se reutiliza para todas las heads del grupo (antes: group× redundancia).
       Scratch K/V por hilo en heap (sin VLA en región OpenMP). */
    {
      i32 ngrp=(c->n_heads+group-1)/group;
#if defined(_OPENMP)
      #pragma omp parallel if(ngrp>=2 && pos>=16)
#endif
      {
        f32 *krow=(f32*)malloc((size_t)hd*sizeof(f32));
        f32 *vrow=(f32*)malloc((size_t)hd*sizeof(f32));
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for(i32 g=0; g<ngrp; g++){
          i32 h0=g*group, h1=h0+group; if(h1>c->n_heads) h1=c->n_heads;
          for(i32 t=0;t<=pos;t++){
            kv_key_row_h(m,L,t,h0/group,krow);
            for(i32 h=h0;h<h1;h++)
              att[(size_t)h*ctx+t]=dot_hd(q+(size_t)h*hd,krow,hd)*scale;
          }
          for(i32 h=h0;h<h1;h++) softmax(att+(size_t)h*ctx,pos+1);
          for(i32 h=h0;h<h1;h++){
            f32 *qh=q+(size_t)h*hd;
            for(i32 j=0;j<hd;j++) qh[j]=0.f;
          }
          for(i32 t=0;t<=pos;t++){
            kv_val_row_h(m,L,t,h0/group,vrow);
            for(i32 h=h0;h<h1;h++)
              fma_hd(q+(size_t)h*hd,vrow,att[(size_t)h*ctx+t],hd);
          }
        }
        free(krow); free(vrow);
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
    /* Fusion gate+up idem: solo con slots contiguos y mismo tipo. */
    if(wg->type==wu->type && slot_ptr(m,wu)==slot_ptr(m,wg)+wg->nbytes)
      matmul_q(hb,xb,slot_ptr(m,wg),wg->type,dim,hid*2,row);
    else {
      matmul_q(hb, xb,slot_ptr(m,wg),wg->type,dim,hid,row);
      matmul_q(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,row);
    }
    silu_mul(hb, hb2, hid); /* gate=silu(gate)*up fusionado */
    if(m->collect_stats && m->ffn_stats){
      f32 *st=m->ffn_stats+(size_t)L*hid;
      for(i32 i=0;i<hid;i++) st[i]+=fabsf(hb[i]);
    }
    matmul_q(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,row);
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];
  }

  Slot *on=slot_get(m,R_OUT_NORM,-1);
  load_vec_f32(m,on,row,dim); rmsnorm(x,x,row,dim,c->eps);
  if(!want_logits || !logits) return; /* prefill: saltar logits vocab×dim (carísimo) */
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
  /* xorshift64* (rand()/RAND_MAX=32767 en Windows destroza la distribución) */
  static u64 rs=0x9E3779B97F4A7C15ull;
  rs^=rs>>12; rs^=rs<<25; rs^=rs>>27;
  f32 r=(f32)(((rs*2685821657736338717ull)>>40)*(1.0/16777216.0));
  f32 c=0;
  for(i32 i=0;i<n;i++){ c+=logits[i]; if(r<=c) return i; }
  return n-1;
}

/* ── Prefill batcheado: B tokens por pasada de pesos ──
   Cada fila de pesos se lee UNA vez y se reusa para los B tokens del chunk
   (aritmética ×B); cada fila K/V se dequantiza una vez para todo el chunk.
   Matemáticamente equivalente al forward secuencial (mismo orden de suma). */
int model_prefill(Model *m, const i32 *toks, i32 n, i32 pos0, f32 *last_logits){
  if(!m || !toks || n<=0) return -1;
  if(m->arch==ARCH_LFM2) return 1; /* conv secuencial: usar el camino por token */
  ModelCfg *c=&m->c;
  if(!m->pf_B || !m->pf_x){ return 1; /* sin buffers: que el llamador use el secuencial */ }
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd, ctx=m->ctx;
  i32 group=c->n_heads/c->n_kv_heads; if(group<1) group=1;
  if(pos0<0 || pos0+n>ctx){
    fprintf(stderr,"prefill: rango [%d,%d) fuera de ctx=%d\n",pos0,pos0+n,ctx);
    return -1;
  }
  Slot *emb=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(emb,"tok_embd",-1)) return -1;

  while(n>0){
    i32 B=n>m->pf_B ? m->pf_B : n;
    if(pos0+B>ctx) B=ctx-pos0;
    if(B<=0) break;
    f32 *x=m->pf_x, *xb=m->pf_xb, *hb=m->pf_hb, *hb2=m->pf_hb2;
    f32 *q=m->pf_q, *k=m->pf_k, *v=m->pf_v, *att=m->pf_att;
    f32 *row=m->buf+dim*3+hid*2+nq+nkv*2+c->n_heads*ctx; /* zona 'row' de buf */

    for(i32 t=0;t<B;t++){
      if(toks[t]<0 || toks[t]>=c->vocab){ fprintf(stderr,"prefill: token %d fuera de vocab\n",toks[t]); return -1; }
      u8 *ep=slot_ptr(m,emb)+(size_t)toks[t]*row_stride(emb->type,dim);
      gguf_dequant(emb->type,ep,x+(size_t)t*dim,(u64)dim);
    }

    for(i32 L=0;L<c->n_layers;L++){
      if(m->skip_layer && m->skip_layer[L]) continue; /* ShortGPT */
      Slot *an=slot_get(m,R_ATTN_NORM,L);
      load_vec_f32(m,an,row,dim);
      for(i32 t=0;t<B;t++) rmsnorm(xb+(size_t)t*dim,x+(size_t)t*dim,row,dim,c->eps);

      Slot *wq=slot_get(m,R_ATTN_Q,L);
      Slot *wk=slot_get(m,R_ATTN_K,L);
      Slot *wv=slot_get(m,R_ATTN_V,L);
      if(require_slot(wq,"attn_q",L)||require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return -1;
      matmul_q_b(q,xb,slot_ptr(m,wq),wq->type,dim,nq,B);
      matmul_q_b(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,B);
      matmul_q_b(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,B);

      { Slot *qb=slot_get(m,R_ATTN_Q_BIAS,L);
        if(qb){ load_vec_f32(m,qb,row,nq); for(i32 t=0;t<B;t++){ f32 *qq=q+(size_t)t*nq; for(i32 i=0;i<nq;i++) qq[i]+=row[i]; } }
        Slot *kb=slot_get(m,R_ATTN_K_BIAS,L);
        if(kb){ load_vec_f32(m,kb,row,nkv); for(i32 t=0;t<B;t++){ f32 *kk=k+(size_t)t*nkv; for(i32 i=0;i<nkv;i++) kk[i]+=row[i]; } }
        Slot *vb=slot_get(m,R_ATTN_V_BIAS,L);
        if(vb){ load_vec_f32(m,vb,row,nkv); for(i32 t=0;t<B;t++){ f32 *vv=v+(size_t)t*nkv; for(i32 i=0;i<nkv;i++) vv[i]+=row[i]; } } }

      if(m->flags & F_QK_NORM){
        Slot *qn=slot_get(m,R_ATTN_Q_NORM,L);
        Slot *kn=slot_get(m,R_ATTN_K_NORM,L);
        if(qn){ load_vec_f32(m,qn,row,hd); for(i32 t=0;t<B;t++) qk_rmsnorm(q+(size_t)t*nq,row,c->n_heads,hd,c->eps); }
        if(kn){ load_vec_f32(m,kn,row,hd); for(i32 t=0;t<B;t++) qk_rmsnorm(k+(size_t)t*nkv,row,c->n_kv_heads,hd,c->eps); }
      }

      for(i32 t=0;t<B;t++){
        apply_rope(m,q+(size_t)t*nq,nq,pos0+t,hd,c->rope_theta);
        apply_rope(m,k+(size_t)t*nkv,nkv,pos0+t,hd,c->rope_theta);
        kv_store(m,L,pos0+t,k+(size_t)t*nkv,v+(size_t)t*nkv);
      }

      f32 scale=1.f/sqrtf((f32)hd);
      /* atención batched: scores [B][heads][ctx]; K/V dequant una vez por posición */
      {
        i32 ngrp=(c->n_heads+group-1)/group;
#if defined(_OPENMP)
        #pragma omp parallel if(ngrp>=2 && pos0+B>=8)
#endif
        {
          f32 *krow=(f32*)malloc((size_t)hd*sizeof(f32));
          f32 *vrow=(f32*)malloc((size_t)hd*sizeof(f32));
#if defined(_OPENMP)
          #pragma omp for schedule(static)
#endif
          for(i32 g=0;g<ngrp;g++){
            i32 h0=g*group, h1=h0+group; if(h1>c->n_heads) h1=c->n_heads;
            for(i32 p=0;p<pos0+B;p++){
              kv_key_row_h(m,L,p,h0/group,krow);
              i32 tmin = p>=pos0 ? p-pos0 : 0;
              for(i32 t=tmin;t<B;t++)
                for(i32 h=h0;h<h1;h++)
                  att[((size_t)t*c->n_heads+h)*ctx+p]=dot_hd(q+(size_t)t*nq+(size_t)h*hd,krow,hd)*scale;
            }
            for(i32 t=0;t<B;t++)
              for(i32 h=h0;h<h1;h++)
                softmax(att+((size_t)t*c->n_heads+h)*ctx,pos0+t+1);
            for(i32 t=0;t<B;t++)
              for(i32 h=h0;h<h1;h++){
                f32 *qh=q+(size_t)t*nq+(size_t)h*hd;
                for(i32 j=0;j<hd;j++) qh[j]=0.f;
              }
            for(i32 p=0;p<pos0+B;p++){
              kv_val_row_h(m,L,p,h0/group,vrow);
              i32 tmin = p>=pos0 ? p-pos0 : 0;
              for(i32 t=tmin;t<B;t++)
                for(i32 h=h0;h<h1;h++)
                  fma_hd(q+(size_t)t*nq+(size_t)h*hd,vrow,att[((size_t)t*c->n_heads+h)*ctx+p],hd);
            }
          }
          free(krow); free(vrow);
        }
      }

      Slot *wo=slot_get(m,R_ATTN_O,L);
      if(require_slot(wo,"attn_o",L)) return -1;
      matmul_q_b(xb,q,slot_ptr(m,wo),wo->type,nq,dim,B);
      for(i32 t=0;t<B;t++){
        f32 *xt=x+(size_t)t*dim, *xbt=xb+(size_t)t*dim;
        for(i32 i=0;i<dim;i++) xt[i]+=xbt[i];
      }

      Slot *fn=slot_get(m,R_FFN_NORM,L);
      load_vec_f32(m,fn,row,dim);
      for(i32 t=0;t<B;t++) rmsnorm(xb+(size_t)t*dim,x+(size_t)t*dim,row,dim,c->eps);

      Slot *wg=slot_get(m,R_FFN_GATE,L);
      Slot *wu=slot_get(m,R_FFN_UP,L);
      Slot *wd=slot_get(m,R_FFN_DOWN,L);
      if(require_slot(wg,"ffn_gate",L)||require_slot(wu,"ffn_up",L)||require_slot(wd,"ffn_down",L)) return -1;
      matmul_q_b(hb ,xb,slot_ptr(m,wg),wg->type,dim,hid,B);
      matmul_q_b(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,B);
      for(i32 t=0;t<B;t++) silu_mul(hb+(size_t)t*hid,hb2+(size_t)t*hid,hid);
      if(m->collect_stats && m->ffn_stats){
        f32 *st=m->ffn_stats+(size_t)L*hid;
        for(i32 t=0;t<B;t++){
          const f32 *h=hb+(size_t)t*hid;
          for(i32 i=0;i<hid;i++) st[i]+=fabsf(h[i]);
        }
      }
      matmul_q_b(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,B);
      for(i32 t=0;t<B;t++){
        f32 *xt=x+(size_t)t*dim, *xbt=xb+(size_t)t*dim;
        for(i32 i=0;i<dim;i++) xt[i]+=xbt[i];
      }
    }

    if(last_logits && n-B==0){ /* solo el último chunk contiene el token final */
      Slot *on=slot_get(m,R_OUT_NORM,-1);
      load_vec_f32(m,on,row,dim);
      rmsnorm(m->buf,x+(size_t)(B-1)*dim,row,dim,c->eps);
      Slot *out=slot_get(m,R_OUTPUT,-1);
      if(!out) out=emb;
      matmul_q(last_logits,m->buf,slot_ptr(m,out),out->type,dim,c->vocab,row);
      last_logits=NULL;
    }
    pos0+=B; toks+=B; n-=B;
  }
  return 0;
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
    i32 chunk=n-pos; if(chunk>8) chunk=8;
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
    fprintf(stderr,"drop: omitiendo bloque %d (BI=%.4f)\n",bis[k].L,(double)bis[k].bi);
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
