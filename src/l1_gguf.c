#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
/* L1 — parser GGUF mínimo (entrada cruda) — con bounds checking */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
/* SO (mmap/fopen) via os_mm; aquí no hay #ifdefs de plataforma. */
#define GGUF_MAGIC 0x46554747u
#define GGUF_MAX_TENSOR_NAME 1024
#define GGUF_HDR_SIZE 24
enum { MV_U8=0, MV_I8=1, MV_U16=2, MV_I16=3, MV_U32=4, MV_I32=5, MV_F32=6, MV_BOOL=7, MV_STR=8, MV_ARR=9, MV_U64=10, MV_I64=11, MV_F64=12 };
static u64 ru64(const u8 *p){ u64 v; memcpy(&v,p,8); return v; }
static u32 ru32(const u8 *p){ u32 v; memcpy(&v,p,4); return v; }
static u16 ru16(const u8 *p){ u16 v; memcpy(&v,p,2); return v; }
static int bounds(const u8 *p, const u8 *end, size_t need){
  return p && (size_t)(end-p) >= need;
}
static i64 meta_i(const u8 *p, u32 vt){
  switch(vt){
    case MV_U8:  return *p;
    case MV_I8:  return (i8)*p;
    case MV_U16: return ru16(p);
    case MV_I16: return (i16)ru16(p);
    case MV_U32: return ru32(p);
    case MV_I32: { i32 v; memcpy(&v,p,4); return v; }
    case MV_U64: return (i64)ru64(p);
    case MV_I64: { i64 v; memcpy(&v,p,8); return v; }
    case MV_F32: { f32 v; memcpy(&v,p,4); return (i64)v; }
    case MV_BOOL:return *p?1:0;
    default: return 0;
  }
}
static u64 meta_adv(u32 vt){
  if(vt==MV_BOOL||vt<=MV_I8) return 1;
  if(vt<=MV_I16) return 2;
  if(vt<=MV_I32||vt==MV_F32) return 4;
  return 8;
}
static u8 *skip_val(u8 *p, const u8 *end, u32 vt){
  if(vt==MV_STR){
    if(!bounds(p,end,8)) return NULL;
    u64 n=ru64(p);
    if(!bounds(p+8,end,n)) return NULL;
    return p+8+n;
  }
  if(vt==MV_ARR){
    if(!bounds(p,end,12)) return NULL; /* at:u32 + n:u64 */
    u32 at=ru32(p); p+=4;
    u64 n=ru64(p); p+=8;
    if(n > (1ull<<30)) return NULL; /* defensivo: arrays de >1B elementos */
    if(at==MV_STR){
      u8 *q=p;
      for(u64 i=0;i<n;i++){
        if(!bounds(q,end,8)) return NULL;
        u64 s=ru64(q); q+=8;
        if(!bounds(q,end,s)) return NULL;
        q+=s;
      }
      return q;
    }
    u64 adv=meta_adv(at);
    if(!adv || !bounds(p,end,adv*n)) return NULL;
    return p + adv*n;
  }
  u64 adv=meta_adv(vt);
  if(!adv||!bounds(p,end,adv)) return NULL;
  return p + adv;
}
int gguf_load(const char *path, GGUF *g){
  memset(g,0,sizeof *g);
  os_map_init(&g->map);
  u64 fsz=0;
  if(os_file_size(path,&fsz)){ fprintf(stderr,"gguf: cannot open %s\n",path); return -1; }
  if(fsz < GGUF_HDR_SIZE){ fprintf(stderr,"gguf: file too small\n"); return -1; }
  g->size=(size_t)fsz;
  /* mmap preferido: pack de modelos grandes sin copiarlos a RAM */
  if(os_map_ro(path,&g->map)==0){
    g->data=(u8*)g->map.view; g->own_data=2;
  } else {
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"gguf: cannot open %s\n",path); return -1; }
    g->data=malloc(g->size);
    if(!g->data||fread(g->data,1,g->size,f)!=g->size){ fclose(f); free(g->data); g->data=NULL; return -1; }
    fclose(f);
    g->own_data=1;
  }
  const u8 *end=g->data+g->size;
  u8 *p=g->data;
  if(!bounds(p,end,GGUF_HDR_SIZE)||ru32(p)!=GGUF_MAGIC){ fprintf(stderr,"gguf: invalid magic\n"); gguf_free(g); return -1; }
  g->version   = ru32(p+4);
  g->n_tensors = ru64(p+8);
  u64 meta_n   = ru64(p+16);
  if(g->n_tensors > (1ull<<20) || meta_n > (1ull<<20)){ fprintf(stderr,"gguf: corrupt header (tensors=%llu meta=%llu)\n",(unsigned long long)g->n_tensors,(unsigned long long)meta_n); gguf_free(g); return -1; }
  p+=GGUF_HDR_SIZE;
  g->alignment=32;
  for(u64 i=0;i<meta_n;i++){
    if(!bounds(p,end,8)){ fprintf(stderr,"gguf: truncated metadata\n"); gguf_free(g); return -1; }
    u64 klen=ru64(p); p+=8;
    if(klen>4096||!bounds(p,end,klen)){ fprintf(stderr,"gguf: corrupt key (klen=%llu)\n",(unsigned long long)klen); gguf_free(g); return -1; }
    char *key=(char*)p; p+=klen;
    if(!bounds(p,end,4)){ fprintf(stderr,"gguf: truncated metadata\n"); gguf_free(g); return -1; }
    u32 vt=ru32(p); p+=4;
    if(klen==17 && !memcmp(key,"general.alignment",17)){
      if(bounds(p,end,meta_adv(vt)))
        g->alignment=(u64)meta_i(p,vt);
    }
    if(!g->alignment) g->alignment=32;
    u8 *np=skip_val(p,end,vt);
    if(!np){ fprintf(stderr,"gguf: corrupt metadata at key=%.*s\n",(int)(klen<64?klen:64),key); gguf_free(g); return -1; }
    p=np;
  }
  g->t=calloc(g->n_tensors,sizeof(GTensor));
  if(!g->t){ gguf_free(g); return -1; }
  for(u64 i=0;i<g->n_tensors;i++){
    if(!bounds(p,end,8)){ fprintf(stderr,"gguf: truncated tensor name header\n"); goto fail_tensors; }
    u64 nlen=ru64(p); p+=8;
    if(nlen>GGUF_MAX_TENSOR_NAME||!bounds(p,end,nlen)){ fprintf(stderr,"gguf: corrupt tensor name (nlen=%llu)\n",(unsigned long long)nlen); goto fail_tensors; }
    g->t[i].name=malloc((size_t)nlen+1);
    if(!g->t[i].name) goto fail_tensors;
    memcpy(g->t[i].name,p,(size_t)nlen); g->t[i].name[nlen]=0; p+=(size_t)nlen;
    if(!bounds(p,end,4)) goto fail_tensors;
    g->t[i].n_dims=ru32(p); p+=4;
    if(g->t[i].n_dims>6){ fprintf(stderr,"gguf: dims=%u (corrupt)\n",g->t[i].n_dims); goto fail_tensors; }
    g->t[i].dims=malloc(g->t[i].n_dims*sizeof(u64));
    if(!g->t[i].dims) goto fail_tensors;
    if(!bounds(p,end,g->t[i].n_dims*8)) goto fail_tensors;
    for(u32 d=0;d<g->t[i].n_dims;d++){ g->t[i].dims[d]=ru64(p); p+=8; }
    if(!bounds(p,end,12)) goto fail_tensors;
    g->t[i].type  =ru32(p); p+=4;
    g->t[i].offset=ru64(p); p+=8;
  }
  u64 off=(u64)(p-g->data);
  g->data_off=(off+g->alignment-1)&~(g->alignment-1);
  return 0;
fail_tensors:
  gguf_free(g); return -1;
}
void gguf_free(GGUF *g){
  if(!g) return;
  if(g->t){
    for(u64 i=0;i<g->n_tensors;i++){ free(g->t[i].name); free(g->t[i].dims); }
    free(g->t); g->t=NULL;
  }
  if(g->own_data==2){
    os_unmap(&g->map);
  } else if(g->own_data==1){
    free(g->data);
  }
  g->data=NULL; g->own_data=0; memset(g,0,sizeof *g);
  os_map_init(&g->map);
}
u8 *gguf_tensor_ptr(GGUF *g, GTensor *t){
  if(!g||!t||!g->data) return NULL;
  u64 ne=1;
  for(u32 i=0;i<t->n_dims;i++){
    if(!t->dims || t->dims[i]==0 || ne>UINT64_MAX/t->dims[i]) return NULL;
    ne*=t->dims[i];
  }
  u64 sz=ggml_type_size(t->type,ne);
  u64 start=g->data_off+t->offset;
  if(start<g->data_off || (sz && start>SIZE_MAX-sz) || start+sz>g->size) return NULL;
  return g->data+start;
}
GTensor *gguf_by_name(GGUF *g, const char *name){
  for(u64 i=0;i<g->n_tensors;i++)
    if(!strcmp(g->t[i].name,name)) return &g->t[i];
  return NULL;
}
static u8 *meta_find(GGUF *g, const char *key, u32 *vt_out){
  if(!g->data||g->size<GGUF_HDR_SIZE) return NULL;
  const u8 *end=(u8*)g->data+g->size;
  u8 *p=g->data+GGUF_HDR_SIZE;
  u64 meta_n=ru64(g->data+16);
  size_t klen=strlen(key);
  for(u64 i=0;i<meta_n;i++){
    if(!bounds(p,end,8)) return NULL;
    u64 n=ru64(p); p+=8;
    if(!bounds(p,end,n)) return NULL;
    char *k=(char*)p; p+=n;
    if(!bounds(p,end,4)) return NULL;
    u32 vt=ru32(p); p+=4;
    if(n==klen && !memcmp(k,key,klen)){ if(vt_out)*vt_out=vt; return p; }
    u8 *np=skip_val(p,end,vt);
    if(!np) return NULL;
    p=np;
  }
  return NULL;
}
i64 gguf_meta_i64(GGUF *g, const char *key){
  u32 vt; u8 *v=meta_find(g,key,&vt);
  return v?meta_i(v,vt):0;
}
f32 gguf_meta_f32(GGUF *g, const char *key){
  u32 vt; u8 *v=meta_find(g,key,&vt);
  if(!v) return 0;
  if(vt==MV_F32){ f32 f; memcpy(&f,v,4); return f; }
  return (f32)meta_i(v,vt);
}
int gguf_meta_str(GGUF *g, const char *key, char *out, i32 outsz){
  u32 vt; u8 *v=meta_find(g,key,&vt);
  if(!v||vt!=MV_STR) return -1;
  u64 n=ru64(v); v+=8;
  i32 m=n<(u64)(outsz-1)?(i32)n:outsz-1;
  memcpy(out,v,m); out[m]=0; return 0;
}
u64 gguf_meta_arr_len(GGUF *g, const char *key){
  u32 vt; u8 *v=meta_find(g,key,&vt);
  if(!v || vt!=MV_ARR) return 0;
  u32 at=ru32(v); v+=4;
  (void)at;
  return ru64(v);
}
int gguf_meta_strarr(GGUF *g, const char *key, char ***out, u64 *n){
  u32 vt; u8 *v=meta_find(g,key,&vt);
  if(!v||vt!=MV_ARR) return -1;
  if(!bounds(v,(u8*)g->data+g->size,12)) return -1;
  u32 at=ru32(v); v+=4;
  u64 cnt=ru64(v); v+=8;
  if(at!=MV_STR) return -1;
  if(cnt==0){ *out=NULL; *n=0; return 0; }
  char **arr=calloc((size_t)cnt,sizeof(char*));
  if(!arr) return -1;
  const u8 *end=(u8*)g->data+g->size;
  for(u64 i=0;i<cnt;i++){
    if(!bounds(v,end,8)) goto fail;
    u64 s=ru64(v); v+=8;
    if(!bounds(v,end,s)) goto fail;
    char *b=malloc((size_t)s+1);
    if(!b) goto fail;
    memcpy(b,v,(size_t)s); b[s]=0; v+=s;
    arr[i]=b;
  }
  *out=arr; *n=cnt; return 0;
fail:
  for(u64 j=0;j<cnt;j++) free(arr[j]);
  free(arr); return -1;
}
