#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
/* L1 — parser GGUF mínimo (entrada cruda) */
#include "g2b.h"
#include <stdlib.h>
#include <string.h>
#define GGUF_MAGIC 0x46554747u
enum { MV_U8=0, MV_I8=1, MV_U16=2, MV_I16=3, MV_U32=4, MV_I32=5, MV_F32=6, MV_BOOL=7, MV_STR=8, MV_ARR=9, MV_U64=10, MV_I64=11, MV_F64=12 };
static u64 ru64(const u8 *p){ u64 v; memcpy(&v,p,8); return v; }
static u32 ru32(const u8 *p){ u32 v; memcpy(&v,p,4); return v; }
static u16 ru16(const u8 *p){ u16 v; memcpy(&v,p,2); return v; }
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
static u8 *skip_val(u8 *p, u32 vt){
  if(vt==MV_STR){ u64 n=ru64(p); return p+8+n; }
  if(vt==MV_ARR){
    u32 at=ru32(p); p+=4;
    u64 n=ru64(p); p+=8;
    if(at==MV_STR){ for(u64 i=0;i<n;i++){ u64 s=ru64(p); p+=8+s; } }
    else p += meta_adv(at)*n;
    return p;
  }
  return p + meta_adv(vt);
}
int gguf_load(const char *path, GGUF *g){
  memset(g,0,sizeof *g);
  FILE *f=fopen(path,"rb");
  if(!f){ fprintf(stderr,"gguf: no abro %s\n",path); return -1; }
# if defined(_WIN32)
  _fseeki64(f,0,SEEK_END); g->size=(size_t)_ftelli64(f); _fseeki64(f,0,SEEK_SET);
# else
  fseeko(f,0,SEEK_END); g->size=(size_t)ftello(f); fseeko(f,0,SEEK_SET);
# endif
  g->data=malloc(g->size);
  if(!g->data||fread(g->data,1,g->size,f)!=g->size){ fclose(f); free(g->data); return -1; }
  fclose(f);
  u8 *p=g->data;
  if(ru32(p)!=GGUF_MAGIC){ fprintf(stderr,"gguf: magic inválido\n"); return -1; }
  g->version   = ru32(p+4);
  g->n_tensors = ru64(p+8);
  u64 meta_n   = ru64(p+16);
  p+=24;
  g->alignment=32;
  for(u64 i=0;i<meta_n;i++){
    u64 klen=ru64(p); p+=8;
    char *key=(char*)p; p+=klen;
    u32 vt=ru32(p); p+=4;
    if(klen==16 && !memcmp(key,"general.alignment",16))
      g->alignment=(u64)meta_i(p,vt);
    if(!g->alignment) g->alignment=32; /* defensivo */
    p=skip_val(p,vt);
  }
  g->t=calloc(g->n_tensors,sizeof(GTensor));
  if(!g->t) return -1;
for(u64 i=0;i<g->n_tensors;i++){
    u64 nlen=ru64(p); p+=8;
    if(nlen>1024){ fprintf(stderr,"gguf: nombre de tensor corrupto (nlen=%llu)\n",(unsigned long long)nlen); free(g->t); g->t=NULL; return -1; }
    g->t[i].name=malloc((size_t)nlen+1);
    memcpy(g->t[i].name,p,(size_t)nlen); g->t[i].name[nlen]=0; p+=(size_t)nlen;
    g->t[i].n_dims=ru32(p); p+=4;
    if(g->t[i].n_dims>6){ fprintf(stderr,"gguf: dims=%u (corrupto)\n",g->t[i].n_dims); free(g->t[i].name); free(g->t); g->t=NULL; return -1; }
    g->t[i].dims=malloc(g->t[i].n_dims*sizeof(u64));
    for(u32 d=0;d<g->t[i].n_dims;d++){ g->t[i].dims[d]=ru64(p); p+=8; }
    g->t[i].type  =ru32(p); p+=4;
    g->t[i].offset=ru64(p); p+=8;
  }
  u64 off=(u64)(p-g->data);
  g->data_off=(off+g->alignment-1)&~(g->alignment-1);
  return 0;
}
void gguf_free(GGUF *g){
  if(!g||!g->data) return;
  for(u64 i=0;i<g->n_tensors;i++){ free(g->t[i].name); free(g->t[i].dims); }
  free(g->t); free(g->data); memset(g,0,sizeof *g);
}
u8 *gguf_tensor_ptr(GGUF *g, GTensor *t){ return g->data+g->data_off+t->offset; }
GTensor *gguf_by_name(GGUF *g, const char *name){
  for(u64 i=0;i<g->n_tensors;i++)
    if(!strcmp(g->t[i].name,name)) return &g->t[i];
  return NULL;
}
static u8 *meta_find(GGUF *g, const char *key, u32 *vt_out){
  u8 *p=g->data+24;
  u64 meta_n=ru64(g->data+16);
  size_t klen=strlen(key);
  for(u64 i=0;i<meta_n;i++){
    u64 n=ru64(p); p+=8;
    char *k=(char*)p; p+=n;
    u32 vt=ru32(p); p+=4;
    if(n==klen && !memcmp(k,key,klen)){ if(vt_out)*vt_out=vt; return p; }
    p=skip_val(p,vt);
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
  u32 at=ru32(v); v+=4;
  u64 cnt=ru64(v); v+=8;
  if(at!=MV_STR) return -1;
  char **arr=malloc((size_t)(cnt?cnt:1)*sizeof(char*));
  for(u64 i=0;i<cnt;i++){
    u64 s=ru64(v); v+=8;
    char *b=malloc((size_t)s+1);
    memcpy(b,v,(size_t)s); b[s]=0; v+=s;
    arr[i]=b;
  }
  *out=arr; *n=cnt; return 0;
}
