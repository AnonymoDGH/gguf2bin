#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
/* L6 — tokenizer BPE byte-level (Qwen2/3 GPT-2) */
#include "g2b.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
static u16 b2u[256]; static u8 u2b[289]; static int tbl_init=0;
static void tok_init_tables(void){
  /* llamada single-thread (carga de modelo); si se usa en multi-hilo agregar pthread_once */
  if(tbl_init) return; tbl_init=1;
  int used[256]; for(int i=0;i<256;i++) used[i]=0;
  for(int b=33;b<=126;b++) used[b]=1;
  for(int b=161;b<=172;b++) used[b]=1;
  for(int b=174;b<=255;b++) used[b]=1;
  int n=0; for(int b=0;b<256;b++){ if(used[b]){ b2u[b]=(u8)b; u2b[b]=(u8)b; } else { b2u[b]=(u16)(256+n); u2b[256+n]=(u8)b; n++; } }
}
static int cp_to_utf8(u32 cp, char *buf){
  if(cp<0x80){ buf[0]=(char)cp; return 1; }
  buf[0]=(char)(0xC0|(cp>>6)); buf[1]=(char)(0x80|(cp&0x3F)); return 2;
}
struct SHash { char **k; i32 *v; u32 cap, mask, cnt; };
static u32 shash_hash(const char *s){ u32 h=5381; while(*s) h=((h<<5)+h)+(u8)*s++; return h; }
static int shash_init(SHash *h, u32 cap){
  if(cap < 16) cap = 16;
  u32 c=16; while(c < cap) c <<= 1;
  h->cap=c; h->mask=c-1; h->cnt=0;
  h->k=calloc(c,sizeof(char*)); h->v=calloc(c,sizeof(i32));
  if(!h->k||!h->v){ free(h->k); free(h->v); h->k=NULL; h->v=NULL; h->cap=h->mask=h->cnt=0; return -1; }
  return 0;
}
static void shash_grow(SHash *h){
  if(!h || !h->k) return;
  if(h->cnt * 10 < h->cap * 7) return; /* load < 0.7 */
  u32 ocap=h->cap;
  char **ok=h->k; i32 *ov=h->v;
  u32 ncap = ocap << 1;
  if(ncap < ocap) return; /* overflow */
  h->k=calloc(ncap,sizeof(char*)); h->v=calloc(ncap,sizeof(i32));
  if(!h->k || !h->v){ free(h->k); free(h->v); h->k=ok; h->v=ov; return; }
  h->cap=ncap; h->mask=ncap-1; h->cnt=0;
  for(u32 i=0;i<ocap;i++){
    if(!ok[i]) continue;
    u32 j=shash_hash(ok[i])&h->mask;
    while(h->k[j]) j=(j+1)&h->mask;
    h->k[j]=ok[i]; h->v[j]=ov[i]; h->cnt++;
  }
  free(ok); free(ov);
}
static int shash_put(SHash *h, const char *k, i32 v){
  if(!h || !h->k || !k) return -1;
  shash_grow(h);
  if(!h->k) return -1;
  u32 i=shash_hash(k)&h->mask;
  while(h->k[i]){ if(!strcmp(h->k[i],k)){ h->v[i]=v; return 0; } i=(i+1)&h->mask; }
  h->k[i]=strdup(k);
  if(!h->k[i]) return -1;
  h->v[i]=v; h->cnt++;
  return 0;
}
static i32 shash_get(SHash *h, const char *k){
  if(!h||!h->k||!k) return -1;
  u32 i=shash_hash(k)&h->mask;
  while(h->k[i]){ if(!strcmp(h->k[i],k)) return h->v[i]; i=(i+1)&h->mask; }
  return -1;
}
static void shash_free(SHash *h){
  if(!h) return;
  if(h->k){ for(u32 i=0;i<h->cap;i++) free(h->k[i]); free(h->k); }
  free(h->v); h->k=NULL; h->v=NULL; h->cap=h->mask=h->cnt=0;
}
int tok_from_gguf(GGUF *g, Tokenizer *t){
  memset(t,0,sizeof *t); tok_init_tables();
  char **toks=NULL; u64 nt=0;
  if(gguf_meta_strarr(g,"tokenizer.ggml.tokens",&toks,&nt)) return -1;
  char **mg=NULL; u64 nm=0; gguf_meta_strarr(g,"tokenizer.ggml.merges",&mg,&nm);
  t->tok=toks; t->n=(i32)nt;
  t->bos=(i32)gguf_meta_i64(g,"tokenizer.ggml.bos_token_id");
  t->eos=(i32)gguf_meta_i64(g,"tokenizer.ggml.eos_token_id");
  t->unk=(i32)gguf_meta_i64(g,"tokenizer.ggml.unknown_token_id"); if(t->unk<0) t->unk=0;
  t->vocab=calloc(1,sizeof(SHash)); t->merges=calloc(1,sizeof(SHash));
  if(!t->vocab||!t->merges){ free(t->vocab); free(t->merges); t->vocab=t->merges=NULL; return -1; }
  if(shash_init(t->vocab,1u<<19)) return -1;
  for(i32 i=0;i<t->n;i++) if(shash_put(t->vocab,t->tok[i],i)) return -1;
  if(shash_init(t->merges,1u<<19)) return -1;
  t->mergestr=mg; t->nmerges=(i32)nm;
  for(i32 i=0;i<(i32)nm;i++){ char *s=mg[i]; char *sp=strchr(s,' '); if(!sp) continue; size_t al=(size_t)(sp-s), bl=strlen(sp+1); char *key=malloc(al+1+bl+1); if(!key) continue; memcpy(key,s,al); key[al]=1; memcpy(key+al+1,sp+1,bl); key[al+1+bl]=0; if(shash_put(t->merges,key,i)){ free(key); return -1; } free(key); }
  return 0;
}
int tok_write_section(FILE *f, Tokenizer *t){
  u32 nv=(u32)t->n, nm=(u32)t->nmerges; i32 b=t->bos, e=t->eos, u=t->unk;
  fwrite(&nv,4,1,f); fwrite(&nm,4,1,f); fwrite(&b,4,1,f); fwrite(&e,4,1,f); fwrite(&u,4,1,f);
  for(i32 i=0;i<t->n;i++){ u32 l=(u32)strlen(t->tok[i]); fwrite(&l,4,1,f); fwrite(t->tok[i],1,l,f); }
  for(i32 i=0;i<t->nmerges;i++){ u32 l=(u32)strlen(t->mergestr[i]); fwrite(&l,4,1,f); fwrite(t->mergestr[i],1,l,f); }
  return 0;
}
int tok_read_section(FILE *f, Tokenizer *t){
  memset(t,0,sizeof *t); tok_init_tables();
  t->vocab=calloc(1,sizeof(SHash)); t->merges=calloc(1,sizeof(SHash));
  if(!t->vocab||!t->merges){ free(t->vocab); free(t->merges); t->vocab=t->merges=NULL; return -1; }
  u32 nv,nm; i32 b,e,u;
  if(fread(&nv,4,1,f)!=1||fread(&nm,4,1,f)!=1||fread(&b,4,1,f)!=1||fread(&e,4,1,f)!=1||fread(&u,4,1,f)!=1){ tok_free(t); return -1; }
  if(nv>(1u<<20)||nm>(1u<<20)) return -1;
  t->n=(i32)nv; t->nmerges=(i32)nm; t->bos=b; t->eos=e; t->unk=u?u:0;
  t->tok=calloc(t->n?(size_t)t->n:1,sizeof(char*));
  if(!t->tok) return -1;
  for(i32 i=0;i<t->n;i++){ u32 l; if(fread(&l,4,1,f)!=1||l>(1u<<20)) goto fail; char *s=malloc((size_t)l+1); if(!s) goto fail; if(fread(s,1,l,f)!=l){ free(s); goto fail; } s[l]=0; t->tok[i]=s; }
  t->mergestr=calloc(t->nmerges?(size_t)t->nmerges:1,sizeof(char*));
  if(!t->mergestr) goto fail;
  for(i32 i=0;i<t->nmerges;i++){ u32 l; if(fread(&l,4,1,f)!=1||l>(1u<<20)) goto fail; char *s=malloc((size_t)l+1); if(!s) goto fail; if(fread(s,1,l,f)!=l){ free(s); goto fail; } s[l]=0; t->mergestr[i]=s; }
  if(shash_init(t->vocab,1u<<19)) goto fail;
  for(i32 i=0;i<t->n;i++) if(shash_put(t->vocab,t->tok[i],i)) goto fail;
  if(shash_init(t->merges,1u<<19)) goto fail;
  for(i32 i=0;i<t->nmerges;i++){ char *s=t->mergestr[i]; char *sp=strchr(s,' '); if(!sp) continue; size_t al=(size_t)(sp-s), bl=strlen(sp+1); char *key=malloc(al+1+bl+1); if(!key) continue; memcpy(key,s,al); key[al]=1; memcpy(key+al+1,sp+1,bl); key[al+1+bl]=0; if(shash_put(t->merges,key,i)){ free(key); goto fail; } free(key); }
  return 0;
fail:
  tok_free(t); return -1;
}
void tok_free(Tokenizer *t){
  if(!t) return;
  if(t->tok){ for(i32 i=0;i<t->n;i++) free(t->tok[i]); free(t->tok); }
  if(t->mergestr){ for(i32 i=0;i<t->nmerges;i++) free(t->mergestr[i]); free(t->mergestr); }
  shash_free(t->vocab); free(t->vocab); shash_free(t->merges); free(t->merges); memset(t,0,sizeof *t);
}
i32 tok_encode(Tokenizer *t, const char *text, i32 **out){
  const u8 *p=(const u8*)text; size_t L=strlen(text); char **sym=NULL; i32 ns=0, scap=0;
  for(size_t i=0;i<L;i++){ u32 cp=b2u[p[i]]; char buf[5]; int bl=cp_to_utf8(cp,buf); buf[bl]=0; if(ns>=scap){ scap=scap?scap*2:32; char **tmp=realloc(sym,(size_t)scap*sizeof(char*)); if(!tmp){ for(i32 j=0;j<ns;j++) free(sym[j]); free(sym); *out=NULL; return 0; } sym=tmp; } sym[ns++]=strdup(buf); if(!sym[ns-1]){ for(i32 j=0;j<ns-1;j++) free(sym[j]); free(sym); *out=NULL; return 0; } }
  i32 *slens = malloc((size_t)ns * sizeof(i32));
  if(!slens){ for(i32 i=0;i<ns;i++) free(sym[i]); free(sym); *out=NULL; return 0; }
  for(i32 i=0;i<ns;i++) slens[i] = (i32)strlen(sym[i]);
  while(ns>=2){
    i32 best=INT_MAX, bi=-1;
    for(i32 i=0;i+1<ns;i++){
      size_t kl = (size_t)slens[i] + 1 + (size_t)slens[i+1] + 1;
      char keybuf[256], *key = keybuf;
      if(kl > sizeof keybuf) key = malloc(kl);
      if(!key) continue;
      memcpy(key, sym[i], (size_t)slens[i]); key[slens[i]] = 1; memcpy(key+slens[i]+1, sym[i+1], (size_t)slens[i+1]); key[slens[i]+1+slens[i+1]] = 0;
      i32 r = shash_get(t->merges, key);
      if(key != keybuf) free(key);
      if(r>=0 && r<best){ best=r; bi=i; }
    }
    if(bi<0) break; size_t ml = (size_t)slens[bi] + (size_t)slens[bi+1] + 1; char *m = malloc(ml); if(!m) break; memcpy(m, sym[bi], (size_t)slens[bi]); memcpy(m+slens[bi], sym[bi+1], (size_t)slens[bi+1]); m[slens[bi]+slens[bi+1]] = 0; free(sym[bi]); free(sym[bi+1]); sym[bi]=m; slens[bi] = slens[bi] + slens[bi+1]; for(i32 k=bi+1;k+1<ns;k++){ sym[k]=sym[k+1]; slens[k]=slens[k+1]; } ns--;
  }
  free(slens);
  i32 *ids=malloc((size_t)(ns+1)*sizeof(i32));
  if(!ids){ for(i32 i=0;i<ns;i++) free(sym[i]); free(sym); *out=NULL; return 0; }
  i32 nid=0;
  for(i32 i=0;i<ns;i++){ i32 id=shash_get(t->vocab,sym[i]); ids[nid++]=(id>=0)?id:(t->unk>=0?t->unk:0); free(sym[i]); }
  free(sym); *out=ids; return nid;
}
char *tok_decode(Tokenizer *t, const i32 *ids, i32 n){
  size_t cap=256, len=0; char *out=malloc(cap);
  if(!out) return NULL;
  for(i32 i=0;i<n;i++){ const char *s=(ids[i]>=0&&ids[i]<t->n&&t->tok[ids[i]])?t->tok[ids[i]]:""; size_t sl=strlen(s); while(len+sl+1>cap){ cap*=2; char *tmp=realloc(out,cap); if(!tmp){ free(out); return NULL; } out=tmp; } memcpy(out+len,s,sl); len+=sl; }
  out[len]=0; size_t bcap=len+1; char *res=malloc(bcap);
  if(!res){ free(out); return NULL; }
  size_t rl=0; size_t i=0;
  while(i<len){ u8 c=(u8)out[i]; u32 cp; int adv;
    if(c<0x80){ cp=c; adv=1; }
    else if((c&0xE0)==0xC0 && i+1<len){ cp=((c&0x1F)<<6)|((u8)out[i+1]&0x3F); adv=2; }
    else if((c&0xF0)==0xE0 && i+2<len){ cp=((c&0x0F)<<12)|(((u8)out[i+1]&0x3F)<<6)|((u8)out[i+2]&0x3F); adv=3; }
    else { cp=c; adv=1; } i+=adv; u8 b = (cp<=288)? u2b[cp] : (u8)cp; if(rl+1>=bcap){ bcap*=2; char *tmp=realloc(res,bcap); if(!tmp){ free(res); free(out); return NULL; } res=tmp; } res[rl++]=(char)b; }
  res[rl]=0; free(out); return res;
}
i32 tok_id(Tokenizer *t, const char *s){ return shash_get(t->vocab, s); }
