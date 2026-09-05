/* sampler.c — RNG xorshift64* + sampler (movido de g2b_api.c, matemática idéntica).
 * Nota: rand()/RAND_MAX=32767 en Windows destrozaba el muestreo; por eso xorshift propio.
 */
#include "internal/sampler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void sampler_init(Sampler *s){ memset(s,0,sizeof *s); }
void sampler_free(Sampler *s){
  if(!s) return;
  free(s->buf); free(s->seen);
  memset(s,0,sizeof *s);
}
void sampler_seed(Sampler *s, uint64_t seed){
  uint64_t r = seed ? seed : (0x9E3779B97F4A7C15ull ^ (uint64_t)time(NULL));
  for(int i=0;i<4;i++){ r^=r>>12; r^=r<<25; r^=r>>27; }
  s->rng.state=r;
}
float sampler_rnd(Sampler *s){
  uint64_t r=s->rng.state;
  r^=r>>12; r^=r<<25; r^=r>>27; s->rng.state=r;
  return (float)(((r*2685821657736338717ull)>>40)*(1.0/16777216.0)); /* [0,1) */
}

static void apply_repeat_penalty(Sampler *s, float *logits, int n,
                                 float penalty, const int32_t *recent, int recent_n){
  if(penalty==1.f || !recent || recent_n<=0) return;
  if(s->seen_n<n){ free(s->seen); s->seen=(uint8_t*)calloc((size_t)n,1); s->seen_n=s->seen?n:0; }
  if(!s->seen){
    for(int i=0;i<recent_n;i++){
      int id=recent[i]; if(id<0||id>=n) continue;
      float v=logits[id]; logits[id]=v>0?v/penalty:v*penalty;
    }
    return;
  }
  memset(s->seen,0,(size_t)n);
  for(int i=0;i<recent_n;i++){
    int id=recent[i]; if(id<0||id>=n||s->seen[id]) continue;
    s->seen[id]=1;
    float v=logits[id]; logits[id]=v>0?v/penalty:v*penalty;
  }
}

/* Quickselect descendente: deja los k mayores en a[0..k-1] (sin orden). O(n). */
static void topk_select(SampPair *a, int n, int k){
  if(k<=0 || k>=n) return;
  int lo=0, hi=n-1;
  while(lo<hi){
    int mid=lo+(hi-lo)/2;
    if(a[mid].logit>a[lo].logit){ SampPair t=a[mid]; a[mid]=a[lo]; a[lo]=t; }
    if(a[hi].logit>a[lo].logit){ SampPair t=a[hi]; a[hi]=a[lo]; a[lo]=t; }
    if(a[hi].logit>a[mid].logit){ SampPair t=a[hi]; a[hi]=a[mid]; a[mid]=t; }
    float p=a[mid].logit;
    int i=lo, j=hi;
    while(i<=j){
      while(a[i].logit>p) i++;
      while(a[j].logit<p) j--;
      if(i<=j){ SampPair t=a[i]; a[i]=a[j]; a[j]=t; i++; j--; }
    }
    if(k<=j) hi=j;
    else if(k>=i) lo=i;
    else break;
  }
}
static int cmp_pair_desc(const void *x, const void *y){
  float a=((const SampPair*)x)->logit, b=((const SampPair*)y)->logit;
  return (a<b)-(a>b);
}
/* Muestreo softmax exacto por truco Gumbel-max: argmax(l/T + Gumbel). O(n), sin exp. */
static int gumbel_sample(Sampler *s, const float *logits, int n){
  int bi=0; float bs=-1e30f;
  for(int i=0;i<n;i++){
    float u=sampler_rnd(s); if(u<1e-7f) u=1e-7f; else if(u>0.9999999f) u=0.9999999f;
    float v=logits[i]-logf(-logf(u));
    if(v>bs){ bs=v; bi=i; }
  }
  return bi;
}
int32_t sampler_sample(Sampler *s, float *logits, int n, float temp,
                       int top_k, float top_p, float repeat_penalty,
                       const int32_t *recent, int recent_n){
  if(n<=0) return 0;
  apply_repeat_penalty(s,logits,n,repeat_penalty,recent,recent_n);
  if(temp<=0.f){
    int bi=0; float bv=logits[0];
    for(int i=1;i<n;i++) if(logits[i]>bv){ bv=logits[i]; bi=i; }
    return bi;
  }
  for(int i=0;i<n;i++) logits[i]/=temp;
  if(top_k<=0 || top_k>=n){
    if(top_p>=1.f) return gumbel_sample(s,logits,n);
  }
  SampPair *arr=NULL;
  if(s->cap>=n) arr=s->buf;
  else {
    free(s->buf);
    s->buf=malloc((size_t)n*sizeof(*s->buf));
    s->cap=s->buf?n:0;
    arr=s->buf;
  }
  if(!arr) return gumbel_sample(s,logits,n);
  for(int i=0;i<n;i++){ arr[i].id=i; arr[i].logit=logits[i]; }

  int keep=n;
  if(top_k>0 && top_k<n){
    topk_select(arr,n,top_k);
    /* insertion sort del prefijo k (k pequeño: típico 20-100) */
    for(int i=1;i<top_k;i++){ SampPair v=arr[i]; int j=i-1; while(j>=0&&arr[j].logit<v.logit){ arr[j+1]=arr[j]; j--; } arr[j+1]=v; }
    keep=top_k;
  } else {
    qsort(arr,(size_t)n,sizeof(arr[0]),cmp_pair_desc);
  }
  float maxl=arr[0].logit, sum=0.f;
  for(int i=0;i<keep;i++){ float e=expf(arr[i].logit-maxl); arr[i].logit=e; sum+=e; }
  if(top_p<1.f){
    float cum=0.f; int last=keep;
    for(int i=0;i<keep;i++){ cum+=arr[i].logit/sum; if(cum>=top_p){ last=i+1; break; } }
    keep=last;
    sum=0.f; for(int i=0;i<keep;i++) sum+=arr[i].logit;
  }
  float r=sampler_rnd(s)*sum, cum=0.f;
  int32_t id=arr[keep-1].id;
  for(int i=0;i<keep;i++){ cum+=arr[i].logit; if(r<=cum){ id=arr[i].id; break; } }
  return id;
}
