#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "g2b.h"

/*
 * Puente JNI para el núcleo gguf2bin2 en Android.
 * v2: carga .g2bx, generación con streaming, muestreo top-k + penalización
 *     de repetición (igual que la CLI) y ajuste de hilos OpenMP.
 */

#define DEFAULT_CTX 2048

typedef struct {
  Model m;
  volatile int live;
  volatile int cancel;
} JModel;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static u64 jrng = 0x9E3779B97F4A7C15ull;
static f32 jrandf(void){
  jrng^=jrng>>12; jrng^=jrng<<25; jrng^=jrng>>27;
  return (f32)(((jrng*2685821657736338717ull)>>40)*(1.0/16777216.0));
}

/* ── muestreo con top-k + repeat penalty (adaptado de main.c) ── */
static void apply_rep_penalty(f32 *logits, i32 n, const i32 *recent, int rn, f32 pen){
  if(pen <= 1.f) return;
  for(int i=0;i<rn;i++){
    i32 t=recent[i];
    if(t<0 || t>=n) continue;
    if(logits[t]>0) logits[t]/=pen;
    else logits[t]*=pen;
  }
}
static i32 quickselect_k(f32 *a, i32 n, i32 k){ /* k-ésimo mayor, O(n) */
  i32 lo=0, hi=n-1;
  while(lo<hi){
    f32 pivot=a[(lo+hi)>>1]; i32 i=lo,j=hi;
    while(i<=j){
      while(a[i]>pivot)i++;
      while(a[j]<pivot)j--;
      if(i<=j){f32 t=a[i];a[i]=a[j];a[j]=t;i++;j--;}
    }
    if(k<=j) hi=j; else if(k>=i) lo=i; else break;
  }
  return k;
}
static i32 sample_topk(f32 *logits, i32 n, f32 temp, int top_k,
                       const i32 *recent, int rn, f32 rep_pen){
  if(temp <= 0.05f){ /* greedy */
    i32 bi=0; f32 bv=logits[0];
    for(i32 i=1;i<n;i++) if(logits[i]>bv){bv=logits[i];bi=i;}
    return bi;
  }
  apply_rep_penalty(logits,n,recent,rn,rep_pen);
  if(top_k<1||top_k>n) top_k=n;
  f32 *tmp=(f32*)malloc((size_t)n*sizeof(f32));
  if(!tmp){ i32 bi=0; f32 bv=logits[0]; for(i32 i=1;i<n;i++) if(logits[i]>bv){bv=logits[i];bi=i;} return bi; }
  memcpy(tmp,logits,(size_t)n*sizeof(f32));
  quickselect_k(tmp,n,top_k-1);
  f32 thresh=tmp[top_k-1];
  free(tmp);
  double mx=-1e30; i32 cnt=0;
  for(i32 i=0;i<n;i++) if(logits[i]>=thresh){cnt++; if(logits[i]>mx)mx=logits[i];}
  if(!cnt) cnt=1;
  double sum=0;
  for(i32 i=0;i<n;i++) if(logits[i]>=thresh) sum+=exp((double)(logits[i]/temp)-mx/temp);
  double r=(double)jrandf()*sum, c=0;
  i32 last=0;
  for(i32 i=0;i<n;i++){
    if(logits[i]<thresh) continue;
    c+=exp((double)(logits[i]/temp)-mx/temp);
    last=i;
    if(r<=c) return i;
  }
  return last;
}

JNIEXPORT jlong JNICALL
Java_com_gguf2bin_app_Native_loadModel(JNIEnv *env, jclass cl, jstring path, jint ctx) {
  const char *p = (*env)->GetStringUTFChars(env, path, NULL);
  if (!p) return 0;
  JModel *j = (JModel *)calloc(1, sizeof(JModel));
  if (!j) { (*env)->ReleaseStringUTFChars(env, path, p); return 0; }
  int rc = model_load_g2bx(p, &j->m);
  (*env)->ReleaseStringUTFChars(env, path, p);
  if (rc) { free(j); return 0; }
  model_set_ctx(&j->m, ctx > 0 ? ctx : DEFAULT_CTX);
  j->live = 1;
  return (jlong)(intptr_t)j;
}

JNIEXPORT void JNICALL
Java_com_gguf2bin_app_Native_requestCancel(JNIEnv *env, jclass cl, jlong ptr) {
  JModel *j = (JModel *)(intptr_t)ptr;
  if (j) j->cancel = 1;
}

JNIEXPORT void JNICALL
Java_com_gguf2bin_app_Native_freeModel(JNIEnv *env, jclass cl, jlong ptr) {
  JModel *j = (JModel *)(intptr_t)ptr;
  if (!j) return;
  j->cancel = 1;
  pthread_mutex_lock(&g_mu);
  if (j->live) { model_free(&j->m); j->live = 0; }
  pthread_mutex_unlock(&g_mu);
  free(j);
}

JNIEXPORT void JNICALL
Java_com_gguf2bin_app_Native_setThreads(JNIEnv *env, jclass cl, jint t) {
  extern void omp_set_num_threads(int);
  omp_set_num_threads(t > 0 ? t : 1);
}

#define REP_WINDOW 64

JNIEXPORT jstring JNICALL
Java_com_gguf2bin_app_Native_generate(JNIEnv *env, jclass cl, jlong ptr,
                                      jstring prompt, jint maxTokens,
                                      jfloat temp, jint topK, jobject sink) {
  JModel *j = (JModel *)(intptr_t)ptr;
  if (!j) return NULL;

  static jclass sink_cls = NULL;
  static jmethodID on_token = NULL;
  if (!sink_cls) {
    jclass c = (*env)->FindClass(env, "com/gguf2bin/app/Native$TokenSink");
    if (!c) return NULL;
    sink_cls = (*env)->NewGlobalRef(env, c);
    on_token = (*env)->GetMethodID(env, sink_cls, "onToken", "(Ljava/lang/String;)V");
    if (!on_token) return NULL;
  }

  pthread_mutex_lock(&g_mu);
  if (!j->live || j->cancel || !j->m.tok) { pthread_mutex_unlock(&g_mu); return NULL; }
  Model *m = &j->m;

  const char *ptext = (*env)->GetStringUTFChars(env, prompt, NULL);
  i32 *ids = NULL;
  i32 nt = tok_encode(m->tok, (char *)ptext, &ids);
  (*env)->ReleaseStringUTFChars(env, prompt, ptext);
  if (nt <= 0) { free(ids); pthread_mutex_unlock(&g_mu); return NULL; }

  if (m->tok->bos >= 0 && ids[0] != m->tok->bos) {
    i32 *tmp = (i32 *)malloc((size_t)(nt + 1) * sizeof(i32));
    if (tmp) { tmp[0] = m->tok->bos; memcpy(tmp + 1, ids, (size_t)nt * sizeof(i32)); free(ids); ids = tmp; nt++; }
  }

  i32 maxctx = m->ctx > 0 ? m->ctx : m->c.seq_len;
  f32 *logits = (f32 *)malloc((size_t)m->c.vocab * sizeof(f32));
  i32 *recent = (i32 *)malloc((size_t)REP_WINDOW * sizeof(i32));
  int rn = 0;
  char *out = NULL; size_t cap = 8192, len = 0;
  if (!logits || !recent || !(out=(char*)malloc(cap))) {
    free(logits); free(recent); free(ids); free(out);
    pthread_mutex_unlock(&g_mu); return NULL;
  }
  out[0]=0;

  i32 pos = 0;
  i32 pre = nt>1 ? nt-1 : 0;
  if (pre>0 && model_prefill(m, ids, pre, 0, NULL)==0) {
    for (i32 i=0;i<pre && rn<REP_WINDOW;i++) recent[rn++]=ids[i];
    if (pre>REP_WINDOW) {
      rn=REP_WINDOW;
      memcpy(recent, ids+pre-REP_WINDOW, (size_t)REP_WINDOW*sizeof(i32));
    }
    pos = pre;
  } else {
    for (i32 i = 0; i < nt-1; i++) {
      if (pos >= maxctx || j->cancel) break;
      model_forward_ex(m, ids[i], pos, NULL, 0);
      if(rn<REP_WINDOW) recent[rn++]=ids[i]; else { memmove(recent,recent+1,(REP_WINDOW-1)*sizeof(i32)); recent[REP_WINDOW-1]=ids[i]; }
      pos++;
    }
  }
  if (pos < maxctx && pos < nt && !j->cancel) {
    model_forward_ex(m, ids[nt-1], pos, logits, 1);
    if(rn<REP_WINDOW) recent[rn++]=ids[nt-1]; else { memmove(recent,recent+1,(REP_WINDOW-1)*sizeof(i32)); recent[REP_WINDOW-1]=ids[nt-1]; }
    pos++;
  }
  pthread_mutex_unlock(&g_mu);

  jstring result = NULL;
  for (i32 k = 0; k < maxTokens; k++) {
    pthread_mutex_lock(&g_mu);
    if (!j->live || j->cancel || pos >= maxctx) { pthread_mutex_unlock(&g_mu); break; }
    i32 next = sample_topk(logits, m->c.vocab, temp, topK>0?topK:40, recent, rn, 1.15f);
    if (next == m->tok->eos) { pthread_mutex_unlock(&g_mu); break; }
    if(rn<REP_WINDOW) recent[rn++]=next; else { memmove(recent,recent+1,(REP_WINDOW-1)*sizeof(i32)); recent[REP_WINDOW-1]=next; }
    char *piece = tok_decode(m->tok, &next, 1);
    pthread_mutex_unlock(&g_mu);
    if (!piece) break;
    size_t plen = strlen(piece);
    if (len + plen + 1 > cap) {
      size_t ncap = (len + plen + 1) * 2;
      char *tmp = realloc(out, ncap);
      if (!tmp) { free(piece); break; }
      out = tmp; cap = ncap;
    }
    memcpy(out + len, piece, plen); len += plen; out[len] = 0;
    jstring jp = (*env)->NewStringUTF(env, piece);
    free(piece);
    if (jp) {
      (*env)->CallVoidMethod(env, sink, on_token, jp);
      (*env)->DeleteLocalRef(env, jp);
      if ((*env)->ExceptionCheck(env)) { free(out); free(logits); free(ids); free(recent); return NULL; }
    }
    pthread_mutex_lock(&g_mu);
    if (!j->live || j->cancel) { pthread_mutex_unlock(&g_mu); break; }
    model_forward(m, next, pos, logits);
    pos++;
    pthread_mutex_unlock(&g_mu);
  }
  result = (*env)->NewStringUTF(env, out);

  free(out); free(logits); free(ids); free(recent);
  return result;
}
