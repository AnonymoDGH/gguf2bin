#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "g2b.h"

/*
 * Puente JNI para el núcleo gguf2bin2 en Android.
 * v2: carga .g2bx, generación con streaming, muestreo top-k + penalización
 *     de repetición (igual que la CLI) y ajuste de hilos OpenMP.
 */

#define DEFAULT_CTX 2048

/* ── muestreo con top-k + repeat penalty (adaptado de main.c) ── */
static void apply_rep_penalty(f32 *logits, const i32 *recent, int rn, f32 pen){
  if(pen <= 1.f) return;
  for(int i=0;i<rn;i++){
    i32 t=recent[i];
    if(t>=0 && logits[t]>0) logits[t]/=pen;
    else if(t>=0) logits[t]*=pen;
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
  apply_rep_penalty(logits,recent,rn,rep_pen);
  if(top_k<1||top_k>n) top_k=n;
  /* copia para hallar umbral del k-ésimo mayor */
  f32 *tmp=(f32*)malloc((size_t)n*sizeof(f32));
  if(!tmp){ i32 bi=0; f32 bv=logits[0]; for(i32 i=1;i<n;i++) if(logits[i]>bv){bv=logits[i];bi=i;} return bi; }
  memcpy(tmp,logits,(size_t)n*sizeof(f32));
  quickselect_k(tmp,n,top_k-1);
  f32 thresh=tmp[top_k-1];
  free(tmp);
  /* softmax sólo sobre >= umbral */
  double mx=-1e30; i32 cnt=0;
  for(i32 i=0;i<n;i++) if(logits[i]>=thresh){cnt++; if(logits[i]>mx)mx=logits[i];}
  if(!cnt) cnt=1;
  double sum=0;
  for(i32 i=0;i<n;i++) if(logits[i]>=thresh) sum+=exp((double)(logits[i]/temp)-mx/temp);
  double r=((double)rand()/(double)RAND_MAX)*sum, c=0;
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
  Model *m = (Model *)calloc(1, sizeof(Model));
  int rc = model_load_g2bx(p, m);
  (*env)->ReleaseStringUTFChars(env, path, p);
  if (rc) { free(m); return 0; }
  model_set_ctx(m, ctx > 0 ? ctx : DEFAULT_CTX);
  return (jlong)(intptr_t)m;
}

JNIEXPORT void JNICALL
Java_com_gguf2bin_app_Native_freeModel(JNIEnv *env, jclass cl, jlong ptr) {
  Model *m = (Model *)(intptr_t)ptr;
  if (m) { model_free(m); free(m); }
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
  Model *m = (Model *)(intptr_t)ptr;
  if (!m || !m->tok) return NULL;

  static jclass sink_cls = NULL;
  static jmethodID on_token = NULL;
  if (!sink_cls) {
    jclass c = (*env)->FindClass(env, "com/gguf2bin/app/Native$TokenSink");
    if (!c) return NULL;
    sink_cls = (*env)->NewGlobalRef(env, c);
    on_token = (*env)->GetMethodID(env, sink_cls, "onToken", "(Ljava/lang/String;)V");
    if (!on_token) return NULL;
  }

  const char *ptext = (*env)->GetStringUTFChars(env, prompt, NULL);
  i32 *ids = NULL;
  i32 nt = tok_encode(m->tok, (char *)ptext, &ids);
  (*env)->ReleaseStringUTFChars(env, prompt, ptext);
  if (nt <= 0) { free(ids); return NULL; }

  if (m->tok->bos >= 0 && ids[0] != m->tok->bos) {
    i32 *tmp = (i32 *)malloc((size_t)(nt + 1) * sizeof(i32));
    if (tmp) { tmp[0] = m->tok->bos; memcpy(tmp + 1, ids, (size_t)nt * sizeof(i32)); free(ids); ids = tmp; nt++; }
  }

  i32 maxctx = m->ctx > 0 ? m->ctx : m->c.seq_len;
  f32 *logits = (f32 *)malloc((size_t)m->c.vocab * sizeof(f32));
  i32 *recent = (i32 *)malloc((size_t)REP_WINDOW * sizeof(i32));
  int rn = 0;
  char *out = NULL; size_t cap = 8192, len = 0;
  if (!logits || !recent || !(out=(char*)malloc(cap))) { free(logits); free(recent); free(ids); return NULL; }
  out[0]=0;

  i32 pos = 0;
  for (i32 i = 0; i < nt; i++) {
    if (pos >= maxctx) break;
    model_forward_ex(m, ids[i], pos, logits, i == nt - 1);
    if(rn<REP_WINDOW) recent[rn++]=ids[i]; else { memmove(recent,recent+1,(REP_WINDOW-1)*sizeof(i32)); recent[REP_WINDOW-1]=ids[i]; }
    pos++;
  }

  jstring result = NULL;
  for (i32 k = 0; k < maxTokens; k++) {
    if (pos >= maxctx) break;
    i32 next = sample_topk(logits, m->c.vocab, temp, topK>0?topK:40, recent, rn, 1.15f);
    if (next == m->tok->eos) break;
    if(rn<REP_WINDOW) recent[rn++]=next; else { memmove(recent,recent+1,(REP_WINDOW-1)*sizeof(i32)); recent[REP_WINDOW-1]=next; }
    char *piece = tok_decode(m->tok, &next, 1);
    if (piece) {
      size_t plen = strlen(piece);
      if (len + plen + 1 > cap) { cap = (len + plen + 1) * 2; out = realloc(out, cap); }
      memcpy(out + len, piece, plen); len += plen; out[len] = 0;
      jstring jp = (*env)->NewStringUTF(env, piece);
      if (jp) { (*env)->CallVoidMethod(env, sink, on_token, jp); (*env)->DeleteLocalRef(env, jp); }
      free(piece);
    } else break;
    model_forward(m, next, pos, logits);
    pos++;
  }
  result = (*env)->NewStringUTF(env, out);

  free(out); free(logits); free(ids); free(recent);
  return result;
}
