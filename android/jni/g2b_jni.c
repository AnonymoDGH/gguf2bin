#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include "g2b.h"

/*
 * Puente JNI para el núcleo gguf2bin2 en Android.
 * v1: cargar modelo .g2bx + generación con streaming por callback.
 */

#define DEFAULT_CTX 2048

JNIEXPORT jlong JNICALL
Java_com_gguf2bin_app_Native_loadModel(JNIEnv *env, jclass cl, jstring path, jint ctx) {
  const char *p = (*env)->GetStringUTFChars(env, path, NULL);
  if (!p) return 0;
  Model *m = (Model *)calloc(1, sizeof(Model));
  int rc = model_load_g2bx(p, m);
  (*env)->ReleaseStringUTFChars(env, path, p);
  if (rc) { free(m); return 0; }
  model_set_ctx(m, ctx > 0 ? ctx : DEFAULT_CTX);
  /* KV grande -> Q8 automático ya lo decide el core (>1 GB dispara Q8) */
  return (jlong)(intptr_t)m;
}

JNIEXPORT void JNICALL
Java_com_gguf2bin_app_Native_freeModel(JNIEnv *env, jclass cl, jlong ptr) {
  Model *m = (Model *)(intptr_t)ptr;
  if (m) { model_free(m); free(m); }
}

/*
 * Genera texto y llama sink.onToken(String) por cada token emitido.
 * prompt: texto plano; historial lo maneja la capa Java.
 */
JNIEXPORT jstring JNICALL
Java_com_gguf2bin_app_Native_generate(JNIEnv *env, jclass cl, jlong ptr,
                                      jstring prompt, jint maxTokens,
                                      jfloat temp, jobject sink) {
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

  /* BOS si el modelo lo exige y falta */
  if (m->tok->bos >= 0 && ids[0] != m->tok->bos) {
    i32 *tmp = (i32 *)malloc((size_t)(nt + 1) * sizeof(i32));
    if (tmp) { tmp[0] = m->tok->bos; memcpy(tmp + 1, ids, (size_t)nt * sizeof(i32)); free(ids); ids = tmp; nt++; }
  }

  i32 maxctx = m->ctx > 0 ? m->ctx : m->c.seq_len;
  f32 *logits = (f32 *)malloc((size_t)m->c.vocab * sizeof(f32));
  if (!logits) { free(ids); return NULL; }

  size_t cap = 4096, len = 0;
  char *out = (char *)malloc(cap);
  if (!out) { free(logits); free(ids); return NULL; }
  out[0] = 0;

  i32 pos = 0;
  /* prefill del prompt (sin logits salvo el último) */
  for (i32 i = 0; i < nt; i++) {
    if (pos >= maxctx) break;
    int last = (i == nt - 1);
    model_forward_ex(m, ids[i], pos, logits, last);
    pos++;
  }

  jstring result = NULL;
  for (i32 k = 0; k < maxTokens; k++) {
    if (pos >= maxctx) break;
    i32 next = model_sample(logits, m->c.vocab, temp);
    if (next == m->tok->eos) break;
    char *piece = tok_decode(m->tok, &next, 1);
    if (piece) {
      size_t plen = strlen(piece);
      if (len + plen + 1 > cap) { cap = (len + plen + 1) * 2; out = realloc(out, cap); }
      memcpy(out + len, piece, plen); len += plen; out[len] = 0;
      jstring jp = (*env)->NewStringUTF(env, piece);
      if (jp) {
        (*env)->CallVoidMethod(env, sink, on_token, jp);
        (*env)->DeleteLocalRef(env, jp);
      }
      free(piece);
    }
    model_forward(m, next, pos, logits);
    pos++;
  }
  result = (*env)->NewStringUTF(env, out);

  free(out); free(logits); free(ids);
  return result;
}
