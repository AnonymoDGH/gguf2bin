/* gguf2bin.h — API pública estable de gguf2bin2 (Fase 1).
 *
 * Este es el ÚNICO header que un consumidor externo necesita:
 * CLI, bindings (Python/ctypes, Rust/FFI) y GUIs programan contra esto.
 * Los internals (kernels, Model, tokenizer, Vulkan) viven en src/internal/g2b.h
 * y NO son parte del contrato: pueden cambiar sin aviso.
 *
 * Convenciones:
 *  - Toda función que puede fallar devuelve g2b_error (G2B_OK=0).
 *  - Los strings devueltos por g2b_decode deben liberarse con free().
 *  - Los ids devueltos por g2b_encode deben liberarse con free().
 *  - Una sesión = un modelo + un estado de conversación. Hoy el cómputo global
 *    subyacente admite una sesión activa a la vez (Fase 2 eliminará el resto
 *    de estado global); la API ya aísla RNG/sampler/conversación por sesión.
 */
#ifndef GGUF2BIN_H
#define GGUF2BIN_H

#include <stdint.h>
#include "version.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct g2b_session g2b_session;

/* ── errores ─────────────────────────────────────────────────────────── */
typedef enum {
  G2B_OK = 0,
  G2B_ERR_IO,          /* archivo no existe / no legible / no escribible */
  G2B_ERR_FORMAT,      /* magic / versión / estructura inválida */
  G2B_ERR_GEOMETRY,    /* cfg inconsistente (GQA, dims, vocab) */
  G2B_ERR_OOM,
  G2B_ERR_CONTEXT,     /* token fuera de vocab, pos fuera de ctx */
  G2B_ERR_UNSUPPORTED, /* tipo de tensor / arquitectura no soportada */
  G2B_ERR_CANCELLED,   /* callback is_cancelled lo pidió */
} g2b_error;

const char *g2b_strerror(g2b_error e);
const char *g2b_version(void);              /* "4.9.0" (de version.h) */

/* ── apertura / introspección ─────────────────────────────────────────── */
typedef struct {
  int      ctx;             /* 0 = el del modelo */
  int      threads;         /* 0 = auto; >0 fija hilos OpenMP */
  int      q8_kv;           /* -1 auto, 0 fuerza F32, 1 fuerza Q8 */
  uint64_t max_ram_bytes;   /* 0 = sin límite (auto-Q8 + shrink ctx hasta encajar) */
  const char *swap_path;    /* NULL = KV en RAM; "@" = ruta por defecto del sistema */
  int      fast;            /* 1 = prioridad alta + todos los hilos, sin swap */
  float    mv_ratio;        /* 0 = off; 0.0..1.0 skip especulativo FFN/SSM */
  float    bvh_keep;        /* 0 = off; >0 sparse-attention BVH keep-ratio */
  int      gpu;             /* 1 = dual-band CPU+GPU del head (si Vulkan ayuda) */
  const char *lora_path;    /* NULL = sin adaptador */
  uint64_t seed;            /* 0 = no determinista */
} g2b_config;

typedef struct {
  char     arch[16];        /* "qwen3", "llama", "lfm2", "qwen35", ... */
  int      dim, n_layers, n_heads, n_kv_heads, head_dim, vocab, ctx_max;
  int      ctx_eff;         /* contexto efectivo (sesión) o == ctx_max (g2b_info) */
  int      bos_id, eos_id;  /* -1 si el modelo no los define */
  uint64_t weight_bytes;    /* blob de pesos (mmap: no cuenta como RAM residente) */
  uint64_t est_ram_bytes;   /* runtime estimado a ctx_max (KV+buffers+tokenizer) */
  int      has_tokenizer;
} g2b_model_info;

/* Metadata SIN cargar pesos (escaneo de header; ideal para listar modelos). */
g2b_error g2b_info(const char *path, g2b_model_info *out);
g2b_error g2b_open(const char *path, const g2b_config *cfg, g2b_session **out);
void      g2b_close(g2b_session *s);
g2b_error g2b_set_ctx(g2b_session *s, int ctx);
/* Misma línea que el reporte de la CLI, en buf (siempre NUL-terminada). */
void      g2b_ram_report(const g2b_session *s, char *buf, int buflen);
int       g2b_ctx_used(const g2b_session *s);   /* tokens consumidos (pos) */
g2b_error g2b_model_info_of(const g2b_session *s, g2b_model_info *out);

/* ── generación ───────────────────────────────────────────────────────── */
typedef struct {
  /* callbacks opcionales (pueden ser NULL salvo on_token si se quiere salida) */
  void (*on_token)(const char *utf8_piece, void *ud);
  void (*on_progress)(int tokens_done, int tokens_max, float tok_per_s, void *ud);
  int  (*is_cancelled)(void *ud);
  void *ud;
  /* sampling */
  float temp;        /* 0 = greedy */
  int   top_k;       /* <=0 = sin límite */
  float top_p;       /* >=1 = sin nucleus */
  float repeat_penalty;
  int   repeat_window;   /* 0 = 64 */
  uint64_t seed;         /* !=0 re-siembra el RNG de la sesión para esta llamada */
  int   max_tokens;
  int   show_think;      /* solo chat: 1 = muestra bloques <think> */
} g2b_gen_params;

/* Nivel bajo: prefill de toks[0..n) + generación hasta max_tokens. */
g2b_error g2b_generate(g2b_session *s, const int32_t *toks, int n_toks,
                       const g2b_gen_params *p);
/* Un turno de chat: formatea (plantilla interna), prefill SOLO del mensaje
 * nuevo (la KV persiste) y genera con criterios de parada del turno.
 * Llamar g2b_chat_begin una vez antes del primer turno. */
g2b_error g2b_chat_begin(g2b_session *s, const char *system_utf8, int no_think);
g2b_error g2b_chat_turn(g2b_session *s, const char *user_utf8,
                        const g2b_gen_params *p);
g2b_error g2b_chat_reset(g2b_session *s);   /* conserva pesos+system, borra KV */

/* ── tokenizer (útil para contar tokens / eco de prompt) ──────────────── */
int   g2b_encode(g2b_session *s, const char *text, int32_t **out_ids); /* n, -1 sin tok */
char *g2b_decode(g2b_session *s, const int32_t *ids, int n);           /* free(), NULL sin tok */

/* ── poda ShortGPT (calibración con corpus embebido) ──────────────────── */
g2b_error g2b_autodrop(g2b_session *s, int ndrop);

/* ── perplexity (harness de calidad) ────────────────────────────────── */
typedef struct {
  int tokens, evaluated;
  double nll_per_token, perplexity;
} g2b_ppl_result;
g2b_error g2b_ppl(g2b_session *s, const char *text, int max_tokens,
                  g2b_ppl_result *out);

/* ── empaquetado ──────────────────────────────────────────────────────── */
typedef struct {
  int downquant;        /* 0 = preservar tipos, 1 = bajar a Q4_0 */
  int out_quant;        /* 0 native|Q4_0, 1 = Q4_0S, 2 = PSY, 3 = VVC */
  float prune;          /* 0..0.9 fracción FFN a podar (0 = sin poda) */
  const char *calib_path;  /* NULL = proxy |g·u| (rápido) o corpus embebido */
  void (*on_progress)(float pct, const char *phase, void *ud); /* reservado Fase 2 */
  void *ud;
} g2b_pack_opts;
g2b_error g2b_pack(const char *gguf, const char *out, const g2b_pack_opts *o);

/* ── utilidades ───────────────────────────────────────────────────────── */
g2b_error g2b_vk_probe(char *report, int buflen);  /* sonda Vulkan, texto humano */
g2b_error g2b_bench(g2b_session *s, int n_tokens, int prefill_tokens,
                    float *decode_tps, float *prefill_tps); /* min-of-3 */

#ifdef __cplusplus
}
#endif
#endif
