/* g2b.h — gguf2bin2: formato propio G2BX + runtime optimizado (Qwen3/Llama) v4.0 */
#ifndef G2B_H
#define G2B_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef uint8_t  u8;  typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t   i8;  typedef int16_t  i16; typedef int32_t  i32; typedef int64_t  i64;
typedef float    f32;

#define G2BX_MAGIC "G2BX"
#define G2BX_VER   2u
#define G2BX_VER_MAX 2u
#define G2BX_CFG_V1 40 /* bytes de ModelCfg en v1 */
#define ALIGN64(x) (((x)+63ull)&~63ull)

enum {
  T_F32=0, T_F16=1, T_Q4_0=2, T_Q4_1=3, T_Q5_0=6, T_Q5_1=7,
  T_Q8_0=8, T_Q8_1=9, T_Q2_K=10, T_Q3_K=11, T_Q4_K=12, T_Q5_K=13, T_Q6_K=14, T_Q8_K=15,
  /* IDs GGUF reales (ggml.h): los IQ estaban desalineados */
  T_IQ2_XXS=16, T_IQ2_XS=17, T_IQ3_XXS=18, T_IQ1_S=19, T_IQ4_NL=20,
  T_IQ3_S=21, T_IQ2_S=22, T_IQ4_XS=23, T_F64=28,
  T_IQ1_M=29, T_BF16=30,
  T_Q4_0S=25 /* interno: reutiliza un ID que GGUF no usa en pesos (compat con g2bx antiguos) */
};
enum { ARCH_LLAMA=0, ARCH_QWEN2=1, ARCH_QWEN3=2, ARCH_LFM2=3, ARCH_QWEN35=4 };
enum { F_TIE_EMBD=1u<<0, F_QK_NORM=1u<<1, F_MMAP=1u<<2, F_KV_Q8=1u<<3 /* runtime KV cache cuantizado Q8_0 (no on-disk) */ };
enum {
  R_TOK_EMBD=0, R_OUT_NORM, R_OUTPUT,
  R_ATTN_NORM, R_ATTN_Q, R_ATTN_K, R_ATTN_V, R_ATTN_O, R_ATTN_Q_NORM, R_ATTN_K_NORM,
  R_FFN_NORM, R_FFN_GATE, R_FFN_UP, R_FFN_DOWN,
  R_ATTN_Q_BIAS, R_ATTN_K_BIAS, R_ATTN_V_BIAS, /* Qwen2.5 attention biases (añadidos al final: no rompe roles existentes) */
  R_EMBD_NORM,   /* LFM2: norm tras el embedding */
  R_CONV_W,      /* LFM2 shortconv: peso depthwise [kernel,channels] F32 */
  R_CONV_IN,     /* LFM2 shortconv: proyección entrada (3*dim) */
  R_CONV_OUT,    /* LFM2 shortconv: proyección salida */
  /* qwen35 híbrido (GDN + atención completa cada N capas) */
  R_ATTN_QKV,    /* qkv fusionado de la capa lineal (gated delta net) */
  R_ATTN_GATE,   /* z gate de la capa lineal / gate sigmoide de la atención completa */
  R_SSM_A,       /* -exp(A_log), vector [dt_rank] */
  R_SSM_DT,      /* dt bias [dt_rank] */
  R_SSM_CONV,    /* conv1d depthwise [d_conv, conv_dim] F32 */
  R_SSM_ALPHA,   /* proj alpha [dim, dt_rank] */
  R_SSM_BETA,    /* proj beta [dim, dt_rank] */
  R_SSM_NORM,    /* rmsnorm gated [head_v_dim] */
  R_SSM_OUT,     /* out_proj [value_dim, dim] */
  R_COUNT
};

typedef struct { char *name; u32 n_dims; u64 *dims; u32 type; u64 offset; } GTensor;
typedef struct {
  u8 *data; size_t size; u32 version; u64 n_tensors, alignment, data_off; GTensor *t;
  u8 own_data; /* 1=malloc (free), 2=mmap (unmap) */
  void *map_view; size_t map_size;
#if defined(_WIN32)
  void *file_handle; void *map_handle;
#else
  int fd;
#endif
} GGUF;

int gguf_load(const char *path, GGUF *g); void gguf_free(GGUF *g);
u8 *gguf_tensor_ptr(GGUF *g, GTensor *t); GTensor *gguf_by_name(GGUF *g, const char *name);
i64 gguf_meta_i64(GGUF *g, const char *key); f32 gguf_meta_f32(GGUF *g, const char *key);
int gguf_meta_str(GGUF *g, const char *key, char *out, i32 outsz);
int gguf_meta_strarr(GGUF *g, const char *key, char ***out, u64 *n);
u64 gguf_meta_arr_len(GGUF *g, const char *key);

u64 ggml_block_size(u32 type); u64 ggml_type_bytes(u32 type); u64 ggml_type_size(u32 type, u64 ne);
f32 half_to_float(u16 h); void gguf_dequant(u32 type, u8 *src, f32 *out, u64 ne);
void matmul_q4_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_q8_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_q(f32 *out, f32 *x, u8 *w, u32 type, i32 n, i32 d, f32 *row);
void matmul_q_rows(f32 *out, const f32 *x, const u8 *w, u32 type, i32 n, i32 r0, i32 r1);
void matmul_q_b(f32 *out, const f32 *x, u8 *w, u32 type, i32 n, i32 d, i32 B); /* out[B*d] */
void matmul(f32 *xout, f32 *x, f32 *w, i32 n, i32 d);
u64  row_stride(u32 type, i32 n);
void q8_dequant_row_avx2(const u8 *src, f32 *out, i32 n);
void rmsnorm(f32 *o, f32 *x, f32 *w, i32 n, f32 eps);
void softmax(f32 *x, i32 n); void silu(f32 *x, i32 n);
void silu_mul(f32 *gate, const f32 *up, i32 n); /* gate = silu(gate)*up fusionado */
void rope_th(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta);
void rope_th_llama(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta);
void rope_th_neox(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta);
void qk_rmsnorm(f32 *x, const f32 *w, i32 n_heads, i32 head_dim, f32 eps);
u16 f32_to_half(f32 f);

#pragma pack(push,1)
typedef struct {
  i32 dim, hidden_dim, n_layers, n_heads, n_kv_heads;
  i32 vocab, seq_len, head_dim; f32 eps, rope_theta;
  /* v2 — híbridos SSM (qwen35); en v1 estos bytes no existen (leídos como 0) */
  i32 fa_interval;                 /* atención completa cada N capas; 0 = denso */
  i32 ssm_d_state, ssm_n_group, ssm_dt_rank, ssm_inner, ssm_d_conv;
  i32 n_rot;                       /* dims rotadas por head en atención (0=hd completo) */
} ModelCfg;
typedef struct { u8 role; u16 layer; u8 type; u32 nbytes; u64 off; } Slot;
#pragma pack(pop)

int g2bx_pack(const char *gguf_path, const char *out_path);
int g2bx_pack_ex(const char *gguf_path, const char *out_path, int downq4);
/* Poda estructurada FFN: elimina la fracción `prune` de bloques de neuronas.
   Con calib (de model_collect_stats) usa importancia real; sin ella, proxy |g·u|. */
int g2bx_pack_prune(const char *gguf_path, const char *out_path, int downq4, float prune);
int g2bx_pack_prune_scores(const char *gguf_path, const char *out_path,
                           int downq4, float prune, const f32 *calib);
void g2bx_set_q4s(int v);
int vk_init(void); void vkinfo_cmd(void); int vk_device_ok(void);
int vk_head_upload(const u8 *weights, i32 n, i32 rows); int vk_head_pipeline(void); void vk_head_run(const f32 *x, f32 *logits); int vk_head_ready(void);
int g2bx_info(const char *path);

typedef struct SHash SHash;
typedef struct {
  char **tok; i32 n; i32 bos, eos, unk;
  SHash *vocab; SHash *merges; char **mergestr; i32 nmerges;
} Tokenizer;

int tok_from_gguf(GGUF *g, Tokenizer *t);
int tok_read_section(FILE *f, Tokenizer *t);
int tok_write_section(FILE *f, Tokenizer *t);
void tok_free(Tokenizer *t);
i32 tok_encode(Tokenizer *t, const char *text, i32 **out_ids);
char *tok_decode(Tokenizer *t, const i32 *ids, i32 n);
i32 tok_id(Tokenizer *t, const char *s);

typedef struct {
  ModelCfg c; u8 arch, flags;
  Slot *slots; u32 n_slots;
  u8 *data; size_t data_size;
  Slot **ix_global; Slot ***ix_layer;
  f32 *kcache, *vcache, *buf; /* F32 KV + runtime scratch */
  u8  *kcq, *vcq;             /* KV cuantizado Q8_0 (cuando F_KV_Q8) */
  i32 ctx;                    /* contexto efectivo en runtime (<= c.seq_len) */
  u8 no_kv_q8;                /* geometría incompatible con KV Q8 (head_dim%32) */
  f32 *conv_state;            /* LFM2/qwen35: estado conv por capa */
  f32 *ssm_st;                /* qwen35: estado recurrente GDN [n_recr][nv][dv][dv] */
  /* ShortGPT: salto de bloques redundantes */
  u8 *skip_layer;             /* [n_layers] 1=omitir bloque completo */
  f32 *bi_pre, *bi_post;      /* scratch dim para medir Block Influence */
  f32 *bi_dot, *bi_n2, *bi_n2p; /* acumuladores [n_layers]: dot(pre,post), |post|², |pre|² */
  u8 collect_bi;
  /* recolección de estadísticas FFN para poda calibrada */
  u8 collect_stats;
  f32 *ffn_stats;             /* [n_layers x hidden]: Σ|silu(g)·u| por neurona */
  /* buffers del prefill batcheado ([B][...]) */
  i32 pf_B;
  f32 *pf_pool; /* base única del pool (lo único que se libera) */
  f32 *pf_x, *pf_xb, *pf_hb, *pf_hb2, *pf_q, *pf_k, *pf_v, *pf_att;
  Tokenizer *tok;
  u8 own_data;
  u8 use_mmap; /* runtime only; not part of on-disk flags */
  /* KV respaldada en archivo (swap) — p.ej. usar D: como RAM */
  u8 use_swap;
  void *swap_view; size_t swap_size;
  char *swap_path;
  char *src_path; /* G2BX realmente mapeado (puede ser cache .gguf.g2bx) */
  /* mmap state (when use_mmap) */
  void *map_view;
  size_t map_size;
#if defined(_WIN32)
  void *file_handle;
  void *map_handle;
  void *swap_f;
  void *swap_m;
#else
  int fd;
  int swap_fd;
#endif
} Model;

int model_load_g2bx(const char *path, Model *m);
/* ── Dual band CPU+GPU: worker Vulkan en proceso hijo (a prueba de drivers rotos) ── */
int  vk_worker_main(int argc, char **argv);          /* modo --gpu-worker (proceso hijo) */
int  vk_dual_start(Model *m, const char *model_path); /* arranca worker (--gpu) */
void vk_dual_stop(void);
int  vk_dual_active(void);
/* Devuelve 1 si calculó los logits completos (GPU rango + CPU resto), 0 si no hay GPU. */
int  vk_head_dual(f32 *logits, const f32 *x, const u8 *w, u32 type, i32 n, i32 vocab);
int model_load_gguf(const char *path, Model *m);
void model_free(Model *m);
int model_set_ctx(Model *m, i32 ctx);
u64 model_kv_bytes(Model *m, int q8);
u64 model_est_ram(Model *m);
void model_ram_report(Model *m);
int model_auto_budget(Model *m, u64 max_ram);
int model_enable_swap(Model *m, const char *path);
void model_forward(Model *m, i32 token, i32 pos, f32 *logits);
void model_forward_ex(Model *m, i32 token, i32 pos, f32 *logits, int want_logits);
/* Prefill batcheado: procesa tos[0..n) desde posición pos0; logits solo del último. */
int model_prefill(Model *m, const i32 *toks, i32 n, i32 pos0, f32 *last_logits);
/* Calibración para poda: acumula Σ|silu(g)·u| por neurona sobre tokens. */
int model_collect_stats(Model *m, const i32 *toks, i32 n);
void model_free_stats(Model *m);
/* ShortGPT: mide Block Influence por capa y marca las `ndrop` menos influyentes. */
int model_autodrop(Model *m, const i32 *toks, i32 n, int ndrop);
i32 model_sample(f32 *logits, i32 n, f32 temp);
u8 *slot_ptr(Model *m, Slot *s);
Slot *slot_get(Model *m, u8 role, i32 layer);
int exp_synth_qwen_tiny(const char *out_path);

#endif
