/* g2b.h — gguf2bin2: formato propio G2BX + runtime optimizado (Qwen3/Llama) v3.3 */
#ifndef G2B_H
#define G2B_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef uint8_t  u8;  typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef int8_t   i8;  typedef int16_t  i16; typedef int32_t  i32; typedef int64_t  i64;
typedef float    f32;

enum {
  T_F32=0, T_F16=1, T_Q4_0=2, T_Q4_1=3, T_Q5_0=6, T_Q5_1=7,
  T_Q8_0=8, T_Q8_1=9, T_Q2_K=10, T_Q3_K=11, T_Q4_K=12, T_Q5_K=13, T_Q6_K=14, T_Q8_K=15
};
enum { ARCH_LLAMA=0, ARCH_QWEN2=1, ARCH_QWEN3=2 };
enum { F_TIE_EMBD=1u<<0, F_QK_NORM=1u<<1, F_MMAP=1u<<2 };
enum {
  R_TOK_EMBD=0, R_OUT_NORM, R_OUTPUT,
  R_ATTN_NORM, R_ATTN_Q, R_ATTN_K, R_ATTN_V, R_ATTN_O, R_ATTN_Q_NORM, R_ATTN_K_NORM,
  R_FFN_NORM, R_FFN_GATE, R_FFN_UP, R_FFN_DOWN, R_COUNT
};

typedef struct { char *name; u32 n_dims; u64 *dims; u32 type; u64 offset; } GTensor;
typedef struct {
  u8 *data; size_t size; u32 version; u64 n_tensors, alignment, data_off; GTensor *t;
} GGUF;

int gguf_load(const char *path, GGUF *g); void gguf_free(GGUF *g);
u8 *gguf_tensor_ptr(GGUF *g, GTensor *t); GTensor *gguf_by_name(GGUF *g, const char *name);
i64 gguf_meta_i64(GGUF *g, const char *key); f32 gguf_meta_f32(GGUF *g, const char *key);
int gguf_meta_str(GGUF *g, const char *key, char *out, i32 outsz);
int gguf_meta_strarr(GGUF *g, const char *key, char ***out, u64 *n);

u64 ggml_block_size(u32 type); u64 ggml_type_bytes(u32 type); u64 ggml_type_size(u32 type, u64 ne);
f32 half_to_float(u16 h); void gguf_dequant(u32 type, u8 *src, f32 *out, u64 ne);
void matmul_q4_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_q8_0(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d);
void matmul_q(f32 *out, f32 *x, u8 *w, u32 type, i32 n, i32 d, f32 *row);
void matmul(f32 *xout, f32 *x, f32 *w, i32 n, i32 d);
void rmsnorm(f32 *o, f32 *x, f32 *w, i32 n, f32 eps);
void softmax(f32 *x, i32 n); void silu(f32 *x, i32 n);
void rope_th(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta);
void rope_th_llama(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta);
void rope_th_neox(f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta);
void qk_rmsnorm(f32 *x, const f32 *w, i32 n_heads, i32 head_dim, f32 eps);

#pragma pack(push,1)
typedef struct {
  i32 dim, hidden_dim, n_layers, n_heads, n_kv_heads;
  i32 vocab, seq_len, head_dim; f32 eps, rope_theta;
} ModelCfg;
typedef struct { u8 role; u16 layer; u8 type; u32 nbytes; u64 off; } Slot;
#pragma pack(pop)

int g2bx_pack(const char *gguf_path, const char *out_path);
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
  f32 *kcache, *vcache, *buf;
  Tokenizer *tok;
  u8 own_data;
  u8 use_mmap; /* runtime only; not part of on-disk flags */
  /* mmap state (when use_mmap) */
  void *map_view;
  size_t map_size;
#if defined(_WIN32)
  void *file_handle;
  void *map_handle;
#else
  int fd;
#endif
} Model;

int model_load_g2bx(const char *path, Model *m);
int model_load_gguf(const char *path, Model *m);
void model_free(Model *m);
void model_forward(Model *m, i32 token, i32 pos, f32 *logits);
i32 model_sample(f32 *logits, i32 n, f32 temp);
u8 *slot_ptr(Model *m, Slot *s);
Slot *slot_get(Model *m, u8 role, i32 layer);
int exp_synth_qwen_tiny(const char *out_path);

#endif
