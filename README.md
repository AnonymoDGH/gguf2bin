# gguf2bin2 — G2BX FIXED v3.3

Runtime propio en C99: **GGUF → G2BX → inferencia** (Qwen3 / Qwen2 / Llama).

**Validado:** `The capital of France is` → `Paris. The capital of Italy is Rome...`

Probado contra llama.cpp b10064 con `Qwen3-0.6B-Q8_0.gguf`.

## Cambios v3.3

- Sampler: repeat-penalty **una sola vez** + ring buffer correcto + dedup de ids
- RoPE por arquitectura: **NEOX** (Qwen2/3), **interleaved** (Llama)
- Contexto: no hace wrap corrupto; corta con mensaje al llenar `seq_len`
- Loader: **mmap** de pesos cuando el SO lo permite (`flags & F_MMAP`)
- Null-checks en slots del forward; matmul genérico sin data-race OpenMP
- Tokenizer hash con **rehash** al 70% de carga
- Pack avisa y omite tipos no soportados (Q1/Q2_K/…)
- Makefile + `make test` portable en Windows/Unix

## Root cause histórico — RoPE NEOX vs LLaMA

Qwen3 usa **NEOX RoPE** (half-split). Con RoPE LLaMA interleaved sale basura UTF-8.

```c
// LLaMA: pares (0,1),(2,3)...
// NEOX:  pares (0,D/2),(1,D/2+1)...
```

## Compilar

```bash
# Linux / MinGW con OpenMP
make

# Smoke test
make test

# Sin OpenMP (fallback)
gcc -O2 -std=c99 -Iinclude -o gguf2bin2.exe \
  src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c \
  src/l5_model.c src/l6_token.c src/main.c -lm
```

## Uso

```bash
# Empaquetar
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen.g2bx

# Info
./gguf2bin2 info qwen.g2bx

# Completar (greedy)
./gguf2bin2 run qwen.g2bx --tokens 785,6722,315,9625,374 -n 15 -t 0

# Chat no-think
./gguf2bin2 chat qwen.g2bx --no-think -n 40 -t 0.7
```

## Formato G2BX

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B aligned | tokenizer
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

## Tipos soportados

| Tipo | Load | Matmul fused |
|------|------|--------------|
| F32 / F16 | sí (pack→Q4_0 si es peso) | F32 path / dequant |
| Q4_0 / Q4_1 | sí | Q4_0 AVX2/scalar |
| Q8_0 | sí | Q8_0 AVX2/scalar |
| Q1 / Q2_K / Q3_K / Q4_K / … | **no** | — |

## Estructura

```
include/g2b.h
src/l1_gguf.c    # parser GGUF
src/l2_codec.c   # dequant + matmul
src/l3_math.c    # rmsnorm, rope, silu
src/l4_gbin.c    # pack G2BX
src/l5_model.c   # load + forward
src/l6_token.c   # BPE tokenizer
src/main.c       # CLI
docs/RESEARCH.md
```
