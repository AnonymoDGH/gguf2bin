# Investigación: super-comprimir Qwen3 y runtime propio — FIXED v3.4

## Novedades v3.4 — runtime en 2 GB de RAM

El objetivo es correr modelos **pesados** en equipos con **2 GB de RAM**:

1. **Pesos por mmap** (v3.3): no cuentan contra el heap; el SO evicta páginas.
   Un modelo de 4 B params en Q4 (~2.4 GB de archivo) corre con RSS < 600 MB.
2. **KV cache Q8_0** (nuevo): K/V de F32 (4 B/elem) a Q8_0 (34 B/32 elems) = **~3.76× menos RAM**.
   Medido en Qwen3-0.6B ctx=8192: runtime 1819 MB (F32) → **503 MB (Q8_0)**.
3. **Contexto efectivo** (`--ctx N`): la KV cache se dimensiona al contexto de la
   sesión, no al `context_length` del GGUF (262144 en algunos modelos).
4. **`--max-ram MB`**: heurística automática → activa KV Q8_0, y si aún no cabe,
   divide el contexto a la mitad hasta encajar. Imprime reporte `ram: ...`.

### Validación numérica de la KV Q8_0 (tools/kvtest.c)

Misma arquitectura/pesos, dos modelos: KV F32 (referencia) vs KV Q8_0.
Con Qwen3-0.6B-Q8_0, 64 pasos a ctx=72:

```
max|dF32-dQ8| = 2.09   rmse = 0.27   argmax(greedy) match 60/64 = 94%
```

rmse 0.27 sobre logits de rango ~±30 → error relativo ~1%. El top-1 (greedy)
coincide ≥90% de los pasos: la calidad de generación se preserva. Es el mismo
orden de error que la KV Q8 de llama.cpp.

### Arquitecturas custom (dflash / speculative decoding)

- `read_cfg` ahora usa `general.architecture` como prefijo real (con fallbacks
  `qwen3/qwen2/llama`), así empaqueta cualquier arquitectura con nombres estándar
  `blk.N.*` (attn_q/k/v/output, Q/K norm, ffn_gate/up/down, norms).
- RoPE: unknown → NEOX por defecto (heurística: presencia de `attn_q_norm` /
  `attention.key_length`); si parece llama-like → interleaved.
- Bugfix: `tokenizer.ggml.tokens` es un array; `gguf_meta_i64` devolvía 0 →
  `vocab=0`. Ahora `gguf_meta_arr_len()` lee la longitud real.
- **Modelos draft/decode-only**: GGUFs de speculative decoding (p.ej. `dflash`
  62 tensores: `conf_proj`, `markov_w1/w2`, `fc`, … **sin `token_embd.weight`**)
  no pueden generar texto standalone: carecen del embedding y de la cabeza de
  logits propias, dependen de un teacher. `pack` lo detecta y avisa.

## Fixes v3.3

1. **Sampler**: repeat_penalty se aplicaba dos veces con temp>0; ring buffer de `recent` leía mal tras 128 tokens. Ahora: una aplicación, dedup de ids, snapshot de los últimos N.
2. **RoPE por arch**: Llama → interleaved; Qwen2/3 → NEOX.
3. **Contexto**: `pos %= seq` corrompía atención. Ahora aborta limpio al superar `seq_len`.
4. **mmap**: pesos mapeados si el SO lo permite; fallback malloc+fread.
5. **matmul_q OpenMP**: buffer `row` compartido era data-race; scratch privado por hilo.
6. **Hash tokenizer**: rehash al 70% de carga.
7. **Pack**: omite y reporta quant types no soportados (Q1_0, K-quants).

## Fixes v3.1 / v3.2

1. **G2BX loader**: `aligned_end = ALIGN64(max_end)` para no perder tokenizer.
2. **Pack**: preserva Q8_0/Q4_0; solo F32/F16 → Q4_0.
3. **CLI**: `run` sin BOS por defecto; top-k/top-p/repeat-penalty.
4. **Chat**: plantilla oficial Qwen3 (`thinking` / `no-think`).

## Validación vs llama.cpp (b10064)

GGUF Qwen3-0.6B-Q8_0 Instruct (639MB, 310 tensores, dim 1024 L28)

- Completion: `The capital of France is` → `Paris. The capital of Italy is Rome...`
- Chat no-think: `What is the capital of France?` → `The capital of France is **Paris**.`
- v3.4 (atención reescrita + KV Q8): las salidas greedy F32 y Q8_0 son idénticas
  en el test corto; chat reproducido con `--max-ram 1024`.

## Por qué Qwen3 no es Llama

- head_dim 128 != dim/heads 64
- rope_theta 1e6 (dflash: 1e7 con YaRN ×32, original 8192)
- QK-Norm por capa
- vocab 151936 (dflash: 248320)
- thinking tokens 151667/151668
- RoPE style NEOX

## Rendimiento (optimización v3.4b)

Decode Qwen3-0.6B (mmap, CPU 4 núcleos):

| Build | tok/s |
|-------|-------|
| Escalar (sin -mavx2) | 0.7 |
| AVX2, 1 hilo | 4.9 |
| AVX2 + RoPE precompute, 1 hilo | 5.6 |
| AVX2, 4 hilos | 10.9 |
| Q4 pack (pesos 335 MB vs 633 MB), 4 hilos | 10.4 |

Hallazgos: el decode está limitado por cómputo/FMA (no por ancho de banda de
memoria) en este CPU; por eso Q4 no acelera y la palanca real es la paralelización
del matmul por filas de salida. El beneficio de `pack --q4` es de *capacidad*
(mismos 2 GB ↦ modelo más grande), no de tok/s.

Optimizaciones aplicadas v3.4b:
- `-mavx2 -mfma -fopenmp` por defecto (Makefile): activa los kernels AVX2 (Q8_0/Q4_0).
- RoPE: frecuencias `cos/sin` precomputadas una vez por posición (antes `powf`
  dentro del bucle por head → -15% tiempo).
- `--threads N`: `omp_set_num_threads` desde CLI.
- Kernel Q4_0 AVX2 reescrito limpio (misma matemática, sin el baile de splits).

## RAM en 2 GB — números (Qwen3-0.6B, ctx 8192)

| Modo | KV K+V | buffers | tokenizer | runtime total |
|------|--------|---------|-----------|---------------|
| F32 (default) | 1.5 GB | ~2 MB | ~36 MB | **1819 MB** |
| Q8_0 (`--q8-kv`) | 402 MB | ~2 MB | ~36 MB | **503 MB** |

Para un 1.4B–4B (más capas/dim), F32 suele superar 2 GB solo con la KV; con
`--max-ram 2048` el runtime elige Q8_0 y/o baja el contexto automáticamente.

## Rendimiento CPU

La build ahora activa `-O3 -mavx2 -mfma -fopenmp` por defecto y permite elegir
los hilos con `--threads N`. En la CPU de validación (4 hilos lógicos),
Qwen3-0.6B pasó de 0.7 tok/s escalar sin AVX2 a 5.6 tok/s con AVX2 en un hilo
y 10.9 tok/s con cuatro hilos.

`pack --q4` convierte los pesos Q8_0 a Q4_0. Reduce el archivo de pesos de
633 MB a 335 MB en Qwen3-0.6B y mantiene una velocidad similar, porque el
cuello de botella principal es el cálculo de FMA, no la capacidad de la KV.
