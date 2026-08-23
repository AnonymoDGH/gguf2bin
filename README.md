# gguf2bin2 — C99 LLM runtime for low-RAM machines

A self-contained C99 runtime: **GGUF → G2BX (own format) → inference** for
Qwen3 / Qwen2 / Llama / custom architectures. Weights are memory-mapped so the
real RAM budget is **KV cache + activations + tokenizer**, not the model file.

## Speed & RAM (measured on Intel i5-6200U · 2C/4T · DDR3L dual-channel)

Sustained decode with `--fast` (high priority, OpenMP threads, quantized KV).
The DDR3L memory bus (~9.4 GB/s) is the hard ceiling; Q4_0 sits at ~50 % of it.

| Model (G2BX) | Format | Weights (mmap) | Runtime RAM | tok/s v3.5 | tok/s v4.3 |
|--------------|--------|----------------|-------------|------------|------------|
| **Qwen2.5-3B** | Q4_0 | 1992 MB | 145 MB (`--fast --q8-kv`) | 1.5 | **4.3** |
| Qwen3-0.6B | Q4_0 | 319 MB | 145 MB (`--q8-kv -c 2048`) | 18.8 | **16.1** |
| SmolLM2-135M | Q4_0 | 72 MB | 40 MB (`--q8-kv -c 2048`) | 53.6 | **59.5** |

Qwen2.5-3B decode at 4.3 tok/s streams 8.6 GB/s — essentially the DDR3L bus
ceiling; the v3.5 kernel managed only 3 GB/s. The deferred-accumulation Q4_0
kernel (one horizontal sum per row instead of one per 32-elem block) closed the
gap. Batched prefill runs at the kernel compute ceiling (~27 GMAC/s).

Prompt processing (prefill, ~771 tokens, Qwen3-0.6B q4, ctx 2048, misma sesión):
**45.7 s → 26.0 s (1.76×)** — el matmul final vocab×dim se salta durante el prompt
y los tokens se procesan en chunks de 8 por pasada de pesos (prefill batcheado,
verificado bit-exacto contra el forward secuencial con `tools/prefilltest`).
Generation with sampling (-n 192, t 0.7): **23.4 s → 13.4 s (1.75×)** —
quickselect top-k + Gumbel-max replace a full 152k-element `qsort` per token.

**RAM minimum / maximum (footprint the OS can't page out):**
- **Minimum ≈ 37 MB** runtime with `--swap` (KV backed on disk, weights mmap;
  heap drops to tens of MB). A 2 GB machine runs even large models.
- **Maximum ≈ 1.9 GB** runtime if KV is forced to F32 at `ctx=8192` (you would
  normally enable `--q8-kv`, which cuts it to ~500 MB).

> Model filename note: the HuggingFace "Qwen3-0.6B" blob we used is actually a
> ~0.6B-param model with `dim=1024, L=28`. The name is cosmetic.

## Key knobs for RAM

### 1. Quantized KV cache (`--q8-kv`)
K/V drops from F32 (4 B/elem) to Q8_0 (34 B/32 elems): **~3.8× less RAM**.
Verified with `make kvtest`: rmse≈0.27 in logits, **94 % argmax agreement** with
F32 KV on greedy decoding → near-identical quality.

```
# Qwen3-0.6B, ctx=8192
ram: pesos=319 MB (mmap) | runtime=1819 MB = KV(F32,ctx=8192)+...   # default F32
ram: pesos=319 MB (mmap) | runtime=503 MB  = KV(Q8_0,ctx=8192)+...  # --q8-kv
```

### 2. Effective context (`-c N` / `--ctx N`)
KV cache sizes to the session's **actual** context, not the model's
`context_length` (which can be 262144). Less context = less RAM instantly.

### 3. Auto budget (`--max-ram MB`)
```
$ gguf2bin2 run qwen.g2bx "Hello" --max-ram 2048
```
Turns on Q8_0 KV first, then halves the context until the runtime footprint
(reported on `ram:`) fits the budget. Ideal for 2 GB machines.

### 4. Use disk as RAM — `--swap [PATH]`
Backs the **KV cache as a file** (default `D:\gguf2bin2_kv.swap` if D: exists).
Cold KV pages get paged to disk, freeing physical RAM. Measured on Qwen3-0.6B
(ctx 8192, 4 threads):

| Config | Private heap | Physical (WorkingSet) | tok/s |
|--------|-------------|----------------------|-------|
| KV F32 in RAM (ctx 8192) | ~1.9 GB | ~4 GB | — |
| Auto-Q8 (KV in RAM) | **513 MB** | 672 MB | 10.1 |
| **`--swap` D: (KV on disk)** | **37 MB** | 670 MB | 11.1 |

```
$ gguf2bin2 chat qwen.g2bx --no-think --swap --threads 4
```

### What counts against a 2 GB budget (and what does not)

| Component | Counts? | How to reduce |
|-----------|---------|---------------|
| Weights (mmap) | **No** (evictable page cache) | — |
| KV cache | **Yes** | `--q8-kv`, `-c N` |
| Buffers / activations | Yes (small) | — |
| Tokenizer (Qwen vocab ~250k) | Yes (~30–60 MB) | — |
| Logits (vocab×4B) | Yes (~1 MB × 250k) | — |


## v4.4 changes

- **--drop N (ShortGPT)**: mide Block Influence (1-cos entrada/salida) por bloque
  durante una calibracion rapida y omite los N bloques menos influyentes.
  Implementado en los tres caminos (Qwen/Llama secuencial, batched y LFM2).
  En LFM2.5-1.2B el tradeoff no compensa (BI minimo 0.106): queda como
  herramienta para modelos con bloques redundantes.
## v4.3 changes

- **Kernel Q4_0 de decode con acumulación diferida**: el dot entero por bloque
  se convierte y escala en un acumulador `__m256` (FMA) con UN hsum por fila —
  antes había un hsum por bloque de 32 (≈la mitad del coste del kernel).
  Qwen2.5-3B: 3.1 → **4.3 tok/s** (2.9× vs v3.5); Qwen3-0.6B: +10%.
- Soporte **Q5_0 end-to-end** (dequant escalar + kernel fusionado AVX2 con LUT
  de bits altos): los GGUF reales tipo Q4_K_M de modelos pequeños usan Q5_0 en
  attn/ffn; antes se descartaban esos tensores silenciosamente.
- Tabla de calidad medida (comando ppl, corpus README, SmolLM2-135M-Instruct):
  uniforme-Q4_0 = ppl **73.7** (roto), Q4_K_M nativo = **48.7**, Q6_K = 48.0,
  Q8_0 = 48.0. Los packs nativos K-quant conservan la calidad del Q8_0 dentro
  del 1.4% y corren MÁS rápido que el uniforme-Q4_0.

## v4.2 changes

- **Kernels AVX2 fusionados para Q4_K y Q6_K**: dot entero contra activación
  Q8 sin dequantizar la fila (`maddubs` + término de corrección m·Σx con sumas
  por bloque de 16). Antes cualquier pack con K-quants caía al fallback
  dequant-por-fila (2-5× más lento). Validados contra una referencia
  independiente byte a byte con `tools/qkcheck` (rel_err ≈ nivel de cuantización
  de la activación; batch B=2 idéntico a B=1).
- **Comando `ppl`**: harness de calidad (cross-entropy / perplexity sobre un
  texto) para medir el impacto real de los quants.
- **bench min-of-3** en decode y prefill: los portátiles con thermal throttling
  mienten; se reporta el mejor de 3 repeticiones.
- Limpieza: campo `kvrow` muerto eliminado, `srand()` muertos fuera,
  semilla temporal cuando no hay `--seed`.

## v4.1 changes

- **K-quants arreglados contra la referencia oficial ggml**: `deq_q3_K`,
  `deq_q4_K` y `deq_q5_K` estaban rotos desde v3.4 (mitad del tensor sin
  escribir + interleave de escalas incorrecto) → un GGUF Q3/Q4/Q5_K producía
  basura silenciosa, incluso al reempaquetar con `--q4`. Ahora verificados
  línea a línea con ggml-quants.c (Q2_K y Q6_K ya eran correctos).
- **Prefill batcheado** (`model_prefill`, B=8): cada fila de pesos se lee UNA
  vez para 8 tokens; cada fila K/V se dequantiza una vez para todo el chunk;
  atención causal batched. Logits idénticos bit a bit al camino secuencial.
- **Validación de geometría en carga**: `n_heads % n_kv_heads != 0` ahora es
  error; `head_dim % 32 != 0` fuerza KV F32 (evitaba lecturas fuera de bloque).
- **Guard de contigüidad** antes de fusionar QKV / gate+up (con padding ALIGN64
  la fusión habría corrompido silenciosamente).
- **Scratch persistente** de activación Q8 (global, cuantizado en el hilo
  maestro): elimina ~210 malloc/free por token del kernel Q4.
- `model_sample()` ya no usa rand() (RAND_MAX=32767 en Windows).

## v4.0 changes

- **Prefill sin logits** (`model_forward_ex(..., want_logits)`): durante el
  prompt solo el último token calcula logits (matmul vocab×dim, ~87 MB de
  tráfico por token en vocab 152k). Prefill 1.32× más rápido.
- **Atención GQA-major**: cada fila K/V se dequantiza UNA vez por grupo de
  heads (antes `group` veces: ×2–8 menos trabajo de dequant + softmax AVX2).
- **softmax/silu AVX2** con exp rápida vectorizada (reducción rango + Horner
  deg-6, err rel <2e-7); `silu_mul` fusiona gate×up (una pasada menos).
- **Fix rmsnorm AVX2**: la cola de la suma solo cubría `n&7` elementos; para
  dims no múltiplo de 32 la norma era incorrecta.
- **Sampling nuevo**: quickselect top-k O(n) (antes qsort de 152k por token),
  truco Gumbel-max para muestreo softmax exacto sin exponenciales, PRNG
  xorshift64* (`rand()`/RAND_MAX=32767 en Windows rompía la distribución) y
  `--seed N` reproducible. Generación con sampling: 1.72× más rápida.
- **GGUF por mmap** en el parser: empaquetar modelos grandes ya no copia el
  archivo completo a RAM.
- **Compactación de contexto en chat**: si el turno no cabe, conserva system +
  turnos recientes y re-prefill; chats largos ya no mueren con ctx pequeño.
- **bench --prefill N**: mide tok/s de procesamiento de prompt.
- g2bx info: nombres de roles de bias; limpieza general.

## v3.4 / v3.5 changes

- **Q4_0 AVX2 kernel** with 2-blocks/iteration interleaving and 8 independent
  accumulators (ILP); Q8_0 2-block kernel. `-mavx2 -mfma -flto`.
- **AVX2 attention**: Q·K dot product and the V-accumulate loop vectorized
  (32 floats/iteration), scalar fallback preserved.
- **K-quant dequant (Q2_K … Q8_K)** scalar support wired into `gguf_dequant`
  (pack can now ingest K-quant GGUFs and re-emit Q4_0 with `--q4`).
- **KV cache Q8_0** (`F_KV_Q8`): `--q8-kv` and auto-enable with `--max-ram`
  (or once F32 KV would exceed 1 GB).
- **Effective context** (`model_set_ctx`): KV no longer tied to `seq_len`.
- **RAM budget** (`model_auto_budget` + `model_ram_report`): auto-Q8 + ctx halving.
- **KV swap** (`--swap [PATH]`): file-backed KV; cold pages to disk.
- **Generic `read_cfg`**: accepts any `general.architecture` (e.g. `dflash`),
  using the real name as the metadata prefix with qwen/llama fallbacks.
- **Bugfix `vocab=0`**: `tokenizer.ggml.tokens` is an array; its length is read
  via `gguf_meta_arr_len` instead of `gguf_meta_i64`.
- **Draft/decode-only model warning**: if `token_embd.weight` is missing
  (vibe-coder DSpark/Dflash speculative-decoding GGUFs), `pack` warns they
  cannot generate standalone rather than silently producing garbage.
- **RoPE fix**: LLaMA-style RoPE now uses the correct `-2.0/head_dim`
  frequency step (was `-1.0`, producing corrupt output on real Llama weights);
  Qwen NEOX RoPE was already correct.

## Historical root cause — NEOX vs LLaMA RoPE

Qwen3 uses **NEOX RoPE** (half-split). With LLaMA interleaved RoPE the output is
UTF-8 garbage.

```c
// LLaMA: pairs (0,1),(2,3)...
// NEOX:  pairs (0,D/2),(1,D/2+1)...
```

## Build

```bash
# Linux / MinGW with OpenMP + AVX2 (recommended)
make

# Smoke test
make test

# Numerical KV F32-vs-Q8 check (builds a synthetic model)
make kvtest

# Without OpenMP/AVX2 (scalar fallback)
gcc -O2 -std=c99 -Iinclude -o gguf2bin2.exe \
  src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c \
  src/l5_model.c src/l6_token.c src/main.c -lm
```

> **Speed:** compile with `-mavx2 -mfma -fopenmp` (the default `make`). Without
> AVX2 the matmuls fall back to scalar kernels and speed drops ~8×.

## Usage

```bash
# Pack (+ --q4 to downsample Q8_0 -> Q4_0: half the weight bytes)
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen.g2bx
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen-q4.g2bx --q4

# Info
./gguf2bin2 info qwen.g2bx

# Completion (greedy), and see the RAM footprint
./gguf2bin2 run qwen.g2bx "The capital of France is" -n 20 -t 0

# Chat no-think: force Q8_0 KV and a 2 GB budget, 4 threads
./gguf2bin2 chat qwen.g2bx --no-think -n 40 -t 0.7 --max-ram 2048 --threads 4

# Small context by hand (less KV RAM)
./gguf2bin2 run qwen.g2bx "Hello" -c 1024 -n 20

# Auto budget (picks Q8 + lowers ctx only if needed)
./gguf2bin2 run qwen.g2bx "Hello" --max-ram 1024

# CPU speed: AVX2 + 4 threads
./gguf2bin2 bench qwen.g2bx -n 32 --threads 4
```

`--threads N` controls OpenMP threads; the recommended value is the number of
physical cores.

## G2BX format

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B-aligned | tokenizer
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

## Supported types

| Type | Load | Fused matmul |
|------|------|--------------|
| F32 / F16 | yes (pack→Q4_0 if a weight) | F32 path / dequant |
| Q4_0 / Q4_1 | yes | Q4_0 AVX2/scalar |
| Q8_0 | yes | Q8_0 AVX2/scalar |
| Q2_K / Q3_K / Q4_K / Q5_K / Q6_K | yes (dequant) | scalar fallback |
| Q8_K / IQ* | partial | — |

## Layout

```
include/g2b.h
src/l1_gguf.c    # GGUF parser (mmap)
src/l2_codec.c   # dequant + matmul (Q8_0/Q4_0 AVX2; K-quant dequant)
src/l3_math.c    # rmsnorm, rope, silu, softmax AVX2 + exp rápida
src/l4_gbin.c    # G2BX pack (generic per-architecture read_cfg)
src/l5_model.c   # load + forward + KV Q8 + RAM budget + atención GQA-major
src/l6_token.c   # BPE tokenizer
src/main.c       # CLI (sampling quickselect/Gumbel, chat con compactación)
tools/kvtest.c   # numerical KV F32 vs Q8 check
tools/prefilltest.c # equivalencia bit-exacta prefill batcheado vs secuencial
tools/exptest.c  # accuracy test del exp vectorizado
tools/mmbench.c  # raw kernel bandwidth microbench
tools/dump_gguf.py
docs/RESEARCH.md
```
