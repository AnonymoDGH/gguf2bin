# gguf2bin2 — C99 LLM runtime for low-RAM machines

A self-contained C99 runtime: **GGUF → G2BX (own format) → inference** for
Qwen3 / Qwen2 / Llama / custom architectures. Weights are memory-mapped so the
real RAM budget is **KV cache + activations + tokenizer**, not the model file.

## Speed & RAM (measured on Intel i5-6200U · 2C/4T · DDR3L dual-channel)

Sustained decode with `--fast` (high priority, OpenMP threads, quantized KV).
The DDR3L memory bus (~9.4 GB/s) is the hard ceiling; Q4_0 sits at ~50 % of it.

| Model (G2BX) | Format | Weights (mmap) | Runtime RAM | tok/s |
|--------------|--------|----------------|-------------|-------|
| Qwen3-0.6B | Q8_0 | 604 MB | 503 MB (auto-Q8, ctx 8192) | 11.4 |
| Qwen3-0.6B | Q4_0 | 319 MB | 503 MB (auto-Q8, ctx 8192) | **12–14** |
| Qwen3-0.6B | Q4_0 | 319 MB | **145 MB (**`--q8-kv -c 2048`**)** | 12.4 |
| Qwen3-0.6B | Q4_0 | 319 MB | **37 MB (**`--swap`**)** | 11.1 |
| SmolLM2-360M | Q4_0 | 194 MB | 657 MB | 18.3 |
| SmolLM2-135M | Q4_0 | 72 MB | 40 MB (`-c 2048`) | **35** |

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
src/l1_gguf.c    # GGUF parser
src/l2_codec.c   # dequant + matmul (Q8_0/Q4_0 AVX2; K-quant dequant)
src/l3_math.c    # rmsnorm, rope, silu, f32_to_half
src/l4_gbin.c    # G2BX pack (generic per-architecture read_cfg)
src/l5_model.c   # load + forward + KV Q8 + RAM budget
src/l6_token.c   # BPE tokenizer
src/main.c       # CLI
tools/kvtest.c   # numerical KV F32 vs Q8 check
tools/mmbench.c  # raw kernel bandwidth microbench
tools/dump_gguf.py
docs/RESEARCH.md
```