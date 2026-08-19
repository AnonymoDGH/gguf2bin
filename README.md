

# gguf2bin2 — G2BX FIXED v3.4 (2 GB RAM)

Custom runtime in C99: **GGUF → G2BX → inference** (Qwen3 / Qwen2 / Llama / custom architectures).

**Validated:** `The capital of France is` → `10,000 km²…` · Chat `What is the capital of France?` → `**Paris**`  
(Qwen3-0.6B-Q8_0 core; also tested against llama.cpp b10064 in v3.3).

## What's New in v3.4 — Running Heavy Models on 2 GB RAM

Weights are loaded with **mmap** (they don't occupy heap; the OS evicts pages), so the real RAM limit is the **KV cache + buffers + tokenizer**. Three levers:

### 1. Q8_0 Quantized KV Cache (`--q8-kv`)

The K/V cache goes from F32 (4 B/elem) to Q8_0 (34 B/32 elems): **~3.8× less RAM**.
Numerically verified (tool `make kvtest`): rmse≈0.27 on logits, **94% argmax match** against F32 KV on greedy steps → nearly identical quality.

```
# Qwen3-0.6B, ctx=8192
ram: weights=604 MB (mmap) | runtime=1819 MB = KV(F32,ctx=8192)+buffers+tokenizer   # no options
ram: weights=604 MB (mmap) | runtime=503 MB  = KV(Q8_0,ctx=8192)+buffers+tokenizer  # --q8-kv
```

### 2. Adjustable Effective Context (`-c N` / `--ctx N`)

The KV cache is sized to the **actual** session context, not the model's `context_length` (which can be 262144). Less context = less RAM instantly.

### 3. Automatic Budget (`--max-ram MB`)

```
$ gguf2bin2 run qwen.g2bx "Hola" --max-ram 2048
```
It adjusts: first enables **KV Q8_0**, then reduces context until the runtime footprint (reported in `ram:`) fits the budget. Ideal for 2 GB machines.

### 4. Use Disk as RAM — `--swap [PATH]`

Backs the **KV cache to a file** (default `D:\gguf2bin2_kv.swap` if D: exists). Cold KV pages are **swapped to disk**, freeing physical RAM. Measured on Qwen3-0.6B (ctx 8192, 4 threads, your PC):

| Config | Heap private | Physical RAM (WorkingSet) | tok/s |
|--------|-------------|---------------------------|-------|
| KV F32 in RAM (ctx 8192) | ~1.9 GB | ~4 GB | — |
| Auto-Q8 (KV in RAM) | **513 MB** | 672 MB | 10.1 |
| **`--swap` D: (KV on disk)** | **37 MB** | 670 MB | 11.1 |

```
$ gguf2bin2 chat qwen.g2bx --no-think --swap --threads 4
ram: weights=604 MB (mmap) | runtime=503 MB = KV(Q8_0,ctx=8192) (file-backed) + buffers + tokenizer
```

### What Counts Against the 2 GB (and What Doesn't)

| Component | Counts? | How to reduce |
|-----------|---------|---------------|
| Weights (mmap) | **No** (page cache evictable) | — |
| KV cache | **Yes** | `--q8-kv`, `-c N` |
| Buffers/activations | Yes (small) | — |
| Tokenizer (Qwen vocab ~250k) | Yes (~30-60 MB) | — |
| Logits (vocab×4B) | Yes (~1 MB × 250k) | — |

## v3.4 Changes (Detail)

- **KV cache Q8_0** (`F_KV_Q8`): `--q8-kv` / auto‑enable with `--max-ram` (and auto if F32 KV exceeds 1 GB).
- **Effective context** (`model_set_ctx`): KV cache no longer tied to `seq_len`.
- **RAM budget** (`model_auto_budget` + `model_ram_report`): auto‑Q8 + down‑scale ctx.
- **KV swap** (`--swap [PATH]` → defaults to `D:`): KV backed to file; cold pages to disk, heap → tens of MB (`--f32-kv` forces KV in RAM).
- **Generic `read_cfg`**: accepts any `general.architecture` (e.g. `dflash`), using the real name as metadata prefix with qwen/llama fallbacks.
- **Bugfix `vocab=0`**: `tokenizer.ggml.tokens` is an array; now reads its length (`gguf_meta_arr_len`) instead of `gguf_meta_i64`.
- **Warning for draft/decode‑only models**: if no `token_embd.weight` (e.g. speculative‑decoding GGUFs like `dflash`/`dspark`), the packer warns they cannot generate standalone text (depend on a teacher model), instead of failing silently.
- **Optimized attention**: the K/V row is dequantized once per position (also in the F32 path), not once per head → fewer cache reads.

## Historical Root Cause — RoPE NEOX vs LLaMA

Qwen3 uses **NEOX RoPE** (half‑split). Using LLaMA interleaved RoPE produces garbage UTF‑8.

```c
// LLaMA: pairs (0,1),(2,3)...
// NEOX:  pairs (0,D/2),(1,D/2+1)...
```

## Compilation

```bash
# Linux / MinGW with OpenMP + AVX2 (recommended)
make

# Smoke test
make test

# Numerical verification KV F32 vs Q8 (generates a synthetic model)
make kvtest

# Without OpenMP (fallback)
gcc -O2 -std=c99 -Iinclude -o gguf2bin2.exe \
  src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c \
  src/l5_model.c src/l6_token.c src/main.c -lm
```

> **Important (speed):** compile with `-mavx2 -mfma -fopenmp` (that's the default `make`). Without AVX2, matmul falls back to scalar kernels and speed drops ~8×.

## Performance (Qwen3-0.6B-Q8_0, i5 4‑core CPU, mmap)

| Build | tok/s |
|-------|-------|
| Scalar, 1 thread | 0.7 |
| AVX2, 1 thread | 4.9 |
| AVX2, 4 threads (Q8) | 10.1 |
| **AVX2, 4 threads (Q4)** | **11.3** |

Decode is limited by the matmul kernel + memory bandwidth. The **optimized Q4_0 kernel (2 blocks/iteration)** matches/exceeds Q8 while using **half the RAM** for weights (335 MB vs 633 MB). `pack --q4` gives the best tok/s and leaves room for larger models in the same 2 GB.

## Usage

```bash
# Pack (+ --q4 to downgrade Q8_0→Q4_0: half the weights)
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen.g2bx
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen-q4.g2bx --q4

# Info
./gguf2bin2 info qwen.g2bx

# Complete (greedy) — and see RAM footprint
./gguf2bin2 run qwen.g2bx "The capital of France is" -n 20 -t 0

# Chat no‑think, forcing KV Q8_0 and max 2 GB RAM, 4 threads
./gguf2bin2 chat qwen.g2bx --no-think -n 40 -t 0.7 --max-ram 2048 --threads 4

# Manual short context (less KV cache RAM)
./gguf2bin2 run qwen.g2bx "Hola" -c 1024 -n 20

# Automatic budget (chooses Q8 + down‑scales ctx only if needed)
./gguf2bin2 run qwen.g2bx "Hola" --max-ram 1024

# CPU performance: AVX2 + 4 threads
./gguf2bin2 bench qwen.g2bx -n 32 --threads 4

# Pack Q8_0 weights as Q4_0: half space, similar speed
./gguf2bin2 pack model.gguf model-q4.g2bx --q4
```

The default build uses `-O3 -mavx2 -mfma -fopenmp`. On a CPU without AVX2, manual compilation without those flags is required. `--threads N` controls OpenMP threads; the recommended value is the number of physical cores.

## G2BX Format

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B aligned | tokenizer
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

## Supported Types

| Type | Load | Matmul fused |
|------|------|--------------|
| F32 / F16 | yes (pack→Q4_0 if weight) | F32 path / dequant |
| Q4_0 / Q4_1 | yes | Q4_0 AVX2/scalar |
| Q8_0 | yes | Q8_0 AVX2/scalar |
| Q1 / Q2_K / Q3_K / Q4_K / … | **no** | — |

## Structure

```
include/g2b.h
src/l1_gguf.c    # GGUF parser
src/l2_codec.c   # dequant + matmul
src/l3_math.c    # rmsnorm, rope, silu, f32_to_half
src/l4_gbin.c    # G2BX packer (generic read_cfg by architecture)
src/l5_model.c   # load + forward + KV Q8 + RAM budget
src/l6_token.c   # BPE tokenizer
src/main.c       # CLI
tools/kvtest.c   # numerical verification KV F32 vs Q8
tools/dump_gguf.py
docs/RESEARCH.md
```
