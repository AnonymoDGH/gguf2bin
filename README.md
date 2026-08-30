# gguf2bin2

**English** | [Español](README.es.md)

**C99 LLM runtime for low-RAM machines** — GGUF → G2BX (own format) → inference.
Weights are memory-mapped: your real RAM budget is **KV cache + activations + tokenizer**, not the model file.

![C99](https://img.shields.io/badge/C99-portable-blue) ![AVX2](https://img.shields.io/badge/kernel-AVX2%20%2B%20FMA-orange) ![Vulkan](https://img.shields.io/badge/GPU-Vulkan%20WIP-purple) ![RAM](https://img.shields.io/badge/min%20RAM-37%20MB-green)

> Runs a **3B model on 145 MB of RAM** and a large model on a **2 GB machine** (`--swap`: KV backed on disk, footprint ≈ 37 MB).

---

## ⚡ Speed

Measured on Intel i5-6200U · 2C/4T · DDR3L dual-channel (~9.4 GB/s bus ceiling).
Sustained decode with `--fast` (high priority + OpenMP + quantized KV).

| Model | Weights (mmap) | Runtime RAM | decode | prefill |
|---|---|---|---|---|
| **Qwen2.5-3B** Q4_0 | 1992 MB | 145 MB | **4.3** | 7.8 |
| **Qwen3-0.6B** Q4_0 | 319 MB | 511 MB | **25.3** | **47.9** |
| **LFM2.5-1.2B** Q4_0S | 567 MB | 631 MB | **15.7** | 17.9 |
| Llama-3.2-1B F16 | 804 MB | 644 MB | 13.8 | — |
| SmolLM2-135M Q4_0 | 72 MB | 40 MB | **59.5** | — |

Measured 2026-08-30 on the same i5-6200U box with `bench -n 32 --prefill 256` (min of 3). At ~25 tok/s the decode streams ~9 GB/s — pinned to the DDR3L bus ceiling. Batched prefill hits the kernel compute ceiling (~27 GMAC/s).

## 🚀 Quick start

```bash
make            # Linux / MinGW · OpenMP + AVX2 recommended
make test       # smoke test on a synthetic model

# Pack a GGUF once, run forever
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen.g2bx --q4
./gguf2bin2 chat qwen.g2bx --no-think --fast
```

`--threads N` picks OpenMP threads; physical cores is the sweet spot.

## 🧠 RAM knobs

| Knob | Effect |
|---|---|
| `--q8-kv` | KV cache F32→Q8_0: **~3.8× less RAM** (94 % greedy argmax agreement) |
| `-c N` | Context sizes to *your* session, not the model's 262k max |
| `--max-ram MB` | Auto: enables Q8 KV first, then halves context until it fits |
| `--swap [PATH]` | KV cache backed on disk → **37 MB heap** even for big models |

```bash
$ gguf2bin2 run qwen.g2bx "Hello" --max-ram 2048   # ideal for 2 GB machines
```

### What counts against a 2 GB budget?

| Component | Counts? | Fix |
|---|:-:|---|
| Weights (mmap page cache) | ❌ evictable | — |
| KV cache | ✅ | `--q8-kv`, `-c N`, `--swap` |
| Buffers / activations | ✅ small | — |
| Tokenizer (~250k vocab) | ✅ ~30–60 MB | — |

## 🎛 CLI

```bash
gguf2bin2 pack model.gguf out.g2bx [--q4]     # GGUF → G2BX (--q4: half the bytes)
gguf2bin2 info model.g2bx                     # slots, geometry, types
gguf2bin2 run m.g2bx "prompt" [-n N] [-t T] [--bos] [--gpu]
gguf2bin2 chat m.g2bx [--no-think] [--fast] [--swap]
gguf2bin2 bench m.g2bx [-n 32] [--prefill 256]
gguf2bin2 ppl m.g2bx -f text.txt              # quality harness (perplexity)
gguf2bin2 vkinfo                              # Vulkan probe
```

Sampling: quickselect top-k O(n) + Gumbel-max + xorshift64\* reproducible via `--seed`.

### 🎮 Dual band CPU+GPU (`--gpu`)

The head GEMV (vocab×dim, the heaviest layer) gets split between CPU and GPU with automatic calibration:

```
[gpu] worker ready
[gpu] dual band: cpu=[0..44855) gpu=[44855..65536)  tc=6.1ms tg=13.1ms
```

- **Vulkan worker in a child process** — if the driver crashes or hangs, the runtime falls back to CPU-only without interrupting generation.
- **Vulkan loader bypass**: loads the ICD directly from DriverStore (useful on systems with a broken `Khronos\Vulkan\Drivers` registry).
- **Automatic optimal split**: `gpu = vocab·tc/(tc+tg)` measured on the first token; if the GPU is >4× slower than the CPU it shuts itself off.
- Supports **Q4_0** and **Q4_0S** heads. Bit-identical output vs the CPU path (greedy).

> ⚠️ On hardware where the iGPU shares the RAM bus with the CPU (HD 520 + DDR3L) there is no net gain — calibration detects it and disables itself. The real payoff comes with a dGPU with dedicated VRAM.

## 📐 Quality

Perplexity (internal corpus, SmolLM2-135M-Instruct): uniform Q4_0 = 73.7 *(broken)*,
native Q4_K_M = **48.7**, Q6_K = 48.0, Q8_0 = 48.0. Native K-quants keep Q8_0-level
quality within 1.4 % while running faster than uniform Q4_0.

Included numerical validation: `make kvtest` (F32 vs Q8 KV), `tools/prefilltest`
(bit-exact batched prefill), `tools/qkcheck` (46/46 K-quant kernels).

<details>
<summary><b>📦 G2BX format & supported types</b></summary>

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B-aligned | tokenizer
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

| Type | Load | Fused matmul |
|------|------|--------------|
| F32 / F16 | yes (pack→Q4_0 if weight) | F32 path / dequant |
| Q4_0 / Q4_1 / Q5_0 | yes | AVX2 fused |
| Q8_0 | yes | AVX2 fused |
| Q4_0S (own, fp16 shared scale /256) | yes | AVX2 batched |
| Q2_K … Q6_K | yes (dequant) | AVX2 fused (Q4_K/Q6_K), scalar rest |
| IQ* / Q8_K | partial | — |

</details>

<details>
<summary><b>🗂 Project layout</b></summary>

```
include/g2b.h      common API
src/l1_gguf.c      GGUF parser (mmap)
src/l2_codec.c     fused dequant + matmul (AVX2)
src/l3_math.c      rmsnorm, rope, silu, softmax AVX2
src/l4_gbin.c      G2BX packer (generic per-architecture read_cfg)
src/l5_model.c     load + forward + Q8 KV + RAM budget + GQA-major attention
src/l6_token.c     BPE tokenizer
src/main.c         CLI (quickselect/Gumbel sampling, chat with context compaction)
src/l7_vulkan.c    dual band GPU backend (child-process worker, crash-proof)
shaders/           compute shaders (Q4_0/Q4_0S GEMV)
tools/             kvtest · prefilltest · exptest · qkcheck · mmbench · dump_gguf.py
docs/RESEARCH.md   research notes and performance roadmap
```

</details>

<details>
<summary><b>📜 Changelog</b></summary>

#### v4.5
- Batched (prefill) kernel with deferred accumulation: same treatment as the decode kernel. Qwen2.5-3B prefill 4.3 → 7.8 tok/s (+81 %), bit-exact (`tools/prefilltest`). Sets the stage for speculative verification.
- **Dual band CPU+GPU head GEMV**: Vulkan worker in a child process (crash-proof), loader bypass loading the ICD straight from DriverStore, automatic split calibration with self-shutdown when the GPU doesn't help. Heads Q4_0/Q4_0S, bit-identical output.

#### v4.4
- **--drop N (ShortGPT)**: measures per-block Block Influence during a quick calibration and skips the N least influential blocks. On LFM2.5-1.2B it doesn't pay off (min BI 0.106).

#### v4.3
- Decode Q4_0 kernel with deferred accumulation: one hsum per row instead of one per block. Qwen2.5-3B 3.1 → 4.3 tok/s (2.9× vs v3.5); Qwen3-0.6B +10 %.
- Q5_0 end-to-end (fused AVX2 kernel with high-bits LUT).
- Measured quality table (`ppl` command).

#### v4.2
- Fused AVX2 kernels for Q4_K and Q6_K (`maddubs` + m·Σx correction term). Before, any K-quant pack fell to the 2–5× slower fallback. Validated byte-by-byte with `tools/qkcheck`.
- `ppl` command; min-of-3 bench (thermal throttling lies).

#### v4.1
- K-quants fixed against official ggml (`deq_q3_K/q4_K/q5_K` broken since v3.4: half the tensor unwritten + wrong scale interleave).
- Batched prefill (B=8) bit-exact; geometry validation at load; persistent Q8 activation scratch (−210 malloc/free per token).

#### v4.0
- Prefill without logits (only the last token computes vocab×dim): prefill 1.32×.
- GQA-major attention: each K/V row dequantized once per head group.
- softmax/silu AVX2 with fast exp (rel err < 2e-7); rmsnorm fix for non-multiple-of-32 tails.
- New sampling: O(n) quickselect top-k, Gumbel-max, xorshift64\*, reproducible `--seed`.
- GGUF via mmap; long-chat context compaction.

#### v3.x
- Q4_0 AVX2 (2 blocks/iter, ILP), AVX2 attention, full K-quant dequant.
- Q8_0 KV cache (`--q8-kv`), effective context, RAM budget (`--max-ram`), disk swap.
- LLaMA RoPE fix (−2.0/head_dim step) and NEOX vs LLaMA: the historical root of corrupt output.

</details>
