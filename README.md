# gguf2bin2

**C99 LLM runtime for low-RAM machines** — GGUF → G2BX (own format) → inference.
Weights are memory-mapped: your real RAM budget is **KV cache + activations + tokenizer**, not the model file.

![C99](https://img.shields.io/badge/C99-portable-blue) ![AVX2](https://img.shields.io/badge/kernel-AVX2%20%2B%20FMA-orange) ![Vulkan](https://img.shields.io/badge/GPU-Vulkan%20WIP-purple) ![RAM](https://img.shields.io/badge/min%20RAM-37%20MB-green)

> Runs a **3B model on 145 MB of RAM** and a large model on a **2 GB machine** (`--swap`: KV backed on disk, footprint ≈ 37 MB).

---

## ⚡ Speed

Measured on Intel i5-6200U · 2C/4T · DDR3L dual-channel (~9.4 GB/s bus ceiling).
Sustained decode with `--fast` (high priority + OpenMP + quantized KV).

| Model | Weights (mmap) | Runtime RAM | tok/s |
|---|---|---|---|
| **Qwen2.5-3B** Q4_0 | 1992 MB | 145 MB | **4.3** |
| **Qwen3-0.6B** Q4_0 | 319 MB | 507 MB | **27+** |
| **LFM2.5-1.2B** Q4_0S | 569 MB | 598 MB | **15.0** |
| SmolLM2-135M Q4_0 | 72 MB | 40 MB | **59.5** |

At 27 tok/s the decode streams ~9 GB/s — essentially pinned to the DDR3L bus ceiling.
Batched prefill runs at the kernel compute ceiling (~27 GMAC/s).

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
$ gguf2bin2 run qwen.g2bx "Hello" --max-ram 2048   # ideal para máquinas de 2 GB
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
gguf2bin2 info model.g2bx                     # slots, geometría, tipos
gguf2bin2 run m.g2bx "prompt" [-n N] [-t T] [--bos] [--gpu]
gguf2bin2 chat m.g2bx [--no-think] [--fast] [--swap]
gguf2bin2 bench m.g2bx [-n 32] [--prefill 256]
gguf2bin2 ppl m.g2bx -f texto.txt             # harness de calidad (perplexity)
gguf2bin2 vkinfo                              # sonda Vulkan
```

Sampling: quickselect top-k O(n) + Gumbel-max + xorshift64\* reproducible con `--seed`.
`--gpu` activa el **dual band CPU+GPU** del head GEMV (worker Vulkan aislado en proceso hijo;
si el driver falla, cae a CPU sin interrumpir nada). Actualmente bloqueado por ICDs rotos
en el equipo de prueba — ver `docs/`.

## 📐 Quality

Perplexity (corpus interno, SmolLM2-135M-Instruct): uniforme-Q4_0 = 73.7 *(roto)*,
Q4_K_M nativo = **48.7**, Q6_K = 48.0, Q8_0 = 48.0. Los K-quants nativos conservan
la calidad del Q8_0 dentro del 1.4 % y corren más rápido que el uniforme-Q4_0.

Validación numérica incluida: `make kvtest` (KV F32 vs Q8), `tools/prefilltest`
(prefill batcheado bit-exacto), `tools/qkcheck` (46/46 kernels K-quant).

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
include/g2b.h      API común
src/l1_gguf.c      parser GGUF (mmap)
src/l2_codec.c     dequant + matmul fusionados (AVX2)
src/l3_math.c      rmsnorm, rope, silu, softmax AVX2
src/l4_gbin.c      packer G2BX (read_cfg genérico por arquitectura)
src/l5_model.c     load + forward + KV Q8 + budget RAM + atención GQA-major
src/l6_token.c     tokenizer BPE
src/main.c         CLI (sampling quickselect/Gumbel, chat con compactación de contexto)
src/l7_vulkan.c    backend GPU dual band (worker en proceso hijo, crash-proof)
shaders/           compute shaders (Q4_0/Q4_0S GEMV)
tools/             kvtest · prefilltest · exptest · qkcheck · mmbench · dump_gguf.py
docs/RESEARCH.md   notas de investigación y roadmap de rendimiento
```

</details>

<details>
<summary><b>📜 Changelog</b></summary>

#### v4.5
- Kernel batched (prefill) con acumulación diferida: mismo trato que el kernel de decode. Prefill Qwen2.5-3B 4.3 → 7.8 tok/s (+81 %), bit-exacto (`tools/prefilltest`). Prepara el terreno para la verificación especulativa.
- Dual band CPU+GPU del head GEMV: worker Vulkan en proceso hijo (crash-proof frente a drivers rotos), calibración automática del split y auto-apagado si la GPU es más lenta. *(v4.5-gpu)*

#### v4.4
- **--drop N (ShortGPT)**: mide Block Influence por bloque durante una calibración rápida y omite los N menos influyentes. En LFM2.5-1.2B no compensa (BI mínimo 0.106).

#### v4.3
- Kernel Q4_0 de decode con acumulación diferida: un hsum por fila en vez de uno por bloque. Qwen2.5-3B 3.1 → 4.3 tok/s (2.9× vs v3.5); Qwen3-0.6B +10 %.
- Q5_0 end-to-end (kernel AVX2 fusionado con LUT de bits altos).
- Tabla de calidad medida (comando `ppl`).

#### v4.2
- Kernels AVX2 fusionados para Q4_K y Q6_K (`maddubs` + corrección m·Σx). Antes cualquier pack K-quant caía al fallback 2–5× más lento. Validado byte a byte con `tools/qkcheck`.
- Comando `ppl`; bench min-of-3 (el thermal throttling miente).

#### v4.1
- K-quants arreglados contra ggml oficial (`deq_q3_K/q4_K/q5_K` rotos desde v3.4: mitad del tensor sin escribir + interleave incorrecto).
- Prefill batcheado (B=8) bit-exacto; validación de geometría en carga; scratch persistente de activación Q8 (−210 malloc/free por token).

#### v4.0
- Prefill sin logits (solo el último token calcula vocab×dim): prefill 1.32×.
- Atención GQA-major: cada fila K/V se dequantiza una vez por grupo de heads.
- softmax/silu AVX2 con exp rápida (err rel < 2e-7); fix rmsnorm cola no múltiplo de 32.
- Sampling nuevo: quickselect top-k O(n), Gumbel-max, xorshift64\*, `--seed`.
- GGUF por mmap; compactación de contexto en chat largo.

#### v3.x
- Q4_0 AVX2 (2 bloques/iter, ILP), atención AVX2, dequant K-quant completo.
- KV Q8_0 (`--q8-kv`), contexto efectivo, presupuesto RAM (`--max-ram`), swap en disco.
- Fix RoPE LLaMA (paso −2.0/head_dim) y NEOX vs LLaMA: la raíz histórica del output corrupto.

</details>
