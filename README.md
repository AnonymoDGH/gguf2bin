# gguf2bin2

**English** | [Español](README.es.md)

**A C99 LLM runtime for low-RAM machines.** Pack a GGUF once into G2BX (its own indexed, CRC-checked format), then run it with memory-mapped weights: the RAM you actually pay for is **KV cache + activations + tokenizer**, not the model file.

[![ci](https://github.com/AnonymoDGH/gguf2bin/actions/workflows/ci.yml/badge.svg)](https://github.com/AnonymoDGH/gguf2bin/actions/workflows/ci.yml)
![version](https://img.shields.io/badge/version-5.1.2-informational)
![C99](https://img.shields.io/badge/C99-portable-blue)
![AVX2](https://img.shields.io/badge/kernels-AVX2%20%2B%20FMA-orange)
![RAM](https://img.shields.io/badge/min%20RAM-37%20MB-green)

> A **3B model in 145 MB of RAM**, and large models on a **2 GB machine** (`--swap`: KV cache on disk, heap ≈ 37 MB).

- **mmap weights** — evictable page cache, not heap; RAM knobs for KV (`--q8-kv`, `-c`, `--max-ram`, `--swap`).
- **Fused AVX2 kernels** for Q4_0, Q8_0, Q5_0, Q4_K, Q6_K, Q3_K, IQ1_S and the own Q4_0S; batched prefill (Q4_0/Q4_0S read each weight row once per 4 tokens).
- **Architectures**: Llama, Qwen2/2.5, Qwen3, LFM2/LFM2.5 (conv hybrid), Qwen3.5 dense hybrid (experimental).
- **Hardened loader** — CRC footer, every slot checked against the model geometry: a corrupt or hostile `.g2bx` is rejected, not executed.
- **Android app** (`android/`) and a small **C API** (`include/gguf2bin.h`).

---

## 🚀 Try it in a minute

Ready-made models live in the [`models` release](https://github.com/AnonymoDGH/gguf2bin/releases/tag/models):

| File | Size | Model |
|---|---|---|
| [`Qwen3-0.6B-q4.g2bx`](https://github.com/AnonymoDGH/gguf2bin/releases/download/models/Qwen3-0.6B-q4.g2bx) | 319 MB | Qwen3-0.6B · Q4_0 |
| [`lfm25-1.2b-q4s.g2bx`](https://github.com/AnonymoDGH/gguf2bin/releases/download/models/lfm25-1.2b-q4s.g2bx) | 569 MB | LFM2.5-1.2B · Q4_0S |

```bash
make                                   # Linux: gcc + OpenMP + AVX2
curl -LO https://github.com/AnonymoDGH/gguf2bin/releases/download/models/Qwen3-0.6B-q4.g2bx
./gguf2bin2 chat Qwen3-0.6B-q4.g2bx --fast
./gguf2bin2 run  Qwen3-0.6B-q4.g2bx "The capital of France is" --bos -n 20 -t 0
```

Your own model: `./gguf2bin2 pack model.gguf model.g2bx --q4` (a `.gguf` path also works directly; it is packed once to `model.gguf.g2bx` and reused).

## 🧩 Supported architectures

| Family | GGUF `general.architecture` | Notes | Checked on |
|---|---|---|---|
| Llama (Llama 3.x, SmolLM2) | `llama` | interleaved RoPE | Llama-3.2-1B, SmolLM2-135M |
| Qwen2 / Qwen2.5 | `qwen2` | Q/K/V biases | Qwen2.5-3B |
| Qwen3 | `qwen3` | QK-norm, NEOX RoPE | Qwen3-0.6B (release model) |
| LFM2 / LFM2.5 | `lfm2` | short-conv + attention hybrid | LFM2.5-1.2B (release model) |
| Qwen3.5 (dense) | `qwen35` | gated delta-net + full attention every N layers, sequential decode | **experimental**: no real-model run since the v5.1.1 fixes |

Not supported: MoE (`qwen2moe`, `bailingmoe*`…) and vision towers. Chat templates: ChatML (`<|im_start|>`) and Llama 3 (`<|eot_id|>`).

## 🔧 Build

| Platform | Command |
|---|---|
| Linux (gcc) | `make` · `make test` |
| Windows (MSYS2 MINGW64) | `pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-libgomp mingw-w64-x86_64-make` → `mingw32-make` |
| CMake (any) | `cmake -S . -B build && cmake --build build && (cd build && ctest)` |
| No AVX2 (older x86, ARM…) | `make CFLAGS="-O2 -std=c99 -fopenmp -Iinclude -Isrc"` — scalar paths, same results, slower |
| Android APK | `android/build_apk.ps1` (Windows, NDK 28, build-tools 35, JDK) → `gguf2bin.apk` |

The default flags assume a CPU with AVX2 + FMA + F16C (Haswell / Zen 1 or newer). `--threads N` picks OpenMP threads; physical cores are the sweet spot.

## ⚡ Speed

Intel i5-6200U · 2C/4T · DDR3L (~9.4 GB/s bus ceiling), `bench -n 32` (min of 3), `--fast`:

| Model | Weights (mmap) | Runtime RAM | decode tok/s | prefill tok/s |
|---|---|---|---|---|
| **Qwen2.5-3B** Q4_0 | 1992 MB | 145 MB | **4.3** | 7.8 |
| **Qwen3-0.6B** Q4_0 | 319 MB | 511 MB | 25.0 | **47.9** |
| **LFM2.5-1.2B** Q4_0S | 567 MB | 631 MB | **15.7** | 17.9 |
| Llama-3.2-1B F16 | 804 MB | 644 MB | 13.8 | — |
| SmolLM2-135M Q4_0 | 72 MB | 40 MB | **59.5** | — |

Measured 2026-08-31. At ~25 tok/s decode is pinned to the ~9 GB/s memory bus; prefill hits the ~27 GMAC/s compute ceiling. Run `gguf2bin2 bench m.g2bx [--prefill 256] [--json]` on your machine.

## 🧠 RAM knobs

| Knob | Effect |
|---|---|
| `--q8-kv` | KV cache F32 → Q8_0: **~3.8× less RAM**; +0.6–0.8 % ppl on Qwen3-0.6B / LFM2.5-1.2B |
| `-c N` / `--ctx N` | Size the context to *your* session, not the model's 32k–262k maximum |
| `--max-ram MB` | Automatic: Q8 KV first, then halves the context until it fits |
| `--swap [PATH]` | KV cache backed by a file → **37 MB heap** even for big models (default: a per-process file in the temp dir; on Windows `D:\` if present) |

| Counts against a 2 GB budget? | | Fix |
|---|:-:|---|
| Weights (mmap page cache) | ❌ evictable | — |
| KV cache | ✅ | `--q8-kv`, `-c N`, `--swap` |
| Buffers / activations | ✅ small | — |
| Tokenizer (~150–250k vocab) | ✅ ~30–60 MB | — |

## 🎛 CLI reference

```text
gguf2bin2 pack   <model.gguf> <out.g2bx> [--q4 | --q4s] [--prune F [--calib text.txt]]
gguf2bin2 info   <model.g2bx|model.gguf>       geometry, types, estimated RAM
gguf2bin2 verify <model.g2bx>                  header + slots + geometry + CRC, no weight load
gguf2bin2 run    <model> [text] [opts]         raw completion
gguf2bin2 chat   <model> [opts]                interactive chat (type 'exit')
gguf2bin2 bench  <model> [-n 32] [--prefill N] [--json]
gguf2bin2 ppl    <model> [-f file|-] [-n max_tokens]
gguf2bin2 synth  <out.g2bx>                    tiny synthetic model for tests
gguf2bin2 vkinfo                               Vulkan probe
```

| Option | Commands | Default |
|---|---|---|
| `-n N` | run / chat / bench / ppl | 64 / 256 / 32 / 4096 |
| `-t TEMP` (0 = greedy) | run / chat | 0.7 |
| `--top-k K` · `--top-p P` | run / chat | 40 · 0.9 |
| `--repeat-penalty R` | run / chat | 1.1 / 1.05 |
| `--seed S` | run / chat | time-based |
| `--bos` · `--tokens a,b,c` | run | off |
| `--system TXT` · `--no-system` · `--think` / `--no-think` | chat | "You are a helpful assistant." · no-think |
| `--cyber adapter.lora` | run / ppl | LoRA adapter (v1/v2) |
| `-c N` · `--q8-kv` · `--f32-kv` · `--max-ram MB` · `--swap [PATH]` | run / chat / bench / ppl | model ctx · auto |
| `--fast` · `--threads N` · `--drop N` · `--gpu` | run / chat / bench | — |

`--q4` converts every weight to Q4_0 (half the bytes); `--q4s` to Q4_0S (one fp16 scale per 256, ~10 % fewer bytes than Q4_0) where the row length allows it. `--drop N` skips the N least influential blocks (ShortGPT). Sampling is O(n) top-k quickselect + Gumbel-max, reproducible with `--seed`.

<details>
<summary><b>🎮 Dual-band CPU+GPU head (<code>--gpu</code>, Windows)</b></summary>

The head GEMV (vocab×dim, the heaviest layer) is split between CPU and GPU with automatic calibration:

```
[gpu] worker ready
[gpu] dual band: cpu=[0..44855) gpu=[44855..65536)  tc=6.1ms tg=13.1ms
```

- Vulkan runs in a **child process**: if the driver crashes or hangs, generation continues on CPU.
- Loads the ICD straight from DriverStore (works with a broken `Khronos\Vulkan\Drivers` registry).
- Split `gpu = vocab·tc/(tc+tg)` measured on the first token; switches itself off if the GPU is >4× slower.
- Q4_0 and Q4_0S heads, bit-identical to the CPU path (greedy).

On iGPUs that share the RAM bus with the CPU (HD 520 + DDR3L) there is no net gain and calibration disables it; the payoff is a dGPU with its own VRAM. On other platforms `--gpu` is a no-op.

</details>

## 🧑‍💻 C API

`include/gguf2bin.h` is the only header an application needs (sessions, chat, generation, tokenizer, pack, ppl, bench):

```c
#include "gguf2bin.h"
#include <stdio.h>

static void on_token(const char *piece, void *ud){ (void)ud; fputs(piece, stdout); fflush(stdout); }

int main(void){
  g2b_config cfg = {0};
  cfg.ctx = 2048; cfg.q8_kv = -1;                 /* -1 = automatic */
  g2b_session *s = NULL;
  g2b_error e = g2b_open("Qwen3-0.6B-q4.g2bx", &cfg, &s);
  if(e){ fprintf(stderr, "open: %s\n", g2b_strerror(e)); return 1; }

  g2b_chat_begin(s, "You are a helpful assistant.", 1 /* no_think */);
  g2b_gen_params p = {0};
  p.on_token = on_token;
  p.temp = 0.7f; p.top_k = 40; p.top_p = 0.9f; p.repeat_penalty = 1.05f;
  p.max_tokens = 128; p.seed = 42;
  g2b_chat_turn(s, "What is the capital of France?", &p);   /* e.g. "The capital of France is Paris." */
  g2b_close(s);
  return 0;
}
```

```bash
cmake -S . -B build -DG2B_BUILD_TESTS=OFF && cmake --build build
gcc -O2 -std=c99 -Iinclude app.c build/libg2bcore.a -fopenmp -lm -ldl -o app
```

One active session per process (the compute scratch is global).

## 📱 Android

`android/` holds a minimal app (Java + JNI, no Gradle): download a `.g2bx` by pasting its URL, then chat with streaming, top-k and repetition penalty. Since files come from arbitrary URLs, the loader validates every slot before running anything. Build with `android/build_apk.ps1` (paths to the SDK/NDK are at the top of the script).

## 📐 Quality

Perplexity, short local checks (not a standard benchmark). Qwen/LFM2 rows: README + spec text, 1024 tokens; SmolLM2 row: a separate internal corpus.

| Model / format | ppl | vs base |
|---|---:|---|
| Qwen3-0.6B Q4_0 | 25.4 | — |
| Qwen3-0.6B Q8_0 | 20.8 | −18 % |
| LFM2.5-1.2B q4max | 28.1 | — |
| LFM2.5-1.2B Q4_0S | 34.9 | +24 % |
| SmolLM2-135M Q4_0 (uniform) / Q4_K_M / Q6_K / Q8_0 | 73.7 / 48.7 / 48.0 / 48.0 | native K-quants ≈ Q8_0 |

Custom types (IDs 0x80+, not readable by other runtimes): **Q4_0S** saves ~10 % of the bytes of Q4_0 at some quality cost (+24 % ppl on LFM2.5 above). **Q4_0S_PSY** had broken kernels until v5.1.1 (its old Llama-3.2-1B ppl of 2,380,556 vs 82.3 base measured the bug, not the format) and **Q4_VVC** is a plain 3-bit quant with one scale per 256 (Llama-3.2-1B: 1,177 vs 84.0 base); both stay unrecommended until re-measured.

## 🧪 Testing

```bash
make test                     # synth model + info/run/bench + apitest + selftest
make kvtest q4bcheck iq1check q3kcheck prefilltest
```

- `selftest`: sampler, mmap, G2BX I/O, CRC, kernel dispatch, tokenizer round-trip, kernels vs reference dequant, rejection of truncated slots.
- `prefilltest`: batched prefill is bit-exact with sequential decode; `kvtest`: F32 vs Q8 KV.
- `q4bcheck` / `iq1check` / `q3kcheck` / `tools/qkcheck.c`: fused kernels vs reference math.
- CI (Linux, MinGW, CMake, ASan+UBSan, libFuzzer on the GGUF/G2BX/tokenizer parsers) runs on every push.

## ⚠️ Known limitations

- The tokenizer is byte-level BPE **without the pre-tokenizer regex**, so token ids can differ from llama.cpp / Hugging Face on some inputs (text round-trips exactly).
- One session per process; Vulkan dual band is Windows-only.
- The packer skips BF16, Q5_1, Q8_1 and F64 tensors. Q8_K has no dequant (reads as zeros; it does not appear in normal model files). Types without a fused kernel (Q2_K, Q5_K, IQ2/3/4…) run through a slower dequant path.
- Multi-turn chat with Qwen3 keeps the empty `<think>` block of past turns in the history (the official template drops it).

<details>
<summary><b>📦 G2BX format</b></summary>

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B-aligned | tokenizer | [v3: crc32 + "G2BX"]
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

Tensors are indexed by role (O(1) lookup), every integer is explicit little-endian, v3 adds a CRC32 footer verified on load. Readers accept v1/v2/v3. Full spec: [`docs/G2BX_SPEC.md`](docs/G2BX_SPEC.md).

| Type | Load | Matmul |
|---|---|---|
| F32 / F16 | yes (packed to Q4_0 when it is a weight) | F32 path / dequant |
| Q4_0 / Q8_0 / Q5_0 | yes | AVX2 fused |
| Q4_0S (own: fp16 scale per 256) | yes | AVX2 fused + batched |
| Q4_K / Q6_K / Q3_K | yes | AVX2 fused |
| Q2_K, Q5_K, Q4_1, IQ2/IQ3/IQ4 | yes | dequant fallback |
| IQ1_S | yes | AVX2 fused |

</details>

<details>
<summary><b>🗂 Project layout</b></summary>

```
include/gguf2bin.h   public API (sessions)       src/internal/   shared internals
src/model.c          G2BX load, validation, RAM   src/kv.c        F32/Q8 KV, disk swap, runtime buffers
src/forward_*.c      dense / lfm2 / qwen35 / batched prefill
src/l1_gguf.c        GGUF parser (mmap)           src/l4_gbin.c   G2BX packer (+ prune)
src/l2_codec.c       dequant + fused matmul        src/l3_math.c   norms, RoPE, softmax
src/l6_token.c       BPE tokenizer                 src/l7_vulkan.c dual-band GPU
src/g2b_api.c        API implementation            src/g2bx_io.c   G2BX reader/writer/CRC/verify
src/main.c           CLI                           android/        app + JNI
tools/               test harnesses + fuzz/        docs/           spec, research notes, perf roadmap
```

</details>

## 📜 Changelog

See [CHANGELOG.md](CHANGELOG.md). Latest: **v5.1.2** — real-model validation of the v5.1.1 fixes, `run` prompts without a trailing space (raw completions now answer "Paris"), emoji-safe Android JNI and a working APK build script.

Vendored reference code from llama.cpp lives in `third_party/` under its own license ([`docs/LICENSE.llama_cpp`](docs/LICENSE.llama_cpp)).
