# Changelog

**English** | [Español](CHANGELOG.es.md)

#### v5.1.2 — real-model validation, Android build, README
- **CLI `run`**: the prompt was built with a trailing space ("…is " → last token `Ġ`), which derailed raw completions. Joined words now have no trailing space: on the release models `run m.g2bx "The capital of France is"` went from "3,000 km²…" / "which makes how many…" to "Paris" on both Qwen3-0.6B and LFM2.5-1.2B.
- **Android JNI**: prompts went through `GetStringUTFChars` (Modified UTF-8), so an emoji reached the tokenizer as a 6-byte surrogate pair and became 6 garbage tokens; strings are now encoded from UTF-16. The final result drops an incomplete trailing character (cut by `maxTokens`) instead of ending in U+FFFD.
- **`android/build_apk.ps1`**: still listed `src/l5_model.c` (split in Phase 3) and missed `src/internal` — the APK native build could not compile. Source list and include paths updated (link checked with `--no-undefined`; JNI exercised end to end under `-Xcheck:jni` with both release models).
- README rewritten (ready-made models, architectures, per-platform build, full CLI, C API example, testing, limitations); changelog moved here.

#### v5.1.1 — review fixes
- **CI green again** (red since Phase 3): `strdup`/`fseeko`/`ftello`/`clock_gettime` were implicitly declared under `-std=c99` — on Linux x86-64 the truncated `strdup` pointer crashed `make test`, and `ftello` truncated offsets >2 GB. Also fixed: fuzz job YAML (`>` folded the clang lines apart), `fmemopen` in the harnesses, MinGW `copy` under MSYS2 `sh`, CMake include dirs, AVX2 flags in the sanitizer job.
- **Tokenizer**: `u2b[289]` overflowed (68 remapped bytes → indices up to 323); bytes 0x7F–0xA0/0xAD decoded wrong (€, à, emojis). `tok_read_section` leaked on early errors.
- **Q4_0S_PSY kernels** (decode + batched) read nibbles 2 bytes off.
- **qwen35**: attention `wo` used `n=dim` instead of `n_heads*head_dim`. **LFM2/qwen35** recurrent state was never reset at `pos 0` (ppl windows, `chat_reset`, compaction and the Android app inherited the previous sequence).
- **LoRA**: batched prefill skipped the adapter (now falls back to sequential); loader validates rank and reads.
- **pack --prune**: `ffn_down` copy assumed one block per group (broken for F16/F32/Q8_0 down); OOM mid-prune now aborts. Tensors >4 GB are rejected instead of truncating `Slot.nbytes`.
- **Untrusted .g2bx hardening** (the Android app downloads from any URL): slot `nbytes` validated against geometry, geometry caps against i32 overflow, blob-past-EOF and offset-overflow checks, non-mmap fallback read from the right offset.
- **Android JNI**: `freeModel` could free the model while `generate` still ran (use-after-free); tokens are emitted as whole UTF-8 characters via UTF-16 (`NewStringUTF` broke on split characters and emojis).
- **Chat (LFM2)**: the empty `<think></think>` block (Qwen3 `enable_thinking=False` convention) was injected into LFM2.5 too, whose template has no such block; the model opened every answer "correcting itself". Found and verified on the release `lfm25-1.2b-q4s.g2bx` (now answers "Paris" / "Madrid").
- **ppl**: windows after the first now restart with BOS, like llama.cpp (LFM2 at `-c 128`: 485 → 109; single-window results unchanged).
- `g2b_pack` no longer leaks the Q4_0S/PSY/VVC mode into later calls; chat prompts are no longer truncated at 4/9 KB; default `--swap` file is per-process and opened with `O_NOFOLLOW`.

#### v5.0 — G2BX v3: CRC'd format + own type namespace
- **CRC32 footer**: every new `.g2bx` ends with `[crc32 of everything before][magic]`; the loader verifies on open (warming the page cache as a side effect) and rejects truncated/corrupt files with a clear message. New `verify` command (header + slots + types + geometry + CRC without loading weights).
- **Internal types at 0x80+**: Q4_0S/PSY/VVC leave IDs 25/26/27 (I16/I32/I64 in ggml today — a real collision). v1/v2 files are normalized on load; the packer already rejected native I16/I32/I64.
- **Written spec**: `docs/G2BX_SPEC.md` (layout, field-by-field LE, v1/v2/v3 compat matrix). Header serialization is now explicit LE in `g2bx_io` (byte-identical on x86).
- Harness-guarded refactors: l5 split, public API, `os_mm/sampler/opts/g2bx_io` modules, CI + CMake.

#### v4.9 — IQ1_S + Q3_K fused (27B desatascado)
- **IQ1_S integer dot** (`madd`+SAD, act Q8, 1 hsum/escala por 32): the 264 IQ1_S slots (~3.4 GB) ran scalar fallback; the old fused prototype existed but was never dispatched. Wired into `matmul_q`/`matmul_q_b`, validated by new `tools/iq1check` (vs exact-Q8 math: maxrel 5e-4).
- **Q3_K integer dot** (values −4..3, per-16 scales, bias −32): covers the 248k head (521 MB) + dense Q3_K models. Validated by new `tools/q3kcheck` incl. a 256-position one-hot sweep (caught a half-vector `cvtepi8` bug pre-ship: high 8 elems silently dropped).
- 27B hybrid (Qwen3.8, hybrid → always sequential decode): stock 132.7 s → **79.6 s (−40 %, 1.67×)** for prompt+2 tokens, warm page cache, i5-6200U. Greedy output differs in argmax (Q8 approximation on 1.5-bit weights — both outputs are IQ1_S-grade mojibake); math bounded by the harnesses above.

#### v4.8 — Blocked prefill (G=4 token blocking)
- **Weight traffic ÷4 in `matmul_q4_0_b` / `matmul_q4_0s_b`**: each weight row is unpacked once and reused for 4 tokens (was: re-streamed per token, 16× per batch). Qwen3-0.6B Q4_0 prefill 38.9 → **53.7 tok/s (+38 %)** and Qwen2.5-3B Q4_0 7.0 → **9.6 tok/s (+37 %)** (interleaved A/B on i5-6200U). Bit-exact (`prefilltest` diff 0 incl. 3B GQA, `q4bcheck` 5/5, ppl identical 58.709). Decode untouched (3B: 5.4 = 5.4); 27B IQ1_S hybrid output byte-identical to stock.

#### v4.7 — Q4_0S_PSY (psicoacústico) + fallback TLS
- **Q4_0S_PSY**: 2 escalas fp16 por 256 (132B vs 130B). La mejora de calidad anunciada queda retirada: la prueba local de 128 tokens dio ppl 2380555.838 frente a 82.325 base. El soporte permanece, desaconsejado hasta validar la causa. Fallback IQ usa TLS para evitar malloc por fila.

#### v4.6 — Swapeculative MV Triple Band
- **--mv 0.0..1.0**: tunable skip of FFN (dense) / SSM delta (hybrid) via hash + 2-bit predictor. `25.0 → 40.1 tok/s (+60%)` on Qwen3-0.6B Q4_0. **[RETIRADO v5.1: ppl 25.4 → 35 629 (×1400) a ratio 0.1, 352 904 a 0.5. La velocidad era la de un modelo roto. Ver Phase 7.]**

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
