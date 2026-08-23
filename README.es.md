# gguf2bin2

[English](README.md) | **Español**

**Runtime LLM en C99 para máquinas con poca RAM** — GGUF → G2BX (formato propio) → inferencia.
Los pesos van memory-mapped: tu presupuesto real de RAM es **KV cache + activaciones + tokenizer**, no el archivo del modelo.

![C99](https://img.shields.io/badge/C99-portable-blue) ![AVX2](https://img.shields.io/badge/kernel-AVX2%20%2B%20FMA-orange) ![Vulkan](https://img.shields.io/badge/GPU-Vulkan%20WIP-purple) ![RAM](https://img.shields.io/badge/min%20RAM-37%20MB-green)

> Corre un modelo de **3B en 145 MB de RAM** y un modelo grande en una **máquina de 2 GB** (`--swap`: KV respaldada en disco, footprint ≈ 37 MB).

---

## ⚡ Velocidad

Medido en Intel i5-6200U · 2C/4T · DDR3L dual-channel (~9.4 GB/s de techo de bus).
Decode sostenido con `--fast` (prioridad alta + OpenMP + KV cuantizada).

| Modelo | Pesos (mmap) | RAM runtime | tok/s |
|---|---|---|---|
| **Qwen2.5-3B** Q4_0 | 1992 MB | 145 MB | **4.3** |
| **Qwen3-0.6B** Q4_0 | 319 MB | 507 MB | **24–28** |
| **LFM2.5-1.2B** Q4_0S | 569 MB | 598 MB | **16.1** |
| SmolLM2-135M Q4_0 | 72 MB | 40 MB | **59.5** |

A ~27 tok/s el decode fluye ~9 GB/s — prácticamente clavado al techo del bus DDR3L.
El prefill batcheado corre al techo de cómputo del kernel (~27 GMAC/s).
Prefill Qwen3-0.6B: 46.6 tok/s.

## 🚀 Inicio rápido

```bash
make            # Linux / MinGW · OpenMP + AVX2 recomendado
make test       # smoke test sobre un modelo sintético

# Empaqueta un GGUF una vez, corre para siempre
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen.g2bx --q4
./gguf2bin2 chat qwen.g2bx --no-think --fast
```

`--threads N` elige los hilos de OpenMP; los núcleos físicos son el punto óptimo.

## 🧠 Perillas de RAM

| Perilla | Efecto |
|---|---|
| `--q8-kv` | KV cache F32→Q8_0: **~3.8× menos RAM** (94 % de coincidencia argmax greedy) |
| `-c N` | El contexto se dimensiona a *tu* sesión, no al máximo de 262k del modelo |
| `--max-ram MB` | Auto: activa KV Q8 primero, luego parte el contexto a la mitad hasta caber |
| `--swap [RUTA]` | KV cache respaldada en disco → **heap de 37 MB** incluso con modelos grandes |

```bash
$ gguf2bin2 run qwen.g2bx "Hola" --max-ram 2048   # ideal para máquinas de 2 GB
```

### ¿Qué cuenta contra un presupuesto de 2 GB?

| Componente | ¿Cuenta? | Solución |
|---|:-:|---|
| Pesos (page cache del mmap) | ❌ desalojable | — |
| KV cache | ✅ | `--q8-kv`, `-c N`, `--swap` |
| Buffers / activaciones | ✅ pequeño | — |
| Tokenizer (vocab ~250k) | ✅ ~30–60 MB | — |

## 🎛 CLI

```bash
gguf2bin2 pack model.gguf out.g2bx [--q4]     # GGUF → G2BX (--q4: la mitad de bytes)
gguf2bin2 info model.g2bx                     # slots, geometría, tipos
gguf2bin2 run m.g2bx "prompt" [-n N] [-t T] [--bos] [--gpu]
gguf2bin2 chat m.g2bx [--no-think] [--fast] [--swap]
gguf2bin2 bench m.g2bx [-n 32] [--prefill 256]
gguf2bin2 ppl m.g2bx -f texto.txt             # harness de calidad (perplexity)
gguf2bin2 vkinfo                              # sonda Vulkan
```

Sampling: quickselect top-k O(n) + Gumbel-max + xorshift64\* reproducible con `--seed`.

### 🎮 Dual band CPU+GPU (`--gpu`)

El head GEMV (vocab×dim, la capa más pesada) se reparte entre CPU y GPU con calibración automática:

```
[gpu] worker listo
[gpu] dual band: cpu=[0..44855) gpu=[44855..65536)  tc=6.1ms tg=13.1ms
```

- **Worker Vulkan en proceso hijo** — si el driver crashea o cuelga, el runtime cae a CPU-only sin interrumpir nada.
- **Bypass del loader Vulkan**: carga el ICD directamente desde DriverStore (útil en sistemas con el registro `Khronos\Vulkan\Drivers` roto).
- **Split óptimo automático**: `gpu = vocab·tc/(tc+tg)` medido en el primer token; si la GPU es >4× más lenta que la CPU, se apaga sola.
- Soporta heads **Q4_0** y **Q4_0S**. Salida bit-idéntica al camino CPU (greedy).

> ⚠️ En hardware donde la iGPU comparte el bus de RAM con la CPU (HD 520 + DDR3L) no hay ganancia neta — la calibración lo detecta y se desactiva sola. El beneficio real llega con una dGPU con VRAM propia.

## 📐 Calidad

Perplexity (corpus interno, SmolLM2-135M-Instruct): uniforme-Q4_0 = 73.7 *(roto)*,
Q4_K_M nativo = **48.7**, Q6_K = 48.0, Q8_0 = 48.0. Los K-quants nativos conservan
la calidad del Q8_0 dentro del 1.4 % y corren más rápido que el uniforme-Q4_0.

Validación numérica incluida: `make kvtest` (KV F32 vs Q8), `tools/prefilltest`
(prefill batcheado bit-exacto), `tools/qkcheck` (46/46 kernels K-quant).

<details>
<summary><b>📦 Formato G2BX y tipos soportados</b></summary>

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B-aligned | tokenizer
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

| Tipo | Carga | Matmul fusionado |
|------|------|--------------|
| F32 / F16 | sí (pack→Q4_0 si es peso) | camino F32 / dequant |
| Q4_0 / Q4_1 / Q5_0 | sí | AVX2 fusionado |
| Q8_0 | sí | AVX2 fusionado |
| Q4_0S (propio, escala fp16 compartida /256) | sí | AVX2 batched |
| Q2_K … Q6_K | sí (dequant) | AVX2 fusionado (Q4_K/Q6_K), escalar resto |
| IQ* / Q8_K | parcial | — |

</details>

<details>
<summary><b>🗂 Estructura del proyecto</b></summary>

```
include/g2b.h      API común
src/l1_gguf.c      parser GGUF (mmap)
src/l2_codec.c     dequant + matmul fusionados (AVX2)
src/l3_math.c      rmsnorm, rope, silu, softmax AVX2
src/l4_gbin.c      packer G2BX (read_cfg genérico por arquitectura)
src/l5_model.c     load + forward + KV Q8 + presupuesto RAM + atención GQA-major
src/l6_token.c     tokenizer BPE
src/main.c         CLI (sampling quickselect/Gumbel, chat con compactación de contexto)
src/l7_vulkan.c    backend GPU dual band (worker en proceso hijo, a prueba de crashes)
shaders/           compute shaders (GEMV Q4_0/Q4_0S)
tools/             kvtest · prefilltest · exptest · qkcheck · mmbench · dump_gguf.py
docs/RESEARCH.md   notas de investigación y roadmap de rendimiento
```

</details>

<details>
<summary><b>📜 Historial de cambios</b></summary>

#### v4.5
- Kernel batched (prefill) con acumulación diferida: mismo trato que el kernel de decode. Prefill Qwen2.5-3B 4.3 → 7.8 tok/s (+81 %), bit-exacto (`tools/prefilltest`). Prepara el terreno para la verificación especulativa.
- **Dual band CPU+GPU del head GEMV**: worker Vulkan en proceso hijo (a prueba de crashes), bypass del loader cargando el ICD directo del DriverStore, calibración automática del split con auto-apagado si la GPU no aporta. Heads Q4_0/Q4_0S, salida bit-idéntica.

#### v4.4
- **--drop N (ShortGPT)**: mide Block Influence por bloque durante una calibración rápida y omite los N menos influyentes. En LFM2.5-1.2B no compensa (BI mínimo 0.106).

#### v4.3
- Kernel Q4_0 de decode con acumulación diferida: un hsum por fila en vez de uno por bloque. Qwen2.5-3B 3.1 → 4.3 tok/s (2.9× vs v3.5); Qwen3-0.6B +10 %.
- Q5_0 end-to-end (kernel AVX2 fusionado con LUT de bits altos).
- Tabla de calidad medida (comando `ppl`).

#### v4.2
- Kernels AVX2 fusionados para Q4_K y Q6_K (`maddubs` + término de corrección m·Σx). Antes cualquier pack K-quant caía al fallback 2–5× más lento. Validado byte a byte con `tools/qkcheck`.
- Comando `ppl`; bench min-de-3 (el thermal throttling miente).

#### v4.1
- K-quants arreglados contra ggml oficial (`deq_q3_K/q4_K/q5_K` rotos desde v3.4: mitad del tensor sin escribir + interleave incorrecto).
- Prefill batcheado (B=8) bit-exacto; validación de geometría en carga; scratch persistente de activación Q8 (−210 malloc/free por token).

#### v4.0
- Prefill sin logits (solo el último token calcula vocab×dim): prefill 1.32×.
- Atención GQA-major: cada fila K/V se dequantiza una vez por grupo de heads.
- softmax/silu AVX2 con exp rápida (err rel < 2e-7); fix rmsnorm cola no múltiplo de 32.
- Sampling nuevo: quickselect top-k O(n), Gumbel-max, xorshift64\*, `--seed` reproducible.
- GGUF por mmap; compactación de contexto en chats largos.

#### v3.x
- Q4_0 AVX2 (2 bloques/iter, ILP), atención AVX2, dequant K-quant completo.
- KV Q8_0 (`--q8-kv`), contexto efectivo, presupuesto RAM (`--max-ram`), swap en disco.
- Fix RoPE LLaMA (paso −2.0/head_dim) y NEOX vs LLaMA: la raíz histórica del output corrupto.

</details>
