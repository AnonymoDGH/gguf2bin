# gguf2bin2

[English](README.md) | **Español**

**Runtime LLM en C99 para máquinas con poca RAM.** Empaqueta un GGUF una vez a G2BX (formato propio, indexado y con CRC) y ejecútalo con los pesos memory-mapped: la RAM que realmente pagas es **KV cache + activaciones + tokenizer**, no el archivo del modelo.

[![ci](https://github.com/AnonymoDGH/gguf2bin/actions/workflows/ci.yml/badge.svg)](https://github.com/AnonymoDGH/gguf2bin/actions/workflows/ci.yml)
![versión](https://img.shields.io/badge/versión-5.1.2-informational)
![C99](https://img.shields.io/badge/C99-portable-blue)
![AVX2](https://img.shields.io/badge/kernels-AVX2%20%2B%20FMA-orange)
![RAM](https://img.shields.io/badge/RAM%20mín-37%20MB-green)

> Un **modelo de 3B en 145 MB de RAM** y modelos grandes en una **máquina de 2 GB** (`--swap`: KV cache en disco, heap ≈ 37 MB).

- **Pesos por mmap** — page cache desalojable, no heap; perillas de RAM para la KV (`--q8-kv`, `-c`, `--max-ram`, `--swap`).
- **Kernels AVX2 fusionados** para Q4_0, Q8_0, Q5_0, Q4_K, Q6_K, Q3_K, IQ1_S y el Q4_0S propio; prefill batcheado (Q4_0/Q4_0S leen cada fila de pesos una vez cada 4 tokens).
- **Arquitecturas**: Llama, Qwen2/2.5, Qwen3, LFM2/LFM2.5 (híbrido con conv), Qwen3.5 híbrido denso (experimental).
- **Cargador endurecido** — footer CRC y cada slot comprobado contra la geometría del modelo: un `.g2bx` corrupto u hostil se rechaza, no se ejecuta.
- **App Android** (`android/`) y una **API en C** pequeña (`include/gguf2bin.h`).

---

## 🚀 Pruébalo en un minuto

Hay modelos listos en el [release `models`](https://github.com/AnonymoDGH/gguf2bin/releases/tag/models):

| Archivo | Tamaño | Modelo |
|---|---|---|
| [`Qwen3-0.6B-q4.g2bx`](https://github.com/AnonymoDGH/gguf2bin/releases/download/models/Qwen3-0.6B-q4.g2bx) | 319 MB | Qwen3-0.6B · Q4_0 |
| [`lfm25-1.2b-q4s.g2bx`](https://github.com/AnonymoDGH/gguf2bin/releases/download/models/lfm25-1.2b-q4s.g2bx) | 569 MB | LFM2.5-1.2B · Q4_0S |

```bash
make                                   # Linux: gcc + OpenMP + AVX2
curl -LO https://github.com/AnonymoDGH/gguf2bin/releases/download/models/Qwen3-0.6B-q4.g2bx
./gguf2bin2 chat Qwen3-0.6B-q4.g2bx --fast
./gguf2bin2 run  Qwen3-0.6B-q4.g2bx "The capital of France is" --bos -n 20 -t 0
```

Tu propio modelo: `./gguf2bin2 pack modelo.gguf modelo.g2bx --q4` (también puedes pasar el `.gguf` directamente: se empaqueta una vez a `modelo.gguf.g2bx` y se reutiliza).

## 🧩 Arquitecturas soportadas

| Familia | `general.architecture` del GGUF | Notas | Comprobado con |
|---|---|---|---|
| Llama (Llama 3.x, SmolLM2) | `llama` | RoPE intercalado | Llama-3.2-1B, SmolLM2-135M |
| Qwen2 / Qwen2.5 | `qwen2` | sesgos Q/K/V | Qwen2.5-3B |
| Qwen3 | `qwen3` | QK-norm, RoPE NEOX | Qwen3-0.6B (modelo del release) |
| LFM2 / LFM2.5 | `lfm2` | híbrido short-conv + atención | LFM2.5-1.2B (modelo del release) |
| Qwen3.5 (denso) | `qwen35` | gated delta-net + atención completa cada N capas, decode secuencial | **experimental**: sin prueba con modelo real desde los fixes de v5.1.1 |

No soportado: MoE (`qwen2moe`, `bailingmoe*`…) ni torres de visión. Plantillas de chat: ChatML (`<|im_start|>`) y Llama 3 (`<|eot_id|>`).

## 🔧 Compilar

| Plataforma | Comando |
|---|---|
| Linux (gcc) | `make` · `make test` |
| Windows (MSYS2 MINGW64) | `pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-libgomp mingw-w64-x86_64-make` → `mingw32-make` |
| CMake (cualquiera) | `cmake -S . -B build && cmake --build build && (cd build && ctest)` |
| Sin AVX2 (x86 antiguo, ARM…) | `make CFLAGS="-O2 -std=c99 -fopenmp -Iinclude -Isrc"` — caminos escalares, mismos resultados, más lento |
| APK Android | `android/build_apk.ps1` (Windows, NDK 28, build-tools 35, JDK) → `gguf2bin.apk` |

Las flags por defecto asumen una CPU con AVX2 + FMA + F16C (Haswell / Zen 1 o posterior). `--threads N` elige los hilos de OpenMP; los núcleos físicos son el punto óptimo.

## ⚡ Velocidad

Intel i5-6200U · 2C/4T · DDR3L (~9.4 GB/s de techo de bus), `bench -n 32` (mín. de 3), `--fast`:

| Modelo | Pesos (mmap) | RAM runtime | decode tok/s | prefill tok/s |
|---|---|---|---|---|
| **Qwen2.5-3B** Q4_0 | 1992 MB | 145 MB | **4.3** | 7.8 |
| **Qwen3-0.6B** Q4_0 | 319 MB | 511 MB | 25.0 | **47.9** |
| **LFM2.5-1.2B** Q4_0S | 567 MB | 631 MB | **15.7** | 17.9 |
| Llama-3.2-1B F16 | 804 MB | 644 MB | 13.8 | — |
| SmolLM2-135M Q4_0 | 72 MB | 40 MB | **59.5** | — |

Medido el 2026-08-31. A ~25 tok/s el decode va clavado al bus de memoria (~9 GB/s); el prefill toca el techo de cómputo (~27 GMAC/s). Mídelo en tu máquina con `gguf2bin2 bench m.g2bx [--prefill 256] [--json]`.

## 🧠 Perillas de RAM

| Perilla | Efecto |
|---|---|
| `--q8-kv` | KV cache F32 → Q8_0: **~3.8× menos RAM**; +0.6–0.8 % de ppl en Qwen3-0.6B / LFM2.5-1.2B |
| `-c N` / `--ctx N` | Dimensiona el contexto a *tu* sesión, no al máximo del modelo (32k–262k) |
| `--max-ram MB` | Automático: primero KV Q8, luego parte el contexto a la mitad hasta caber |
| `--swap [RUTA]` | KV cache respaldada en archivo → **heap de 37 MB** incluso con modelos grandes (por defecto: archivo por proceso en el directorio temporal; en Windows `D:\` si existe) |

| ¿Cuenta contra un presupuesto de 2 GB? | | Solución |
|---|:-:|---|
| Pesos (page cache del mmap) | ❌ desalojable | — |
| KV cache | ✅ | `--q8-kv`, `-c N`, `--swap` |
| Buffers / activaciones | ✅ pequeño | — |
| Tokenizer (vocab ~150–250k) | ✅ ~30–60 MB | — |

## 🎛 Referencia de la CLI

```text
gguf2bin2 pack   <modelo.gguf> <salida.g2bx> [--q4 | --q4s] [--prune F [--calib texto.txt]]
gguf2bin2 info   <modelo.g2bx|modelo.gguf>     geometría, tipos, RAM estimada
gguf2bin2 verify <modelo.g2bx>                 header + slots + geometría + CRC, sin cargar pesos
gguf2bin2 run    <modelo> [texto] [opts]       continuación de texto
gguf2bin2 chat   <modelo> [opts]               chat interactivo (escribe 'exit')
gguf2bin2 bench  <modelo> [-n 32] [--prefill N] [--json]
gguf2bin2 ppl    <modelo> [-f archivo|-] [-n max_tokens]
gguf2bin2 synth  <salida.g2bx>                 modelo sintético diminuto para tests
gguf2bin2 vkinfo                               sonda Vulkan
```

| Opción | Comandos | Por defecto |
|---|---|---|
| `-n N` | run / chat / bench / ppl | 64 / 256 / 32 / 4096 |
| `-t TEMP` (0 = greedy) | run / chat | 0.7 |
| `--top-k K` · `--top-p P` | run / chat | 40 · 0.9 |
| `--repeat-penalty R` | run / chat | 1.1 / 1.05 |
| `--seed S` | run / chat | según la hora |
| `--bos` · `--tokens a,b,c` | run | desactivado |
| `--system TXT` · `--no-system` · `--think` / `--no-think` | chat | "You are a helpful assistant." · no-think |
| `--cyber adaptador.lora` | run / ppl | adaptador LoRA (v1/v2) |
| `-c N` · `--q8-kv` · `--f32-kv` · `--max-ram MB` · `--swap [RUTA]` | run / chat / bench / ppl | ctx del modelo · auto |
| `--fast` · `--threads N` · `--drop N` · `--gpu` | run / chat / bench | — |

`--q4` convierte todos los pesos a Q4_0 (la mitad de bytes); `--q4s` a Q4_0S (una escala fp16 por 256, ~10 % menos bytes que Q4_0) donde la longitud de fila lo permite. `--drop N` omite los N bloques menos influyentes (ShortGPT). El sampling es top-k por quickselect O(n) + Gumbel-max, reproducible con `--seed`.

<details>
<summary><b>🎮 Head dual band CPU+GPU (<code>--gpu</code>, Windows)</b></summary>

El GEMV del head (vocab×dim, la capa más pesada) se reparte entre CPU y GPU con calibración automática:

```
[gpu] worker ready
[gpu] dual band: cpu=[0..44855) gpu=[44855..65536)  tc=6.1ms tg=13.1ms
```

- Vulkan corre en un **proceso hijo**: si el driver crashea o se cuelga, la generación sigue en CPU.
- Carga el ICD directamente desde DriverStore (funciona aunque el registro `Khronos\Vulkan\Drivers` esté roto).
- Split `gpu = vocab·tc/(tc+tg)` medido en el primer token; se apaga solo si la GPU es >4× más lenta.
- Heads Q4_0 y Q4_0S, bit-idéntico al camino CPU (greedy).

En iGPUs que comparten el bus de RAM con la CPU (HD 520 + DDR3L) no hay ganancia neta y la calibración lo desactiva; el beneficio llega con una dGPU con VRAM propia. En otras plataformas `--gpu` no hace nada.

</details>

## 🧑‍💻 API en C

`include/gguf2bin.h` es el único header que necesita una aplicación (sesiones, chat, generación, tokenizer, pack, ppl, bench):

```c
#include "gguf2bin.h"
#include <stdio.h>

static void on_token(const char *piece, void *ud){ (void)ud; fputs(piece, stdout); fflush(stdout); }

int main(void){
  g2b_config cfg = {0};
  cfg.ctx = 2048; cfg.q8_kv = -1;                 /* -1 = automático */
  g2b_session *s = NULL;
  g2b_error e = g2b_open("Qwen3-0.6B-q4.g2bx", &cfg, &s);
  if(e){ fprintf(stderr, "open: %s\n", g2b_strerror(e)); return 1; }

  g2b_chat_begin(s, "You are a helpful assistant.", 1 /* no_think */);
  g2b_gen_params p = {0};
  p.on_token = on_token;
  p.temp = 0.7f; p.top_k = 40; p.top_p = 0.9f; p.repeat_penalty = 1.05f;
  p.max_tokens = 128; p.seed = 42;
  g2b_chat_turn(s, "What is the capital of France?", &p);   /* p. ej. "The capital of France is Paris." */
  g2b_close(s);
  return 0;
}
```

```bash
cmake -S . -B build -DG2B_BUILD_TESTS=OFF && cmake --build build
gcc -O2 -std=c99 -Iinclude app.c build/libg2bcore.a -fopenmp -lm -ldl -o app
```

Una sesión activa por proceso (el scratch de cómputo es global).

## 📱 Android

`android/` contiene una app mínima (Java + JNI, sin Gradle): descarga un `.g2bx` pegando su URL y chatea con streaming, top-k y penalización de repetición. Como los archivos vienen de URLs arbitrarias, el cargador valida cada slot antes de ejecutar nada. Se compila con `android/build_apk.ps1` (las rutas del SDK/NDK están al principio del script).

## 📐 Calidad

Perplexity, pruebas locales cortas (no es un benchmark estándar). Filas Qwen/LFM2: texto del README + spec, 1024 tokens; fila SmolLM2: otro corpus interno.

| Modelo / formato | ppl | vs base |
|---|---:|---|
| Qwen3-0.6B Q4_0 | 25.4 | — |
| Qwen3-0.6B Q8_0 | 20.8 | −18 % |
| LFM2.5-1.2B q4max | 28.1 | — |
| LFM2.5-1.2B Q4_0S | 34.9 | +24 % |
| SmolLM2-135M Q4_0 (uniforme) / Q4_K_M / Q6_K / Q8_0 | 73.7 / 48.7 / 48.0 / 48.0 | K-quants nativos ≈ Q8_0 |

Tipos propios (IDs 0x80+, ningún otro runtime los lee): **Q4_0S** ahorra ~10 % de bytes frente a Q4_0 a cambio de algo de calidad (+24 % de ppl en LFM2.5, arriba). **Q4_0S_PSY** tuvo los kernels rotos hasta v5.1.1 (su ppl en Llama-3.2-1B de 2.380.556 frente a 82,3 de base medía el bug, no el formato) y **Q4_VVC** es una cuantización de 3 bits uniforme con una escala por 256 (Llama-3.2-1B: 1.177 frente a 84,0 de base); ambos siguen desaconsejados hasta volver a medirlos.

## 🧪 Pruebas

```bash
make test                     # modelo sintético + info/run/bench + apitest + selftest
make kvtest q4bcheck iq1check q3kcheck prefilltest
```

- `selftest`: sampler, mmap, E/S G2BX, CRC, dispatch de kernels, round-trip del tokenizer, kernels vs dequant de referencia, rechazo de slots truncados.
- `prefilltest`: el prefill batcheado es bit-exacto con el decode secuencial; `kvtest`: KV F32 vs Q8.
- `q4bcheck` / `iq1check` / `q3kcheck` / `tools/qkcheck.c`: kernels fusionados vs matemática de referencia.
- El CI (Linux, MinGW, CMake, ASan+UBSan, libFuzzer sobre los parsers GGUF/G2BX/tokenizer) corre en cada push.

## ⚠️ Limitaciones conocidas

- El tokenizer es BPE byte-level **sin la regex de pre-tokenización**, así que los ids pueden diferir de llama.cpp / Hugging Face en algunas entradas (el texto hace round-trip exacto).
- Una sesión por proceso; el dual band Vulkan solo existe en Windows.
- El packer omite tensores BF16, Q5_1, Q8_1 y F64. Q8_K no tiene dequant (se lee como ceros; no aparece en archivos de modelo normales). Los tipos sin kernel fusionado (Q2_K, Q5_K, IQ2/3/4…) van por un camino de dequant más lento.
- En chats de varios turnos con Qwen3 el historial conserva el bloque `<think>` vacío de los turnos pasados (la plantilla oficial lo quita).

<details>
<summary><b>📦 Formato G2BX</b></summary>

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B-aligned | tokenizer | [v3: crc32 + "G2BX"]
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

Los tensores se indexan por rol (búsqueda O(1)), todo entero es little-endian explícito y v3 añade un footer CRC32 que se verifica al cargar. Los lectores aceptan v1/v2/v3. Spec completa: [`docs/G2BX_SPEC.md`](docs/G2BX_SPEC.md).

| Tipo | Carga | Matmul |
|---|---|---|
| F32 / F16 | sí (se empaqueta a Q4_0 si es peso) | camino F32 / dequant |
| Q4_0 / Q8_0 / Q5_0 | sí | AVX2 fusionado |
| Q4_0S (propio: escala fp16 por 256) | sí | AVX2 fusionado + batched |
| Q4_K / Q6_K / Q3_K | sí | AVX2 fusionado |
| Q2_K, Q5_K, Q4_1, IQ2/IQ3/IQ4 | sí | fallback por dequant |
| IQ1_S | sí | AVX2 fusionado |

</details>

<details>
<summary><b>🗂 Estructura del proyecto</b></summary>

```
include/gguf2bin.h   API pública (sesiones)       src/internal/   internals compartidos
src/model.c          carga G2BX, validación, RAM  src/kv.c        KV F32/Q8, swap a disco, buffers
src/forward_*.c      denso / lfm2 / qwen35 / prefill batcheado
src/l1_gguf.c        parser GGUF (mmap)           src/l4_gbin.c   packer G2BX (+ poda)
src/l2_codec.c       dequant + matmul fusionado    src/l3_math.c   norms, RoPE, softmax
src/l6_token.c       tokenizer BPE                 src/l7_vulkan.c GPU dual band
src/g2b_api.c        implementación de la API      src/g2bx_io.c   reader/writer/CRC/verify G2BX
src/main.c           CLI                           android/        app + JNI
tools/               harnesses de test + fuzz/     docs/           spec, notas, roadmap de rendimiento
```

</details>

## 📜 Historial de cambios

Ver [CHANGELOG.es.md](CHANGELOG.es.md). Último: **v5.1.2** — validación con modelos reales de los fixes de v5.1.1, prompts de `run` sin espacio final (las continuaciones ya responden "Paris"), JNI de Android seguro con emojis y script del APK funcionando.

El código de referencia de llama.cpp en `third_party/` tiene su propia licencia ([`docs/LICENSE.llama_cpp`](docs/LICENSE.llama_cpp)).
