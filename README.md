# gguf2bin2 — G2BX FIXED v3.4 (2 GB RAM)

Runtime propio en C99: **GGUF → G2BX → inferencia** (Qwen3 / Qwen2 / Llama / arquitecturas custom).

**Validado:** `The capital of France is` → `10,000 km²…` · Chat `What is the capital of France?` → `**Paris**`
(núcleo Qwen3-0.6B-Q8_0; probado también contra llama.cpp b10064 en v3.3).

## Novedades v3.4 — correr modelos pesados en 2 GB de RAM

Los pesos se cargan con **mmap** (no ocupan heap; el SO evicta páginas), así que el
límite real de RAM es la **KV cache + buffers + tokenizer**. Tres palancas:

### 1. KV cache cuantizada Q8_0 (`--q8-kv`)

La caché K/V pasa de F32 (4 B/elem) a Q8_0 (34 B/32 elems): **~3.8× menos RAM**.
Verificado numéricamente (herramienta `make kvtest`): rmse≈0.27 en logits,
**94% de coincidencia de argmax** frente a KV F32 en pasos greedy → calidad casi idéntica.

```
# Qwen3-0.6B, ctx=8192
ram: pesos=604 MB (mmap) | runtime=1819 MB = KV(F32,ctx=8192)+buffers+tokenizer   # sin opciones
ram: pesos=604 MB (mmap) | runtime=503 MB  = KV(Q8_0,ctx=8192)+buffers+tokenizer  # --q8-kv
```

### 2. Contexto efectivo ajustable (`-c N` / `--ctx N`)

La KV cache se dimensiona con el contexto **real** de la sesión, no el «context_length»
del modelo (que puede ser 262144). Menos contexto = menos RAM al instante.

### 3. Presupuesto automático (`--max-ram MB`)

```
$ gguf2bin2 run qwen.g2bx "Hola" --max-ram 2048
```
Ajusta solo: primero activa **KV Q8_0**, luego reduce el contexto hasta que el
footprint de runtime (reportado en `ram:`) quepa en el presupuesto. Ideal para
equipos de 2 GB.

### 4. Usar el disco como RAM — `--swap [PATH]`

Respala la **KV cache en un archivo** (por defecto `D:\gguf2bin2_kv.swap`, si D:
existe). Las páginas de KV que no estén calientes **se vuelcan a disco** y dejan
libre la RAM física. Medido en Qwen3-0.6B (ctx 8192, 4 hilos, tu PC):

| Config | Heap privado | RAM física (WorkingSet) | tok/s |
|--------|-------------|------------------------|-------|
| KV F32 en RAM (ctx 8192) | ~1.9 GB | ~4 GB | — |
| Auto-Q8 (KV en RAM) | **513 MB** | 672 MB | 10.1 |
| **`--swap` D: (KV en disco)** | **37 MB** | 670 MB | 11.1 |

```
$ gguf2bin2 chat qwen.g2bx --no-think --swap --threads 4
ram: pesos=604 MB (mmap) | runtime=503 MB = KV(Q8_0,ctx=8192) (file-backed) + buffers + tokenizer
```

### Qué cuenta contra los 2 GB (y qué no)

| Componente | ¿Cuenta? | Cómo reducirlo |
|-----------|----------|----------------|
| Pesos (mmap) | **No** (page cache evictable) | — |
| KV cache | **Sí** | `--q8-kv`, `-c N` |
| Buffers/activaciones | Sí (pequeños) | — |
| Tokenizer (vocab Qwen ~250k) | Sí (~30-60 MB) | — |
| Logits (vocab×4B) | Sí (~1 MB × 250k) | — |

## Cambios v3.4 (detalle)

- **KV cache Q8_0** (`F_KV_Q8`): `--q8-kv` / activación automática con `--max-ram` (y auto si la KV F32 pasa de 1 GB).
- **Contexto efectivo** (`model_set_ctx`): la KV cache deja de depender de `seq_len`.
- **Presupuesto de RAM** (`model_auto_budget` + `model_ram_report`): auto-Q8 + baja ctx.
- **Swap KV** (`--swap [PATH]` → por defecto `D:`): KV respaldada en archivo; páginas frías a disco, heap → decenas de MB (`--f32-kv` fuerza KV en RAM).
- **`read_cfg` genérico**: acepta cualquier `general.architecture` (p.ej. `dflash`),
  usando el nombre real como prefijo de metadatos con fallbacks qwen/llama.
- **Bugfix `vocab=0`**: `tokenizer.ggml.tokens` es un array; ahora se lee su longitud
  (`gguf_meta_arr_len`) en vez de `gguf_meta_i64`.
- **Aviso de modelos draft/decode-only**: si no hay `token_embd.weight` (p.ej. GGUFs de
  speculative decoding como `dflash`/`dspark`), pack avisa que no pueden generar
  texto standalone (dependen de un modelo teacher), en vez de fallar en silencio.
- **Atención optimizada**: la fila K/V se dequantiza una vez por posición (también en
  el path F32), no una vez por head → menos lecturas de caché.

## Root cause histórico — RoPE NEOX vs LLaMA

Qwen3 usa **NEOX RoPE** (half-split). Con RoPE LLaMA interleaved sale basura UTF-8.

```c
// LLaMA: pares (0,1),(2,3)...
// NEOX:  pares (0,D/2),(1,D/2+1)...
```

## Compilar

```bash
# Linux / MinGW con OpenMP + AVX2 (recomendado)
make

# Smoke test
make test

# Verificación numérica KV F32 vs Q8 (genera un modelo sintético)
make kvtest

# Sin OpenMP (fallback)
gcc -O2 -std=c99 -Iinclude -o gguf2bin2.exe \
  src/l1_gguf.c src/l2_codec.c src/l3_math.c src/l4_gbin.c \
  src/l5_model.c src/l6_token.c src/main.c -lm
```

> **Importante (velocidad):** compila con `-mavx2 -mfma -fopenmp` (que es el
> `make` por defecto). Sin AVX2 los matmul caen a kernels escalares y la velocidad
> baja ~8×.

## Rendimiento (Qwen3-0.6B-Q8_0, CPU i5 4 núcleos, mmap)

| Build | tok/s |
|-------|-------|
| Escalar, 1 hilo | 0.7 |
| AVX2, 1 hilo | 4.9 |
| AVX2, 4 hilos (Q8) | 10.1 |
| **AVX2, 4 hilos (Q4)** | **11.3** |

El decode está limitado por el kernel de matmul + ancho de banda de memoria. El
kernel **Q4_0 optimizado (2 bloques/iteración)** iguala/supera a Q8 a la vez que
los pesos usan la **mitad de RAM** (335 MB vs 633 MB). `pack --q4` da el mejor
tok/s y deja espacio para modelos más grandes en los mismos 2 GB.

## Uso

```bash
# Empaquetar (+ --q4 para bajar Q8_0->Q4_0: pesos a la mitad)
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen.g2bx
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen-q4.g2bx --q4

# Info
./gguf2bin2 info qwen.g2bx

# Completar (greedy) — y ver el footprint de RAM
./gguf2bin2 run qwen.g2bx "The capital of France is" -n 20 -t 0

# Chat no-think, forzando KV Q8_0 y max 2 GB de RAM, 4 hilos
./gguf2bin2 chat qwen.g2bx --no-think -n 40 -t 0.7 --max-ram 2048 --threads 4

# Contexto corto a mano (menos RAM de KV cache)
./gguf2bin2 run qwen.g2bx "Hola" -c 1024 -n 20

# Presupuesto automático (elige Q8 + baja ctx solo si hace falta)
./gguf2bin2 run qwen.g2bx "Hola" --max-ram 1024
```

## Uso

```bash
# Empaquetar
./gguf2bin2 pack Qwen3-0.6B-Q8_0.gguf qwen.g2bx

# Info
./gguf2bin2 info qwen.g2bx

# Completar (greedy) — y ver el footprint de RAM
./gguf2bin2 run qwen.g2bx "The capital of France is" -n 20 -t 0

# Chat no-think, forzando KV Q8_0 y max 2 GB de RAM
./gguf2bin2 chat qwen.g2bx --no-think -n 40 -t 0.7 --max-ram 2048

# Contexto corto a mano (menos RAM de KV cache)
./gguf2bin2 run qwen.g2bx "Hola" -c 1024 -n 20

# Presupuesto automático (elige Q8 + baja ctx solo si hace falta)
./gguf2bin2 run qwen.g2bx "Hola" --max-ram 1024

# Rendimiento CPU: AVX2 + 4 hilos
./gguf2bin2 bench qwen.g2bx -n 32 --threads 4

# Empaquetar pesos Q8_0 como Q4_0: mitad de espacio, misma velocidad aproximada
./gguf2bin2 pack modelo.gguf modelo-q4.g2bx --q4
```

La compilación por defecto usa `-O3 -mavx2 -mfma -fopenmp`. En una CPU sin
AVX2 debe usarse la compilación manual sin esas flags. `--threads N` controla
los hilos OpenMP; el valor recomendado es el número de núcleos físicos.

## Formato G2BX

```
G2BX | ver:u16 | arch:u8 | flags:u8 | ModelCfg | n_slots:u32 | Slot[] | data[] 64B aligned | tokenizer
Slot: role:u8 layer:u16 type:u8 nbytes:u32 off:u64
```

## Tipos soportados

| Tipo | Load | Matmul fused |
|------|------|--------------|
| F32 / F16 | sí (pack→Q4_0 si es peso) | F32 path / dequant |
| Q4_0 / Q4_1 | sí | Q4_0 AVX2/scalar |
| Q8_0 | sí | Q8_0 AVX2/scalar |
| Q1 / Q2_K / Q3_K / Q4_K / … | **no** | — |

## Estructura

```
include/g2b.h
src/l1_gguf.c    # parser GGUF
src/l2_codec.c   # dequant + matmul
src/l3_math.c    # rmsnorm, rope, silu, f32_to_half
src/l4_gbin.c    # pack G2BX (read_cfg genérico por arquitectura)
src/l5_model.c   # load + forward + KV Q8 + presupuesto RAM
src/l6_token.c   # BPE tokenizer
src/main.c       # CLI
tools/kvtest.c   # verificación numérica KV F32 vs Q8
tools/dump_gguf.py
docs/RESEARCH.md
```
