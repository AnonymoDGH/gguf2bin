# G2BX — especificación del formato (v3.1 del documento; formato v1/v2/v3)

`G2BX` es el formato propio de gguf2bin2: tensores indexados por rol
(en vez de por nombre) para carga O(1) en runtime. Todo entero en disco es
**little-endian explícito, campo a campo** (nunca `fwrite` de structs: el
layout C no es contrato). Los readers aceptan v1, v2 y v3.

## 1. Layout

```
offset  contenido
0       magic "G2BX" (4 B)
4       ver:u16           (2=v2, 3=v3 con footer; v1 también existe en archivos viejos)
6       arch:u8           (0=llama 1=qwen2 2=qwen3 3=lfm2 4=qwen35)
7       flags:u8          (bit0=F_TIE_EMBD bit1=F_QK_NORM bit2=F_MMAP; bit3=F_KV_Q8 nunca en disco)
8       ModelCfg          (v1: 40 B; v2+: 68 B — §2)
8+C     n_slots:u32
12+C    Slot[n_slots]     (16 B cada uno — §3)
H       blob de pesos     (offsets relativos a H; huecos a cero; total ALIGN64)
H+B     tokenizer         (sección BPE; puede faltar en modelos draft)
H+B+T   [solo v3] footer  (8 B — §5)
```

`H` (data_start) se calcula leyendo el header (no es fijo: depende de
`n_slots`). `B` (blob) = ALIGN64(max(off+nbytes)).

## 2. ModelCfg

| orden | campo | tipo | v1 |
|---|---|---|---|
| 1-8 | dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab, seq_len, head_dim | i32 | ✅ |
| 9-10 | eps, rope_theta | f32 | ✅ (= 40 B) |
| 11-17 | fa_interval, ssm_d_state, ssm_n_group, ssm_dt_rank, ssm_inner, ssm_d_conv, n_rot | i32 | ❌ (cero en v1) |

## 3. Slot (16 B)

```
role:u8 | layer:u16 | type:u8 | nbytes:u32 | off:u64
```

- `role`: enum R_* (0=tok_embd …; ver `g2b.h`). `layer=0xFFFF` = global (no por capa).
- `off`: relativo al inicio del blob. `nbytes`: u32 (límite documentado: 4 GB por tensor; el packer debe rechazar más — pendiente, §7).
- El blob puede tener huecos (padding a 64 B); el reader no los interpreta.

## 4. Tipos de tensor (`type:u8`)

| ID | Nombre | Origen |
|---|---|---|
| 0,1,2,3,6,7,8,9 | F32 F16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 | GGUF |
| 10-15 | Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K | GGUF |
| 16-23 | IQ2_XXS IQ2_XS IQ3_XXS IQ1_S IQ4_NL IQ3_S IQ2_S IQ4_XS | GGUF |
| 24,25,26,27 | I8 I16 I32 I64 | GGUF (sin kernels: el packer los **rechaza**) |
| 28,29,30 | F64 IQ1_M BF16 | GGUF |
| **0x80,0x81,0x82** | **Q4_0S Q4_0S_PSY Q4_VVC** | **propios (desde v3)** |
| 25,26,27 en v1/v2 | legacy Q4_0S/PSY/VVC | normalizados a 0x80+ al leer |

Regla: en v3 los IDs 25/26/27 (y cualquier ID desconocido) en un slot son
**inválidos** (`verify` los rechaza). En v1/v2 se aceptan 25/26/27 con el
significado legacy.

## 5. Footer v3 (8 B al final del archivo)

```
[crc32 LE (4 B) de todos los bytes previos][magic "G2BX" (4 B)]
```

- CRC32-IEEE (tabla propia, sin dependencias), calculado en streaming 64 KB.
- El loader lo verifica al abrir (una pasada secuencial que además deja la
  page cache caliente para el mmap posterior).
- Truncado → falta el magic → error. Corrupción → CRC mismatch → error.

## 6. Sección tokenizer

```
nv:u32 nm:u32 bos:i32 eos:i32 unk:i32
nv × (len:u32 + bytes)   # vocabulario
nm × (len:u32 + bytes)   # merges "a b"
```

Presente si el archivo se extiende más allá del blob (+20 B mínimo).
El peek es posicional (lee en `blob_end`), así que el footer v3 no interfiere.

## 7. Matriz de compatibilidad

| Lector | v1 | v2 | v3 |
|---|---|---|---|
| v5.0+ | ✅ (cfg 40 B, resto cero) | ✅ (+normaliza tipos legacy) | ✅ (+verifica CRC) |
| v4.x | ✅ | ✅ | ❌ (rechaza: versión no soportada) |

El packer/synth desde v5.0 solo escriben v3.

## 8. Reservado / futuro

- `chat_template`: campo string tras los slots (antes del blob) reservado;
  sin motor de plantillas no se almacena nada (Fase 8).
- Validación packer de tensor >4 GB (hoy truncaría el u32 en silencio).
- `g2bx diff a b` (tooling futuro; `verify` ya existe).
