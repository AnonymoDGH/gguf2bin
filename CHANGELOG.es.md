# Historial de cambios

[English](CHANGELOG.md) | **Español**

#### v5.1.2 — validación con modelos reales, build Android, README
- **CLI `run`**: el prompt se construía con un espacio final ("…is " → último token `Ġ`), lo que descarrilaba las continuaciones. Ahora las palabras se unen sin espacio final: con los modelos del release, `run m.g2bx "The capital of France is"` pasó de "3,000 km²…" / "which makes how many…" a "Paris" en Qwen3-0.6B y LFM2.5-1.2B.
- **JNI Android**: los prompts pasaban por `GetStringUTFChars` (Modified UTF-8), así que un emoji llegaba al tokenizer como par surrogate de 6 bytes y se convertía en 6 tokens basura; ahora se codifica desde UTF-16. El resultado final descarta un carácter incompleto al final (corte por `maxTokens`) en vez de terminar en U+FFFD.
- **`android/build_apk.ps1`**: seguía listando `src/l5_model.c` (partido en la Fase 3) y no incluía `src/internal`: el build nativo del APK no compilaba. Lista de fuentes e includes actualizados (enlace verificado con `--no-undefined`; JNI probado de punta a punta con `-Xcheck:jni` y los dos modelos del release).
- README reescrito (modelos listos, arquitecturas, build por plataforma, CLI completa, ejemplo de API en C, pruebas, limitaciones); el historial se mueve aquí.

#### v5.1.1 — correcciones de revisión
- **CI en verde** (rojo desde la Fase 3): `strdup`/`fseeko`/`ftello`/`clock_gettime` quedaban sin declarar con `-std=c99`; en Linux x86-64 el puntero truncado de `strdup` tumbaba `make test` y `ftello` truncaba offsets >2 GB. También: YAML del job fuzz, `fmemopen`, `copy` de MinGW bajo `sh` de MSYS2, includes de CMake, flags AVX2 en el job de sanitizers.
- **Tokenizer**: `u2b[289]` desbordaba y los bytes 0x7F–0xA0/0xAD se decodificaban mal (€, à, emojis).
- **Kernels Q4_0S_PSY** (decode y batched) leían los nibbles desplazados 2 bytes: la fila PSY de la tabla mide un kernel roto, no el formato.
- **qwen35**: `wo` de atención usaba `n=dim` en vez de `n_heads*head_dim`. El estado recurrente de **LFM2/qwen35** no se reiniciaba en `pos 0`.
- **LoRA** ignorado en el prefill batcheado; **pack --prune** corrompía `ffn_down` con F16/F32/Q8_0; tensores >4 GB rechazados.
- **Endurecimiento ante .g2bx no confiables** (la app Android descarga de cualquier URL): `nbytes` de cada slot validado contra la geometría, topes contra overflow i32, comprobaciones de EOF/offsets.
- **JNI Android**: use-after-free entre `freeModel` y `generate`; tokens emitidos como caracteres UTF-8 completos.
- **Chat LFM2**: ya no se inyecta el bloque `<think></think>` vacío (convención de Qwen3) que hacía que LFM2.5 empezara "corrigiéndose". **ppl**: las ventanas 2+ reabren con BOS como llama.cpp.

#### v5.0 — G2BX v3: formato con CRC + namespace de tipos propio
- **Footer CRC32**: todo `.g2bx` nuevo termina en `[crc32 de lo previo][magic]`; el loader lo verifica al abrir y rechaza truncados/corruptos con mensaje claro. Nuevo comando `verify` (header + slots + tipos + geometría + CRC sin cargar pesos).
- **Tipos internos en 0x80+**: Q4_0S/PSY/VVC dejan los IDs 25/26/27 (hoy I16/I32/I64 en ggml — colisión real). Archivos v1/v2 se normalizan al cargar.
- **Spec escrita**: `docs/G2BX_SPEC.md` (layout, LE campo-a-campo, matriz de compatibilidad). Serialización del header ahora explícita LE.
- Refactors amparados por harnesses: split l5, API pública, `os_mm/sampler/opts/g2bx_io`, CI + CMake.

#### v4.9 — IQ1_S + Q3_K fusionados (27B desatascado)
- **Dot entero IQ1_S** (`madd`+SAD, act Q8, 1 hsum/escala por 32): los 264 slots IQ1_S (~3.4 GB) iban por fallback escalar; el prototipo fusionado existía pero nunca se despachaba. Conectado en `matmul_q`/`matmul_q_b`, validado con nuevo `tools/iq1check` (vs matemática exacta-Q8: maxrel 5e-4).
- **Dot entero Q3_K** (valores −4..3, escalas por 16, bias −32): cubre el head de 248k (521 MB) + modelos Q3_K densos. Validado con nuevo `tools/q3kcheck` incl. barrido one-hot de 256 posiciones (cazó pre-ship un bug de medio vector `cvtepi8`: los 8 altos se perdían en silencio).
- Híbrido 27B (Qwen3.8, siempre decode secuencial): stock 132.7 s → **79.6 s (−40 %, 1.67×)** para prompt+2 tokens, page cache caliente, i5-6200U. El greedy difiere en argmax (aprox Q8 sobre pesos de 1.5-bit — ambas salidas son mojibake de IQ1_S); la matemática, acotada por los harnesses.

#### v4.8 — Prefill bloqueado (blocking G=4 por tokens)
- **Tráfico de pesos ÷4 en `matmul_q4_0_b` / `matmul_q4_0s_b`**: cada fila de pesos se desempaqueta una vez y se reusa para 4 tokens (antes: re-leída por token, 16× por batch). Prefill Qwen3-0.6B Q4_0 38.9 → **53.7 tok/s (+38 %)** y Qwen2.5-3B Q4_0 7.0 → **9.6 tok/s (+37 %)** (A/B intercalado en i5-6200U). Bit-exacto (`prefilltest` diff 0 incl. 3B GQA, `q4bcheck` 5/5, ppl idéntica 58.709). Decode intacto (3B: 5.4 = 5.4); salida del híbrido 27B IQ1_S byte-idéntica a stock.

#### v4.7 — Q4_0S_PSY (psicoacústico) + fallback TLS
- La mejora de calidad anunciada para PSY queda retirada. En la prueba local de 128 tokens: ppl 2380555.838 frente a 82.325 base. VVC: 1177.176 frente a 83.966 base (256 tokens). El soporte permanece, desaconsejado hasta aislar la causa. Son pruebas cortas, no resultados generales por formato. Ver la tabla de [README.es.md](README.es.md#-calidad) y `docs/ROADMAP_PERF.md`.

#### v4.6 — Swapeculative MV Triple Band
- Retirado en v5.1 junto a BVH por la degradación observada en las pruebas locales. Se archivan los comandos CYBER con métricas sintéticas y se conserva la carga de adaptadores `--cyber`. La eliminación de campos de `g2b_config` requiere recompilar clientes de la API con el nuevo header.

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
