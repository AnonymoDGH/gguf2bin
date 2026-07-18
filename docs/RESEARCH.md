# Investigación: super-comprimir Qwen3 y runtime propio — FIXED v3.3

## Fixes v3.3

1. **Sampler**: repeat_penalty se aplicaba dos veces con temp>0; ring buffer de `recent` leía mal tras 128 tokens. Ahora: una aplicación, dedup de ids, snapshot de los últimos N.
2. **RoPE por arch**: Llama → interleaved; Qwen2/3 → NEOX.
3. **Contexto**: `pos %= seq` corrompía atención. Ahora aborta limpio al superar `seq_len`.
4. **mmap**: pesos mapeados si el SO lo permite; fallback malloc+fread.
5. **matmul_q OpenMP**: buffer `row` compartido era data-race; scratch privado por hilo.
6. **Hash tokenizer**: rehash al 70% de carga.
7. **Pack**: omite y reporta quant types no soportados (Q1_0, K-quants).

## Fixes v3.1 / v3.2

1. **G2BX loader**: `aligned_end = ALIGN64(max_end)` para no perder tokenizer.
2. **Pack**: preserva Q8_0/Q4_0; solo F32/F16 → Q4_0.
3. **CLI**: `run` sin BOS por defecto; top-k/top-p/repeat-penalty.
4. **Chat**: plantilla oficial Qwen3 (`thinking` / `no-think`).

## Validación vs llama.cpp (b10064)

GGUF Qwen3-0.6B-Q8_0 Instruct (639MB, 310 tensores, dim 1024 L28)

- Completion: `The capital of France is` → `Paris. The capital of Italy is Rome...`
- Chat no-think: `What is the capital of France?` → `The capital of France is **Paris**.`

## Por qué Qwen3 no es Llama

- head_dim 128 != dim/heads 64
- rope_theta 1e6
- QK-Norm por capa
- vocab 151936
- thinking tokens 151667/151668
- RoPE style NEOX
