# Roadmap de rendimiento LFM2.5 (auditoría post-96%-del-bus)

Estado: decode 14.6 tok/s = 9.6 GB/s ≈ 96% del techo DDR3L. Solo cortar bytes mueve el headline.

## 1. Q4_0s — escala fp16 compartida por superbloque de 256 (+8-10% tok/s)
Hoy: 144B/256elems. Con 1 escala por superbloque: 130B (-9.7% bytes global).
- Pack: amax por 256 -> escala fp16 + 8x16B nibbles; nuevo tipo T_Q4_0S.
- Kernel: clonar matmul_q4_0 con escala scl_f32[b>>3] preconvertida.
- Validar ppl/qkcheck por fases: FFN+attn primero, head/embd despues.
- Variante E8M0 (escala pow2, 129B) si el delta de ppl molesta.

## 2. Prefill batcheado para LFM2 (TTFT x3-6)
model_prefill retorna 1 para ARCH_LFM2. Lo secuencial real es solo la conv
k=3 (trivial): batchear embd+rmsnorm+in_proj/out_proj+FFN con kernels _b,
serializando unicamente B*x -> conv(state) -> C*y entre proyecciones.

## 3. Cirugia menor (+1-2%)
a) Cachear norms/conv_w como F32 al cargar (hoy se dequantizan por token).
b) kr/vr persistentes por omp_get_thread_num (hoy malloc/free por capa-token).
c) Vectorizar lroundf en q4_quant_act (cvttps + truco ±0.5).
d) Preconvertir escalas fp16 a F32 scratch por llamada.

## Descartados (argumentados)
- Fold RMS-norm en W: ahorra KBs contra 656MB = ~0.1%.
- Interleave gate/up: ya fusionables (contiguos sin padding).
- Large pages: CreateFileMapping no soporta SEC_LARGE_PAGES para ficheros;
  KV/buffers son pequenos. <=1-2% por esfuerzo alto.
- Sparsidad dinamica SiLU: no hay ceros exactos; umbralizar = aproximacion
  con mispredicts en B=1. El beneficio ya lo captura --prune offline.

Medicion de afinidad: 2 fisicos=13.4 vs 4 HT=14.0 -> mantener 4 hilos.
