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
# Hallazgo critico: CANAL UNICO
Win32_PhysicalMemory = ChannelB-DIMM0 8GB unicamente -> single channel.
Anadir SO-DIMM DDR3L-1600 en ChannelA => dual channel (~19-20 GB/s reales)
=> TODOS los modelos x2 velocidad instantanea sin tocar codigo.
(LFM2.5 q4s: 15.9 -> ~30 tok/s; Qwen2.5-3B: 4.3 -> ~8.5)

## Roadmap software restante (ordenado)
1. matmul_q4_0s_b (batched streamea B veces hoy - bug en dispatch l2:792)
2. Especulacion n-gram + verificacion batcheada: 1.3-1.85x (alpha=0.35-0.5)
   - LFM2: checkpoint conv_state antes del draft, rollback KV en rechazo
3. Shortlist head 2 niveles (top-4096 tokens = 85-92% emisiones): 8-11%
4. Large pages para pesos (MEM_LARGE_PAGES + SeLockMemoryPrivilege): 2-5%
5. INT8 activations en todos los kernels: habilita k=6-8 especulativo
# Backend GPU (R5 M330 / Vulkan) — diseño para próxima sesión
Objetivo: pesos residentes en los 2GB VRAM (~15 GB/s > canal unico RAM).
Fases: (1) loader Vulkan minimo: buffers de pesos por slot en device-local
heap; (2) compute shader GEMV Q4_0S (1 workgroup por fila, nibbles via
unpack); (3) pipeline por bloque LFM2 (conv/att), sincronizando x en VRAM;
(4) transferencia inicial una vez al arrancar (595MB PCIe ~2-4s, solo 1 vez);
(5) fallback automatico CPU si Vulkan<1.2 o OOM.
Riesgos: driver AMD 2022 GCN1 (soporte minimo), validar con triangulo de
prueba antes de portar los 6 kernels. Estimacion honesta: 2-3 sesiones.
Alternativa ya existente hoy: llama.cpp -DGGML_VULKAN=ON con el GGUF original.

## Diagnostico Vulkan (fase 1 abortada)
vkCreateInstance SE CUELGA en este sistema (API 1.0 y 1.1): el cargador ICD de
AMD/driver-2022 sobre portatil hibrido Intel+R5M330 no completa la enumeracion.
Verificado con log por etapas: dll carga OK, punteros OK, instancia = hang.
Rutas: actualizar driver AMD/Radeon Settings; probar llama.cpp-Vulkan (mismo
riesgo); o hardware: SO-DIMM canal A (x2 garantizado, sin driver que cuelgue).
El codigo de sonda queda en src/l7_vulkan.c + comando vkinfo para retestear
tras actualizar drivers.
