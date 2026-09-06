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
## Actualizacion diagnostico Vulkan
Comportamiento INESTABLE confirmado: 3 corridas seguidas salen silenciosas
sin enumerar dispositivos; antes colgaba en vkCreateInstance. Sintoma clasico
de conflictos ICD en laptops hibridas con drivers viejos (AMD 27.20 ago-2022,
Intel 30.0.101 feb-2022).

Plan para el usuario (fuera del codigo):
1. Actualizar driver AMD (Adrenalin actual soporta R5 M330).
2. Revisar JSONs ICD en C:\Windows\System32\DriverStore y registro
   HKLM\SOFTWARE\Khronos\Vulkan\Drivers.
3. Retest: gguf2bin2.exe vkinfo  (sonda conservada en src/l7_vulkan.c)
4. Cross-check independiente: binario llama.cpp-Vulkan oficial — si tambien
   falla, es 100% sistema/driver.
Mientras tanto el runtime CPU (15.9 tok/s q4s) no depende de nada de esto.

## Fase 5 (2026-09-05): R1/R3/R4 medidos con A/B intercalado — veredicto
Maquina: el mismo i5-6200U (2C/4T). Qwen3-0.6B Q4: ~340 MB/token;
22-25 tok/s = 7.5-8.5 GB/s ~ TECHO del bus single-channel. Conclusión:
el decode está limitado por BYTES, no por ALU. Todo lo que añada tráfico
(por pequeño que sea) pierde; lo que ahorre ALU es invisible.
- **R1 (pre-conversión de escalas fp16, VLA por fila): −8 % consistente**
  (3 pares A/B: 22.0>21.3, 23.2>20.1, 24.6>22.6). Causa: el array scv añade
  8 B de tráfico (write+read) por bloque contra 2 B que ahorra. Con `_cvtsh_ss`
  por hardware la conversión ya era ~gratis. REVERTIDO. Lección: con F16C no
  repetir este experimento; sin F16C (build escalar) podría ganar.
- **R4 (norms en F32 en carga): neutro** (24.8 vs 24.4, dentro del ruido ±3 %).
  El dequant de norms es ~0.04 % del FLOP/token por construcción. REVERTIDO
  (árbol esbelto > micro-opt no demostrable).
- **R3 (caché cos/sin por posición): neutro** (decode +1.0/−0.5/−1.3,
  prefill +0.3/−0.8). El trig es ~0.5 % del token. REVERTIDO por el mismo
  estándar. La factorización tabla/aplicación queda como idea si algún día
  el prefill largo domina.
- **Threads**: sweep 1/2/3/4/5/6/8 → 13.7/19.6/19.7/23.3/19.3/21.9/21.9.
  4 hilos (default OpenMP) es el óptimo; más hilos solo añaden contienda.
  `OMP_PROC_BIND` ±2 % (no concluyente, sin cambio).
- **Conservado**: `--json` en bench (infra de medición para CI histórico).
- **Lo único que puede mover decode aquí**: menos bytes/token (R6 atención
  por bloques en ctx>1k, R10 especulativo) o menos pesos/token (--mv/--prune,
  ya existen con su tradeoff). R6 exige antes un harness de bench en ctx
  largo (el bench actual cicla pos<32) + validación de calidad.

## Fase 7 (2026-09-05): veredictos §17 ejecutados — el experimento que se mide, se jubila
- **17.1 CYBER-mRNA** → `experimental/cyber-mrna/` (opción b): el archivo
  histórico adaptado + README que dice lo que era de verdad (búsqueda
  estocástica de perturbación, NO gradiente; curva sintética en --particle;
  nombres DoRA/GaLore/MoE/SecEval = marketing). Lo que era REAL (alloc/
  apply/save/load LoRA) vive en `src/l8_lora.c`. `cyber-train`, `cyber-pack`,
  `bench-cyber` (score constante 4/1) eliminados de la CLI. `--cyber <lora>`
  se conserva: carga adaptadores v1/v2 reales.
- **17.2 MV + 17.3 BVH** → **retirados**, no avisados. ppl base 25.4 →
  `--mv 0.1` 35 629 (×1400), `--mv 0.3` 61 985, `--mv 0.5` 352 904;
  `--bvh` 19 966 (×786) en las pruebas 0.1/0.3/0.5. Aunque keep cambia
  la máscara al cruzar 0.2, los resultados redondeados fueron iguales;
  no se aisló la causa. Se elimina mv_table/bvh_*/hitrate de Model/API/CLI.
  El hitrate contaba repetición de tokens, no aciertos de predicción;
  las primeras apariciones no incrementaban misses.
- **17.4/17.5 OrderBook/HDR/ZRAM/FM-index/rope_th** → eliminados (código
  muerto verificado; fm_build era O(n²) y `fm_contains` nunca se llamó).
- **17.6 formatos propios** → ppl medido (corpus: README+README.es+SPEC,
  1024 tok, procesos separados): Q4_0 25.4 | Q8_0 20.8 (qwen3-0.6B);
  lfm2-1.2B: q4max 28.1 vs q4s 34.9. Llama: baseline 83.966 vs VVC
  1177.176 (256 tokens); baseline 82.325 vs PSY 2380555.838 (128 tokens).
  Son pruebas locales cortas de los archivos disponibles, no una evaluación
  general del formato. PSY/VVC siguen soportados, pero desaconsejados;
  falta aislar la causa de la degradación y verificar la procedencia del baseline.
- Nota método: PowerShell `>` produce UTF-16LE con BOM; el tokenizador
  encuentra un NUL y trunca el texto. Se descartaron esas pruebas y se usaron
  bytes UTF-8. El corpus temporal y los logs se borraron al limpiar: los
  números anteriores quedan como observaciones de sesión, no evidencia
  reproducible archivada. Los README posteriores ya no son el mismo corpus.

## Fase 6 (2026-09-05): unificación parcial — la plantilla total, rechazada
- Hecho: tabla de dispatch única (`qmat_lookup`: 11 tipos; B2 fixed — `rows`
  ya no deja `out` sin escribir; eliminada línea duplicada T_Q4_0S) +
  helpers de preámbulo (`q4_decode_setup`/`q4_batch_setup`, 13 sitios) +
  test de wiring en selftest (direcciones exactas + rows-vs-directo + B2).
  qkcheck 46/46 + suite bit-idéntica.
- Rechazado (argumentado): macro-plantilla única Q4_0/Q4_0S/PSY. Los loops
  difieren en unroll (4 vs 8) y blocking (por-32 vs por-superbloque;
  token-outer vs G-grupos en _b) de forma INTENCIONAL (tuning medido).
  Forzarlos a una plantilla añade ramas al hot loop o macro-soup ilegible,
  con riesgo en hardware al límite del bus y cero ganancia funcional. Lo
  compartible de verdad (primitivas Q4_BLK_*, preámbulos, dispatch) ya está
  compartido; los cuerpos quedan libres para evolucionar por formato.
- Bug cazado por el camino (lección): al extraer el preámbulo se pasó
  `g_q4sum` como argumento — evaluado ANTES del posible realloc en
  `q4_scratch` (use-after-free en crecimiento; detonaba como heap-corrupt
  flaky solo en secuencias small→big). Regla: los globales realojables se
  leen DENTRO (tras asegurar), nunca como argumento. El harness que lo
  habría cazado en el acto: secuencia multi-tamaño en un solo proceso
  (q4bcheck ya la tenía — por eso se detectó aquí y no en producción).
