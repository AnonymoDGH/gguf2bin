# CYBER-mRNA — experimento archivado (NO es entrenamiento real)

**Estado:** retirado del build en Fase 7 (§17.1, opción b). Este directorio no se
compila; se conserva por arqueología.

## Qué era de verdad

`cyber_train_exp.c` implementa **búsqueda estocástica de perturbación** sobre los
pesos `B_gate` de un adaptador LoRA: propone ruido aleatorio (con momento y
saltos tipo Lévy), mide el loss real en una muestra de texto (`compute_loss`),
y acepta la propuesta solo si mejora (greedy). Eso es *random search*, no
entrenamiento por gradiente: **no hay backprop, no hay DoRA/GaLore/MoE** (esos
nombres en los logs no corresponden a los algoritmos reales), el "parallel
tempering" no mantiene réplicas, el "curriculum" es un índice modular, y la
variante `--particle` optimiza una **curva sintética** (`sim_loss`), ni siquiera
el modelo.

Las cifras que imprimía (`loss`, `acc 42→71%`, `SecEval`) eran **ilustrativas,
no resultados medidos**. No deben citarse como benchmarks.

## Qué sí es real (y sigue en el build)

- `src/l8_lora.c`: alloc/apply/save/load del adaptador LoRA (inferencia).
- `ppl`: perplejidad medida de verdad sobre texto (`g2b_ppl`).
- Los `.lora` del repo raíz son pesos reales en formato v1/v2 (cargables con
  `--cyber`), aunque su *procedencia* sea este experimento: trátese como
  adaptadores opacos, no como "modelos entrenados a loss 1.0".

## Si alguien quiere retomarlo (hacerlo real)

Backprop para LoRA r=8–32 solo en q/v/gate: el forward ya existe
(`model_forward_ex` + `lora_add`); falta el backward de las 3 proyecciones +
SGD/Adam. Esfuerzo estimado: 2–3 semanas. Entonces las métricas se medirán de
verdad con `ppl` antes/después.
