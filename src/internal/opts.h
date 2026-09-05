/* opts.h — parser común de flags CLI (Fase 2: una sola implementación).
 *
 * Los 4 comandos de generación (run/chat/bench/ppl) compartían ~11 flags
 * copiados con divergencias (A44). opts_common_try consume los comunes;
 * cada comando conserva solo sus flags propios.
 */
#ifndef G2B_OPTS_H
#define G2B_OPTS_H
#include <stdint.h>

typedef struct {
  int ctx, threads, q8kv, f32kv, fast, ndrop, gpu;
  uint64_t max_ram_mb;   /* MB (fill_cfg la pasa a bytes) */
  uint64_t seed;
  float mv_ratio;
  const char *swap;      /* "@" = ruta por defecto del sistema */
} OptsCommon;

void opts_common_init(OptsCommon *o);
/* 1 si argv[*i] era un flag común (consume su argumento si lo lleva). */
int opts_common_try(OptsCommon *o, int argc, char **argv, int *i);

#endif
