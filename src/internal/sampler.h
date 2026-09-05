/* sampler.h — RNG + sampler top-k/top-p/repeat-penalty (Fase 2: implementación única).
 *
 * Antes vivía en 3 sitios (main.c, l5 model_sample, api por sesión).
 * Estado explícito en Sampler: cero globales, apto multi-sesión.
 */
#ifndef G2B_SAMPLER_H
#define G2B_SAMPLER_H
#include <stdint.h>

typedef struct { int32_t id; float logit; } SampPair;
typedef struct { uint64_t state; } SampRng;
typedef struct {
  SampRng rng;
  SampPair *buf; int cap;
  uint8_t *seen; int seen_n;
} Sampler;

void sampler_init(Sampler *s);   /* cero */
void sampler_free(Sampler *s);   /* libera scratch */
void sampler_seed(Sampler *s, uint64_t seed); /* 0 = no determinista (time) */
float sampler_rnd(Sampler *s);   /* xorshift64* en [0,1) */
/* top-k O(n) + Gumbel-max exacto + repeat-penalty (una sola pasada). */
int32_t sampler_sample(Sampler *s, float *logits, int n, float temp,
                       int top_k, float top_p, float repeat_penalty,
                       const int32_t *recent, int recent_n);

#endif
