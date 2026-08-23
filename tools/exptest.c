/* prueba numerica de exp256 vs expf */
#include <stdio.h>
#include <math.h>
#include <immintrin.h>

static inline __m256 exp256(__m256 x){
  const __m256 log2e=_mm256_set1_ps(1.4426950408889634f);
  const __m256 ln2hi=_mm256_set1_ps(0.693359375f);
  const __m256 ln2lo=_mm256_set1_ps(-2.12194440e-4f);
  const __m256 half=_mm256_set1_ps(0.5f);
  __m256 t=_mm256_mul_ps(x,log2e);
  __m256 n=_mm256_floor_ps(_mm256_add_ps(t,half));
  __m256 r=_mm256_fnmadd_ps(n,ln2hi,x);
  r=_mm256_fnmadd_ps(n,ln2lo,r);
  __m256 p=_mm256_fmadd_ps(_mm256_set1_ps(1.f/720.f),r,_mm256_set1_ps(1.f/120.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/24.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/6.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(0.5f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f));
  p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f));
  __m256i ni=_mm256_cvtps_epi32(n);
  ni=_mm256_max_epi32(ni,_mm256_set1_epi32(-126));
  ni=_mm256_min_epi32(ni,_mm256_set1_epi32(127));
  __m256 s=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni,_mm256_set1_epi32(127)),23));
  return _mm256_mul_ps(p,s);
}

int main(void){
  double maxrel=0; int worst=0;
  for(int i=-300;i<=300;i++){
    float x=(float)i/100.f;
    __m256 v=exp256(_mm256_set1_ps(x));
    float got[8]; _mm256_storeu_ps(got,v);
    float ref=expf(x);
    double rel=fabs((double)got[0]-(double)ref)/(ref>1e-30?(double)ref:1.0);
    if(rel>maxrel){ maxrel=rel; worst=i; }
    if(i%97==0) printf("exp(%+.2f)=%g ref=%g\n",x,got[0],ref);
  }
  printf("MAX_REL_ERR=%g en x=%+.2f\n",maxrel,(float)worst/100.f);
  return 0;
}
