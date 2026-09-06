/* kv.c — KV cache F32/Q8, swap a disco, runtime alloc, TLS (split de l5, Fase 3) */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
__thread f32 *tls_krow=NULL; __thread f32 *tls_vrow=NULL; __thread i32 tls_cap=0;
void tls_kv_ensure(i32 hd){
  if(tls_cap>=hd) return;
  free(tls_krow); free(tls_vrow);
  tls_krow=(f32*)malloc((size_t)hd*sizeof(f32)); tls_vrow=(f32*)malloc((size_t)hd*sizeof(f32));
  tls_cap=(tls_krow&&tls_vrow)?hd:0;
}
#else
f32 *tls_krow=NULL; f32 *tls_vrow=NULL; i32 tls_cap=0;
void tls_kv_ensure(i32 hd){
  if(tls_cap>=hd) return;
  free(tls_krow); free(tls_vrow);
  tls_krow=(f32*)malloc((size_t)hd*sizeof(f32)); tls_vrow=(f32*)malloc((size_t)hd*sizeof(f32));
  tls_cap=(tls_krow&&tls_vrow)?hd:0;
}
#endif
size_t kv_q8_rowsize(i32 nkv){ return ((size_t)nkv + 31u)/32u * 34u; }

static void q8_quant_row(const f32 *src, u8 *dst, i32 n){
  for(i32 off=0; off<n; off+=32){
    i32 m=(n-off)<32?(n-off):32;
    f32 amax=0; for(i32 j=0;j<m;j++){ f32 a=fabsf(src[off+j]); if(a>amax) amax=a; }
    f32 d=amax/127.f; if(!(d>0.f)) d=1e-10f;
    u16 sd=f32_to_half(d); memcpy(dst,&sd,2);
    for(i32 j=0;j<m;j++){ i32 q=(i32)lroundf(src[off+j]/d); if(q>127)q=127; else if(q<-127)q=-127; dst[2+j]=(u8)(i8)q; }
    for(i32 j=m;j<32;j++) dst[2+j]=0;
    dst+=34;
  }
}
#if !defined(__AVX2__)
static void q8_dequant_row(const u8 *src, f32 *out, i32 n){
  for(i32 off=0; off<n; off+=32){
    f32 s=half_to_float(*(const u16*)src); src+=2;
    i32 m=(n-off)<32?(n-off):32;
    for(i32 j=0;j<m;j++) out[off+j]=s*(i8)src[j];
    src+=32;
  }
}
#endif

#if defined(__AVX2__)
void q8_dequant_row_avx2(const u8 *src, f32 *out, i32 n){
  for(i32 off=0; off<n; off+=32){
    f32 sf=half_to_float(*(const u16*)src); src+=2;
    __m256 vs=_mm256_set1_ps(sf);
    __m256i q8=_mm256_loadu_si256((const __m256i*)src); src+=32;
    __m128i lo=_mm256_castsi256_si128(q8), hi=_mm256_extracti128_si256(q8,1);
    i32 m=(n-off)<32?(n-off):32;
    if(m==32){
      _mm256_storeu_ps(out+off,   _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs));
      _mm256_storeu_ps(out+off+8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs));
      _mm256_storeu_ps(out+off+16,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs));
      _mm256_storeu_ps(out+off+24,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs));
    } else {
      /* tail: dequant completo y copia parcial */
      f32 tmp[32];
      _mm256_storeu_ps(tmp,   _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),vs));
      _mm256_storeu_ps(tmp+8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8))),vs));
      _mm256_storeu_ps(tmp+16,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),vs));
      _mm256_storeu_ps(tmp+24,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8))),vs));
      memcpy(out+off,tmp,(size_t)m*4);
    }
  }
}
#endif

/* Respaldar la KV cache en un archivo (p.ej. D:) -> páginas frías se vuelcan a disco. */
static int kv_swap_alloc(Model *m, size_t bytes){
  if(!m->swap_path || bytes==0) return -1;
  if(os_map_rw_new(m->swap_path,bytes,&m->swapmap)) return -1;
  m->use_swap=1;
  return 0;
}
static void kv_swap_free(Model *m){
  if(!m->use_swap || !m->swapmap.view) return;
  os_unmap(&m->swapmap);
  m->use_swap=0;
  if(m->swap_path) os_unlink(m->swap_path);
}

/* Híbrido qwen35: solo las capas de atención completa usan KV cache. */
static i32 kv_iv(const Model *m){ return m->c.fa_interval>0?m->c.fa_interval:0; }
i32 kv_nlayers(const Model *m){
  i32 iv=kv_iv(m);
  if(!iv) return m->c.n_layers;
  return (m->c.n_layers+iv-1)/iv;
}
static i32 kv_lidx(const Model *m, i32 layer){
  i32 iv=kv_iv(m);
  return iv? layer/iv : layer;
}

void free_rt(Model *m){
  free(m->buf);
  m->buf=NULL; m->buf_floats=0;
  free(m->pf_pool);
  m->pf_pool=NULL;
  m->pf_x=m->pf_xb=m->pf_hb=m->pf_hb2=m->pf_q=m->pf_k=m->pf_v=m->pf_att=NULL;
  m->pf_B=0;
  free(m->ffn_stats); m->ffn_stats=NULL;
  free(m->conv_state); m->conv_state=NULL;
  free(m->ssm_st); m->ssm_st=NULL;
  if(m->use_swap && m->swapmap.view){ kv_swap_free(m); return; }
  free(m->kcache); free(m->vcache); free(m->kcq); free(m->vcq);
  m->kcache=NULL; m->vcache=NULL; m->kcq=NULL; m->vcq=NULL;
}
int alloc_rt(Model *m, i32 ctx){
  ModelCfg *c=&m->c;
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd;
  int q8=((m->flags&F_KV_Q8) && !m->no_kv_q8)?1:0;
  i32 maxn=dim>hid?dim:hid; if(c->vocab>maxn) maxn=c->vocab;
  if(m->arch==ARCH_LFM2 && 3*dim>maxn) maxn=3*dim;
  i32 nbuf=dim*3 + hid*2 + nq + nkv*2 + c->n_heads*ctx + maxn;
  /* qwen35: scratch GDN (qkv[inner+2gds] + z + o) + atención gated (wq 2nq + gate nq)
     + scores (nh*ctx); ver hybrid_scratch() */
  if(m->arch==ARCH_QWEN35)
    nbuf += c->ssm_inner*3 + 2*c->ssm_n_group*c->ssm_d_state
          + 3*c->n_heads*c->head_dim + c->n_heads*ctx + 64;
  m->buf=malloc((size_t)nbuf*sizeof(f32)); /* no calloc: se sobreescribe antes de leer */
  if(!m->buf) return -1;
  m->buf_floats=(size_t)nbuf; /* tope para los asserts de carve (A25) */
  size_t half = 0, usize = 0;
  i32 nkvL=kv_nlayers(m);
  if(q8){ size_t qr=kv_q8_rowsize(nkv); half=(size_t)nkvL*(size_t)ctx*qr; }
  else   half=(size_t)nkvL*(size_t)ctx*(size_t)nkv*sizeof(f32);
  usize=2*half;
  if(m->swap_path && kv_swap_alloc(m,usize)==0){
    u8 *base=(u8*)m->swapmap.view;
    if(q8){ m->kcq=base; m->vcq=base+half; m->kcache=NULL; m->vcache=NULL; }
    else  { m->kcache=(f32*)base; m->vcache=(f32*)(base+half); m->kcq=NULL; m->vcq=NULL; }
  } else {
    if(q8){
      m->kcq=malloc(half); m->vcq=malloc(half);
      if(!m->kcq || !m->vcq){ free_rt(m); return -1; }
    } else {
      m->kcache=malloc(half); m->vcache=malloc(half);
      if(!m->kcache || !m->vcache){ free_rt(m); return -1; }
    }
  }
  /* buffers del prefill batcheado */
  i32 B=G2BX_PF_B;
  size_t nx=(size_t)B*(size_t)(dim + dim + hid*2 + nq + nkv*2)*sizeof(f32)
           +(size_t)B*(size_t)c->n_heads*(size_t)ctx*sizeof(f32);
  f32 *pool=malloc(nx);
  if(!pool){ /* sin prefill batcheado, pero el runtime secuencial funciona */ m->pf_B=0; return 0; }
  m->pf_pool=pool;
  m->pf_B=B;
  m->pf_x=pool; pool+=(size_t)B*dim;
  m->pf_xb=pool; pool+=(size_t)B*dim;
  m->pf_hb=pool; pool+=(size_t)B*hid;
  m->pf_hb2=pool; pool+=(size_t)B*hid;
  m->pf_q=pool; pool+=(size_t)B*nq;
  m->pf_k=pool; pool+=(size_t)B*nkv;
  m->pf_v=pool; pool+=(size_t)B*nkv;
  m->pf_att=pool;
#ifdef DBG_RT
  fprintf(stderr,"alloc_rt: buf=%p pool=%p B=%d nx=%.1fMB ctx=%d\n",(void*)m->buf,(void*)m->pf_pool,B,nx/1048576.0,ctx);
#endif
  if(m->arch==ARCH_LFM2){
    m->conv_state=calloc((size_t)c->n_layers*2u*(size_t)dim,sizeof(f32));
    if(!m->conv_state){ free_rt(m); return -1; }
  }
  if(m->arch==ARCH_QWEN35 && c->fa_interval>0){
    /* estados GDN: [n_recr][nv][dv][dv] + conv [n_recr][conv_dim][d_conv-1] */
    i32 iv=c->fa_interval, nrecr=0;
    for(i32 L=0;L<c->n_layers;L++) if((L+1)%iv) nrecr++;
    i32 dv=c->ssm_inner/c->ssm_dt_rank;
    size_t ns=(size_t)nrecr*(size_t)c->ssm_dt_rank*(size_t)dv*(size_t)dv;
    size_t nc=(size_t)nrecr*(size_t)(c->ssm_inner+2*c->ssm_n_group*c->ssm_d_state)*(size_t)c->ssm_d_conv;
    m->ssm_st=calloc(ns,sizeof(f32));
    m->conv_state=calloc(nc,sizeof(f32));
    if(!m->ssm_st||!m->conv_state){ free_rt(m); return -1; }
  }
  return 0;
}

/* Respaldar la KV cache en un archivo (p.ej. D:) para que las páginas frías
   se vuelquen a disco y la RAM física quede para lo caliente. */
int model_enable_swap(Model *m, const char *path){
  if(!m || !path || !*path) return -1;
  free(m->swap_path); m->swap_path=NULL;
  m->swap_path=strdup(path);
  if(!m->swap_path){ fprintf(stderr,"swap: OOM\n"); return -1; }
  if(m->ctx>0) {
    if(model_set_ctx(m,m->ctx)){ free(m->swap_path); m->swap_path=NULL; return -1; }
  }
  fprintf(stderr,"swap: KV cache backed by %s\n", path);
  return 0;
}

int kv_is_q8(const Model *m){ return (m->flags&F_KV_Q8) && !m->no_kv_q8; }
/* Híbrido qwen35: solo las capas de atención completa usan KV cache.
   El slot KV de la capa L es L/interval; las capas SSM no tocan la cache. */

static size_t kv_pos_offset(Model *m, i32 layer, i32 pos){
  i32 nl=kv_nlayers(m), idx=kv_lidx(m,layer);
  return ((size_t)idx*m->ctx + (size_t)pos) * (size_t)m->c.n_kv_heads * (size_t)m->c.head_dim;
}
void kv_store(Model *m, i32 layer, i32 pos, const f32 *k, const f32 *v){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim;
  i32 idx=kv_lidx(m,layer);
  if(kv_is_q8(m)){
    size_t qr=kv_q8_rowsize(nkv);
    u8 *kb=m->kcq+((size_t)idx*m->ctx+(size_t)pos)*qr;
    u8 *vb=m->vcq+((size_t)idx*m->ctx+(size_t)pos)*qr;
    q8_quant_row(k,kb,nkv); q8_quant_row(v,vb,nkv);
  } else {
    f32 *kc=m->kcache+kv_pos_offset(m,layer,pos);
    f32 *vc=m->vcache+kv_pos_offset(m,layer,pos);
    memcpy(kc,k,(size_t)nkv*4); memcpy(vc,v,(size_t)nkv*4);
  }
}
/* Dequant solo la fila de UN head (hd elementos), no la fila KV completa.
   head_dim múltiplo de 32 → el segmento cae en límite de bloque Q8
   (garantizado en carga: si no, no_kv_q8 fuerza KV F32). */
void kv_key_row_h(Model *m, i32 layer, i32 pos, i32 kvh, f32 *out){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim, hd=m->c.head_dim;
  i32 eo=kvh*hd, idx=kv_lidx(m,layer);
  if(kv_is_q8(m)){
    size_t qr=kv_q8_rowsize(nkv);
    const u8 *src=m->kcq+((size_t)idx*m->ctx+(size_t)pos)*qr + (size_t)(eo/32)*34u;
#if defined(__AVX2__)
    q8_dequant_row_avx2(src, out, hd);
#else
    q8_dequant_row(src, out, hd);
#endif
  } else {
    memcpy(out, m->kcache+kv_pos_offset(m,layer,pos)+(size_t)eo, (size_t)hd*4);
  }
}
void kv_val_row_h(Model *m, i32 layer, i32 pos, i32 kvh, f32 *out){
  i32 nkv=m->c.n_kv_heads*m->c.head_dim, hd=m->c.head_dim;
  i32 eo=kvh*hd, idx=kv_lidx(m,layer);
  if(kv_is_q8(m)){
    size_t qr=kv_q8_rowsize(nkv);
    const u8 *src=m->vcq+((size_t)idx*m->ctx+(size_t)pos)*qr + (size_t)(eo/32)*34u;
#if defined(__AVX2__)
    q8_dequant_row_avx2(src, out, hd);
#else
    q8_dequant_row(src, out, hd);
#endif
  } else {
    memcpy(out, m->vcache+kv_pos_offset(m,layer,pos)+(size_t)eo, (size_t)hd*4);
  }
}

