/* forward_dense.c — forward Llama/Qwen denso + dispatch + helpers compartidos (Fase 3) */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

void load_vec_f32(Model *m, Slot *s, f32 *dst, i32 n){
  if(!s){ memset(dst,0,(size_t)n*4); return; }
  if(s->type==T_F32){ memcpy(dst,slot_ptr(m,s),(size_t)n*4); return; }
  gguf_dequant(s->type, slot_ptr(m,s), dst, (u64)n);
}

void apply_rope(Model *m, f32 *x, i32 len, i32 pos, i32 head_dim, f32 theta){
  /* Llama: interleaved pairs. Qwen2/Qwen3: NEOX half-split. */
  if(m->arch == ARCH_LLAMA) rope_th_llama(x, len, pos, head_dim, theta);
  else rope_th_neox(x, len, pos, head_dim, theta);
}

int require_slot(Slot *s, const char *what, i32 layer){
  if(s) return 0;
  if(layer>=0) fprintf(stderr,"fwd: missing slot %s L%d\n", what, layer);
  else fprintf(stderr,"fwd: missing slot %s\n", what);
  return -1;
}

/* dot y fma sobre head_dim: AVX2 (4 acumuladores) con fallback escalar.
   Igual matemática que la atención previa, ahora reutilizable por head. */
f32 dot_hd(const f32 *a, const f32 *b, i32 n){
#if defined(__AVX2__)
  __m256 s0=_mm256_setzero_ps(), s1=_mm256_setzero_ps(), s2=_mm256_setzero_ps(), s3=_mm256_setzero_ps();
  i32 j=0;
  for(;j+31<n;j+=32){
    s0=_mm256_fmadd_ps(_mm256_loadu_ps(a+j),   _mm256_loadu_ps(b+j),   s0);
    s1=_mm256_fmadd_ps(_mm256_loadu_ps(a+j+8), _mm256_loadu_ps(b+j+8), s1);
    s2=_mm256_fmadd_ps(_mm256_loadu_ps(a+j+16),_mm256_loadu_ps(b+j+16),s2);
    s3=_mm256_fmadd_ps(_mm256_loadu_ps(a+j+24),_mm256_loadu_ps(b+j+24),s3);
  }
  __m256 ss=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
  f32 s=hsum_ps(ss);
  for(;j<n;j++) s+=a[j]*b[j];
  return s;
#else
  f32 s=0; for(i32 j=0;j<n;j++) s+=a[j]*b[j]; return s;
#endif
}
void fma_hd(f32 *dst, const f32 *src, f32 scale, i32 n){
#if defined(__AVX2__)
  __m256 va=_mm256_set1_ps(scale);
  i32 j=0;
  for(;j+31<n;j+=32){
    _mm256_storeu_ps(dst+j,   _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j),   _mm256_loadu_ps(dst+j)));
    _mm256_storeu_ps(dst+j+8, _mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+8), _mm256_loadu_ps(dst+j+8)));
    _mm256_storeu_ps(dst+j+16,_mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+16),_mm256_loadu_ps(dst+j+16)));
    _mm256_storeu_ps(dst+j+24,_mm256_fmadd_ps(va,_mm256_loadu_ps(src+j+24),_mm256_loadu_ps(dst+j+24)));
  }
  for(;j<n;j++) dst[j]+=scale*src[j];
#else
  for(i32 j=0;j<n;j++) dst[j]+=scale*src[j];
#endif
}

void model_forward(Model *m, i32 token, i32 pos, f32 *logits){
  model_forward_ex(m,token,pos,logits,1);
}

/* ── LFM2 (LiquidAI): híbrido shortconv + atención ── */

void model_forward_ex(Model *m, i32 token, i32 pos, f32 *logits, int want_logits){
  if(m->arch==ARCH_LFM2){ forward_lfm2(m,token,pos,logits,want_logits); return; }
  if(m->arch==ARCH_QWEN35){ forward_hybrid(m,token,pos,logits,want_logits); return; }
  ModelCfg *c=&m->c;
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd, ctx=m->ctx;
  i32 group = c->n_kv_heads>0 ? c->n_heads/c->n_kv_heads : 1;
  if(group<1) group=1;
  if(ctx<=0) ctx=c->seq_len;

  if(pos < 0 || pos >= ctx){
    static int warned;
    if(!warned){ fprintf(stderr,"fwd: pos=%d out of [0,%d); aborting forward\n", pos, ctx); warned=1; }
    if(logits) memset(logits, 0, (size_t)c->vocab * sizeof(f32));
    return;
  }
  if(token < 0 || token >= c->vocab){
    static int warned;
    if(!warned){ fprintf(stderr,"fwd: token=%d out of vocab %d\n", token, c->vocab); warned=1; }
    if(logits) memset(logits, 0, (size_t)c->vocab * sizeof(f32));
    return;
  }
  if(!m->buf || (!m->kcache && !m->kcq)){
    fprintf(stderr,"fwd: runtime not initialized\n"); return;
  }

  FwdScratch _sc; fwd_scratch_dense(m,&_sc);
  f32 *x=_sc.x, *xb=_sc.xb, *hb=_sc.hb, *hb2=_sc.hb2;
  f32 *q=_sc.q, *k=_sc.k, *v=_sc.v, *att=_sc.att, *row=_sc.row;

  Slot *emb=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(emb,"tok_embd",-1)) return;
  u8 *ep=slot_ptr(m,emb)+(size_t)token*row_stride(emb->type,dim);
  gguf_dequant(emb->type, ep, x, (u64)dim);

  for(i32 L=0;L<c->n_layers;L++){
      if(m->skip_layer && m->skip_layer[L]) continue; /* ShortGPT: bloque redundante */
    Slot *an=slot_get(m,R_ATTN_NORM,L);
    load_vec_f32(m,an,row,dim); rmsnorm(xb,x,row,dim,c->eps);
    if(m->collect_bi){ f32 *pre=m->bi_pre+(size_t)L*dim; f32 s=0;
      for(i32 i=0;i<dim;i++){ pre[i]=x[i]; s+=x[i]*x[i]; } m->bi_n2p[L]+=s; }

    Slot *wq=slot_get(m,R_ATTN_Q,L);
    Slot *wk=slot_get(m,R_ATTN_K,L);
    Slot *wv=slot_get(m,R_ATTN_V,L);
    if(require_slot(wq,"attn_q",L)||require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return;
    /* Fusion Q+K+V: solo si los slots son contiguos y del mismo tipo
       (los offsets densos lo permiten; con padding ALIGN64 no). */
    if(m->lora_r){
      matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq,row); lora_add(q,xb,m->loraA_q[L],m->loraB_q[L],m->loraM_q[L],dim,nq,m->lora_r);
      matmul_q(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,row);
      matmul_q(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,row); lora_add(v,xb,m->loraA_v[L],m->loraB_v[L],m->loraM_v[L],dim,nkv,m->lora_r);
    } else if(wq->type==wk->type && wk->type==wv->type
       && slot_ptr(m,wk)==slot_ptr(m,wq)+wq->nbytes
       && slot_ptr(m,wv)==slot_ptr(m,wk)+wk->nbytes)
      matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq+nkv+nkv,row);
    else {
      matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq,row);
      matmul_q(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,row);
      matmul_q(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,row);
    }

    /* Sesgos de atencion (exclusivo Qwen2.5): Q += b_q, K += b_k, V += b_v */
    { Slot *qb=slot_get(m,R_ATTN_Q_BIAS,L); if(qb){ load_vec_f32(m,qb,row,nq); for(i32 i=0;i<nq;i++) q[i]+=row[i]; }
      Slot *kb=slot_get(m,R_ATTN_K_BIAS,L); if(kb){ load_vec_f32(m,kb,row,nkv); for(i32 i=0;i<nkv;i++) k[i]+=row[i]; }
      Slot *vb=slot_get(m,R_ATTN_V_BIAS,L); if(vb){ load_vec_f32(m,vb,row,nkv); for(i32 i=0;i<nkv;i++) v[i]+=row[i]; } }

    if(m->flags & F_QK_NORM){
      Slot *qn=slot_get(m,R_ATTN_Q_NORM,L);
      Slot *kn=slot_get(m,R_ATTN_K_NORM,L);
      f32 *qw=row;
      if(qn){ load_vec_f32(m,qn,qw,hd); qk_rmsnorm(q,qw,c->n_heads,hd,c->eps); }
      if(kn){ load_vec_f32(m,kn,qw,hd); qk_rmsnorm(k,qw,c->n_kv_heads,hd,c->eps); }
    }

    apply_rope(m, q, nq, pos, hd, c->rope_theta);
    apply_rope(m, k, nkv, pos, hd, c->rope_theta);

    kv_store(m, L, pos, k, v);

    f32 scale=1.f/sqrtf((f32)hd);
    /* Atención GQA-major: para cada grupo kv, cada fila K/V se dequantiza UNA vez
       y se reutiliza para todas las heads del grupo (antes: group× redundancia).
       Scratch K/V por hilo en heap (sin VLA en región OpenMP). */
    {
      i32 ngrp=(c->n_heads+group-1)/group;
#if defined(_OPENMP)
      #pragma omp parallel if(ngrp>=2 && pos>=16)
#endif
      {
          tls_kv_ensure(hd); f32 *krow=tls_krow; f32 *vrow=tls_vrow;
#if defined(_OPENMP)
        #pragma omp for schedule(static)
#endif
        for(i32 g=0; g<ngrp; g++){
          i32 h0=g*group, h1=h0+group; if(h1>c->n_heads) h1=c->n_heads;
          for(i32 t=0;t<=pos;t++){
            kv_key_row_h(m,L,t,h0/group,krow);
            for(i32 h=h0;h<h1;h++)
              att[(size_t)h*ctx+t]=dot_hd(q+(size_t)h*hd,krow,hd)*scale;
          }
          for(i32 h=h0;h<h1;h++) softmax(att+(size_t)h*ctx,pos+1);
          for(i32 h=h0;h<h1;h++){
            f32 *qh=q+(size_t)h*hd;
            for(i32 j=0;j<hd;j++) qh[j]=0.f;
          }
          for(i32 t=0;t<=pos;t++){
            kv_val_row_h(m,L,t,h0/group,vrow);
            for(i32 h=h0;h<h1;h++)
              fma_hd(q+(size_t)h*hd,vrow,att[(size_t)h*ctx+t],hd);
          }
          }
      }
    }

    Slot *wo=slot_get(m,R_ATTN_O,L);
    if(require_slot(wo,"attn_o",L)) return;
    matmul_q(xb,q,slot_ptr(m,wo),wo->type,nq,dim,row);
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];

    Slot *fn=slot_get(m,R_FFN_NORM,L);
    load_vec_f32(m,fn,row,dim); rmsnorm(xb,x,row,dim,c->eps);

    Slot *wg=slot_get(m,R_FFN_GATE,L);
    Slot *wu=slot_get(m,R_FFN_UP,L);
    Slot *wd=slot_get(m,R_FFN_DOWN,L);
    if(require_slot(wg,"ffn_gate",L)||require_slot(wu,"ffn_up",L)||require_slot(wd,"ffn_down",L)) return;
    /* Fusion gate+up idem: solo con slots contiguos y mismo tipo. */
    if(m->lora_r){
      matmul_q(hb, xb,slot_ptr(m,wg),wg->type,dim,hid,row); lora_add(hb,xb,m->loraA_gate[L],m->loraB_gate[L],m->loraM_gate[L],dim,hid,m->lora_r);
      matmul_q(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,row);
    } else if(wg->type==wu->type && slot_ptr(m,wu)==slot_ptr(m,wg)+wg->nbytes)
      matmul_q(hb,xb,slot_ptr(m,wg),wg->type,dim,hid*2,row);
    else {
      matmul_q(hb, xb,slot_ptr(m,wg),wg->type,dim,hid,row);
      matmul_q(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,row);
    }
    silu_mul(hb, hb2, hid); /* gate=silu(gate)*up fusionado */
    if(m->collect_stats && m->ffn_stats){
      f32 *st=m->ffn_stats+(size_t)L*hid;
      for(i32 i=0;i<hid;i++) st[i]+=fabsf(hb[i]);
    }
    matmul_q(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,row);
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];
  }

  Slot *on=slot_get(m,R_OUT_NORM,-1);
  load_vec_f32(m,on,row,dim); rmsnorm(x,x,row,dim,c->eps);
  if(!want_logits || !logits) return; /* prefill: saltar logits vocab×dim (carísimo) */
  Slot *out=slot_get(m,R_OUTPUT,-1);
  if(!out) out=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(out,"output",-1)) return;
  /* dual band CPU+GPU: si el worker Vulkan está activo, reparte el head */
  if(vk_head_dual(logits,x,slot_ptr(m,out),out->type,dim,c->vocab)) return;
  matmul_q(logits,x,slot_ptr(m,out),out->type,dim,c->vocab,row);
}

