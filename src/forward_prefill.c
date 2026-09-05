/* forward_prefill.c — prefill batcheado denso (Fase 3) */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Prefill batcheado: B tokens por pasada de pesos ──
   Cada fila de pesos se lee UNA vez y se reusa para los B tokens del chunk
   (aritmética ×B); cada fila K/V se dequantiza una vez para todo el chunk.
   Matemáticamente equivalente al forward secuencial (mismo orden de suma). */
int model_prefill(Model *m, const i32 *toks, i32 n, i32 pos0, f32 *last_logits){
  if(!m || !toks || n<=0) return -1;
  if(m->arch==ARCH_LFM2 || m->arch==ARCH_QWEN35) return 1; /* recurrente: camino secuencial */
  ModelCfg *c=&m->c;
  if(!m->pf_B || !m->pf_x){ return 1; /* sin buffers: que el llamador use el secuencial */ }
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd, ctx=m->ctx;
  i32 group=c->n_heads/c->n_kv_heads; if(group<1) group=1;
  if(pos0<0 || pos0+n>ctx){
    fprintf(stderr,"prefill: range [%d,%d) outside ctx=%d\n",pos0,pos0+n,ctx);
    return -1;
  }
  Slot *emb=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(emb,"tok_embd",-1)) return -1;

  while(n>0){
    i32 B=n>m->pf_B ? m->pf_B : n;
    if(pos0+B>ctx) B=ctx-pos0;
    if(B<=0) break;
    f32 *x=m->pf_x, *xb=m->pf_xb, *hb=m->pf_hb, *hb2=m->pf_hb2;
    f32 *q=m->pf_q, *k=m->pf_k, *v=m->pf_v, *att=m->pf_att;
    f32 *row=m->buf+dim*3+hid*2+nq+nkv*2+c->n_heads*ctx; /* zona 'row' de buf */

    for(i32 t=0;t<B;t++){
      if(toks[t]<0 || toks[t]>=c->vocab){ fprintf(stderr,"prefill: token %d out of vocab\n",toks[t]); return -1; }
      u8 *ep=slot_ptr(m,emb)+(size_t)toks[t]*row_stride(emb->type,dim);
      gguf_dequant(emb->type,ep,x+(size_t)t*dim,(u64)dim);
    }

    for(i32 L=0;L<c->n_layers;L++){
      if(m->skip_layer && m->skip_layer[L]) continue; /* ShortGPT */
      Slot *an=slot_get(m,R_ATTN_NORM,L);
      load_vec_f32(m,an,row,dim);
      for(i32 t=0;t<B;t++) rmsnorm(xb+(size_t)t*dim,x+(size_t)t*dim,row,dim,c->eps);

      Slot *wq=slot_get(m,R_ATTN_Q,L);
      Slot *wk=slot_get(m,R_ATTN_K,L);
      Slot *wv=slot_get(m,R_ATTN_V,L);
      if(require_slot(wq,"attn_q",L)||require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return -1;
      matmul_q_b(q,xb,slot_ptr(m,wq),wq->type,dim,nq,B);
      matmul_q_b(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,B);
      matmul_q_b(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,B);

      { Slot *qb=slot_get(m,R_ATTN_Q_BIAS,L);
        if(qb){ load_vec_f32(m,qb,row,nq); for(i32 t=0;t<B;t++){ f32 *qq=q+(size_t)t*nq; for(i32 i=0;i<nq;i++) qq[i]+=row[i]; } }
        Slot *kb=slot_get(m,R_ATTN_K_BIAS,L);
        if(kb){ load_vec_f32(m,kb,row,nkv); for(i32 t=0;t<B;t++){ f32 *kk=k+(size_t)t*nkv; for(i32 i=0;i<nkv;i++) kk[i]+=row[i]; } }
        Slot *vb=slot_get(m,R_ATTN_V_BIAS,L);
        if(vb){ load_vec_f32(m,vb,row,nkv); for(i32 t=0;t<B;t++){ f32 *vv=v+(size_t)t*nkv; for(i32 i=0;i<nkv;i++) vv[i]+=row[i]; } } }

      if(m->flags & F_QK_NORM){
        Slot *qn=slot_get(m,R_ATTN_Q_NORM,L);
        Slot *kn=slot_get(m,R_ATTN_K_NORM,L);
        if(qn){ load_vec_f32(m,qn,row,hd); for(i32 t=0;t<B;t++) qk_rmsnorm(q+(size_t)t*nq,row,c->n_heads,hd,c->eps); }
        if(kn){ load_vec_f32(m,kn,row,hd); for(i32 t=0;t<B;t++) qk_rmsnorm(k+(size_t)t*nkv,row,c->n_kv_heads,hd,c->eps); }
      }

      for(i32 t=0;t<B;t++){
        apply_rope(m,q+(size_t)t*nq,nq,pos0+t,hd,c->rope_theta);
        apply_rope(m,k+(size_t)t*nkv,nkv,pos0+t,hd,c->rope_theta);
        kv_store(m,L,pos0+t,k+(size_t)t*nkv,v+(size_t)t*nkv);
      }

      f32 scale=1.f/sqrtf((f32)hd);
      /* atención batched: scores [B][heads][ctx]; K/V dequant una vez por posición */
      {
        i32 ngrp=(c->n_heads+group-1)/group;
#if defined(_OPENMP)
        #pragma omp parallel if(ngrp>=2 && pos0+B>=8)
#endif
        {
          tls_kv_ensure(hd); f32 *krow=tls_krow; f32 *vrow=tls_vrow;
#if defined(_OPENMP)
          #pragma omp for schedule(static)
#endif
          for(i32 g=0;g<ngrp;g++){
            i32 h0=g*group, h1=h0+group; if(h1>c->n_heads) h1=c->n_heads;
            for(i32 p=0;p<pos0+B;p++){
              kv_key_row_h(m,L,p,h0/group,krow);
              i32 tmin = p>=pos0 ? p-pos0 : 0;
              for(i32 t=tmin;t<B;t++)
                for(i32 h=h0;h<h1;h++)
                  att[((size_t)t*c->n_heads+h)*ctx+p]=dot_hd(q+(size_t)t*nq+(size_t)h*hd,krow,hd)*scale;
            }
            for(i32 t=0;t<B;t++)
              for(i32 h=h0;h<h1;h++)
                softmax(att+((size_t)t*c->n_heads+h)*ctx,pos0+t+1);
            for(i32 t=0;t<B;t++)
              for(i32 h=h0;h<h1;h++){
                f32 *qh=q+(size_t)t*nq+(size_t)h*hd;
                for(i32 j=0;j<hd;j++) qh[j]=0.f;
              }
            for(i32 p=0;p<pos0+B;p++){
              kv_val_row_h(m,L,p,h0/group,vrow);
              i32 tmin = p>=pos0 ? p-pos0 : 0;
              for(i32 t=tmin;t<B;t++)
                for(i32 h=h0;h<h1;h++)
                  fma_hd(q+(size_t)t*nq+(size_t)h*hd,vrow,att[((size_t)t*c->n_heads+h)*ctx+p],hd);
            }
            }
        }
      }

      Slot *wo=slot_get(m,R_ATTN_O,L);
      if(require_slot(wo,"attn_o",L)) return -1;
      matmul_q_b(xb,q,slot_ptr(m,wo),wo->type,nq,dim,B);
      for(i32 t=0;t<B;t++){
        f32 *xt=x+(size_t)t*dim, *xbt=xb+(size_t)t*dim;
        for(i32 i=0;i<dim;i++) xt[i]+=xbt[i];
      }

      Slot *fn=slot_get(m,R_FFN_NORM,L);
      load_vec_f32(m,fn,row,dim);
      for(i32 t=0;t<B;t++) rmsnorm(xb+(size_t)t*dim,x+(size_t)t*dim,row,dim,c->eps);

      Slot *wg=slot_get(m,R_FFN_GATE,L);
      Slot *wu=slot_get(m,R_FFN_UP,L);
      Slot *wd=slot_get(m,R_FFN_DOWN,L);
      if(require_slot(wg,"ffn_gate",L)||require_slot(wu,"ffn_up",L)||require_slot(wd,"ffn_down",L)) return -1;
      if(wg->type==wu->type && slot_ptr(m,wu)==slot_ptr(m,wg)+wg->nbytes){
        matmul_q_b(hb,xb,slot_ptr(m,wg),wg->type,dim,hid*2,B);
        for(i32 t=0;t<B;t++){
          f32 *g=hb+(size_t)t*(size_t)(hid*2);
          silu_mul(g,g+hid,hid);
        }
        for(i32 t=1;t<B;t++)
          memmove(hb+(size_t)t*hid, hb+(size_t)t*(size_t)(hid*2), (size_t)hid*sizeof(f32));
      } else {
        matmul_q_b(hb ,xb,slot_ptr(m,wg),wg->type,dim,hid,B);
        matmul_q_b(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,B);
        for(i32 t=0;t<B;t++) silu_mul(hb+(size_t)t*hid,hb2+(size_t)t*hid,hid);
      }
      if(m->collect_stats && m->ffn_stats){
        f32 *st=m->ffn_stats+(size_t)L*hid;
        for(i32 t=0;t<B;t++){
          const f32 *h=hb+(size_t)t*hid;
          for(i32 i=0;i<hid;i++) st[i]+=fabsf(h[i]);
        }
      }
      matmul_q_b(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,B);
      for(i32 t=0;t<B;t++){
        f32 *xt=x+(size_t)t*dim, *xbt=xb+(size_t)t*dim;
        for(i32 i=0;i<dim;i++) xt[i]+=xbt[i];
      }
    }

    if(last_logits && n-B==0){ /* solo el último chunk contiene el token final */
      Slot *on=slot_get(m,R_OUT_NORM,-1);
      load_vec_f32(m,on,row,dim);
      rmsnorm(m->buf,x+(size_t)(B-1)*dim,row,dim,c->eps);
      Slot *out=slot_get(m,R_OUTPUT,-1);
      if(!out) out=emb;
      if(!vk_head_dual(last_logits,m->buf,slot_ptr(m,out),out->type,dim,c->vocab))
        matmul_q(last_logits,m->buf,slot_ptr(m,out),out->type,dim,c->vocab,row);
      last_logits=NULL;
    }
    pos0+=B; toks+=B; n-=B;
  }
  return 0;
}

