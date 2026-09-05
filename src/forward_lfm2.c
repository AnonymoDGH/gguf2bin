/* forward_lfm2.c — forward híbrido shortconv+atención (Fase 3) */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

void forward_lfm2(Model *m, i32 token, i32 pos, f32 *logits, int want_logits){
  static i8 dbg=-1; if(dbg==-1){ dbg=getenv("G2BX_DBG")?1:0; }
  ModelCfg *c=&m->c;
  i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd, ctx=m->ctx;
  i32 group=c->n_kv_heads>0?c->n_heads/c->n_kv_heads:1;
  if(group<1) group=1;
  if(ctx<=0) ctx=c->seq_len;
  if(pos<0||pos>=ctx||token<0||token>=c->vocab||!m->buf) return;

  FwdScratch _sc; fwd_scratch_dense(m,&_sc);
  f32 *x=_sc.x, *xb=_sc.xb, *xb2=_sc.xb2, *hb=_sc.hb, *hb2=_sc.hb2;
  f32 *q=_sc.q, *k=_sc.k, *v=_sc.v, *att=_sc.att, *row=_sc.row;

  Slot *emb=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(emb,"tok_embd",-1)) return;
  gguf_dequant(emb->type,slot_ptr(m,emb)+(size_t)token*row_stride(emb->type,dim),x,(u64)dim);
  /* OJO: LFM2 no normaliza tras el embedding; token_embd_norm es la norma FINAL */

  if(dbg&&pos<3){ f32 s=0; for(i32 i=0;i<dim;i++) s+=x[i]*x[i]; fprintf(stderr,"[dbg] pos=%d post-embdnorm |x|=%g\n",pos,sqrtf(s)); }

  for(i32 L=0;L<c->n_layers;L++){
      if(m->skip_layer && m->skip_layer[L]) continue; /* ShortGPT: bloque redundante */
    Slot *an=slot_get(m,R_ATTN_NORM,L);
    load_vec_f32(m,an,row,dim); rmsnorm(xb,x,row,dim,c->eps);
    if(m->collect_bi){ f32 *pre=m->bi_pre+(size_t)L*dim; f32 s=0;
      for(i32 i=0;i<dim;i++){ pre[i]=x[i]; s+=x[i]*x[i]; } m->bi_n2p[L]+=s; }
    if(dbg&&pos<1&&L<3){ f32 s=0,w=0,w2=0,xr=0; for(i32 i=0;i<dim;i++){ s+=xb[i]*xb[i]; w+=row[i]; w2+=row[i]*row[i]; xr+=x[i]*x[i]; }
      fprintf(stderr,"[dbg] L%d: |x_resid|=%g rms_w=%g |xb|=%g\n",L,sqrtf(xr),sqrtf(w2/dim),sqrtf(s)); }

    Slot *wq=slot_get(m,R_ATTN_Q,L);
    if(wq){
      Slot *wk=slot_get(m,R_ATTN_K,L);
      Slot *wv=slot_get(m,R_ATTN_V,L);
      if(require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return;
      matmul_q(q,xb,slot_ptr(m,wq),wq->type,dim,nq,row);
      matmul_q(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,row);
      matmul_q(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,row);
      if(dbg&&pos<1&&L==2){
        f32 sq0=0,sv=0,svx=0; for(i32 j=0;j<hd;j++) sq0+=q[j]*q[j];
        for(i32 j=0;j<nkv;j++){ sv+=v[j]*v[j]; svx+=fabsf(v[j]); }
        fprintf(stderr,"[dbg] pre-qknorm: |q_h0|=%g rms|v|=%g mean|v|=%g xb=%g %g %g %g\n",
          sqrtf(sq0),sqrtf(sv/nkv),svx/nkv,xb[0],xb[1],xb[2],xb[3]);
      }
      { Slot *qn=slot_get(m,R_ATTN_Q_NORM,L);
        Slot *kn=slot_get(m,R_ATTN_K_NORM,L);
        if(qn){ load_vec_f32(m,qn,row,hd); qk_rmsnorm(q,row,c->n_heads,hd,c->eps); }
        if(kn){ load_vec_f32(m,kn,row,hd); qk_rmsnorm(k,row,c->n_kv_heads,hd,c->eps); } }
      apply_rope(m,q,nq,pos,hd,c->rope_theta);
      apply_rope(m,k,nkv,pos,hd,c->rope_theta);
      kv_store(m,L,pos,k,v);
      f32 scale=1.f/sqrtf((f32)hd);
      {
        i32 ngrp=(c->n_heads+group-1)/group;
#if defined(_OPENMP)
        #pragma omp parallel if(ngrp>=2 && pos>=16)
#endif
        {
            tls_kv_ensure(hd);
            f32 *kr=tls_krow; f32 *vr=tls_vrow;
            if(!kr||!vr){
#if defined(_OPENMP)
              #pragma omp critical
#endif
              { tls_kv_ensure(hd); kr=tls_krow; vr=tls_vrow; }
            }
#if defined(_OPENMP)
          #pragma omp for schedule(static)
#endif
          for(i32 g2=0;g2<ngrp;g2++){
            i32 h0=g2*group, h1=h0+group; if(h1>c->n_heads) h1=c->n_heads;
            for(i32 t=0;t<=pos;t++){
              kv_key_row_h(m,L,t,h0/group,kr);
              for(i32 h=h0;h<h1;h++)
                att[(size_t)h*ctx+t]=dot_hd(q+(size_t)h*hd,kr,hd)*scale;
            }
            for(i32 h=h0;h<h1;h++) softmax(att+(size_t)h*ctx,pos+1);
            for(i32 h=h0;h<h1;h++){ f32*qh=q+(size_t)h*hd; for(i32 j=0;j<hd;j++) qh[j]=0.f; }
            for(i32 t=0;t<=pos;t++){
              kv_val_row_h(m,L,t,h0/group,vr);
              for(i32 h=h0;h<h1;h++)
                fma_hd(q+(size_t)h*hd,vr,att[(size_t)h*ctx+t],hd);
            }
          }
        }
      }
      Slot *wo=slot_get(m,R_ATTN_O,L);
      if(require_slot(wo,"attn_o",L)) return;
      matmul_q(xb,q,slot_ptr(m,wo),wo->type,nq,dim,row);
      { f32 s=0,sq=0; for(i32 i=0;i<dim;i++) s+=xb[i]*xb[i]; for(i32 j=0;j<hd;j++) sq+=q[j]*q[j];
        if(dbg&&pos<1&&L==2) fprintf(stderr,"[dbg] att-out |o|=%g |q_head0|=%g att0=%g v0=%g\n",sqrtf(s),sq,att[0],v[0]); }
      for(i32 i=0;i<dim;i++) x[i]+=xb[i];
    } else {
      /* ── bloque shortconv: B,C,x = in_proj(h); g=B*x; conv causal k=3; y=C*conv ── */
      Slot *ci=slot_get(m,R_CONV_IN,L);
      Slot *co=slot_get(m,R_CONV_OUT,L);
      Slot *cw=slot_get(m,R_CONV_W,L);
      if(require_slot(ci,"shortconv.in_proj",L)||require_slot(co,"shortconv.out_proj",L)||require_slot(cw,"shortconv.conv",L)) return;
      if(hid<3*dim){ fprintf(stderr,"fwd: LFM2 hidden_dim=%d < 3*dim=%d\n",hid,3*dim); return; }
      if(!m->conv_state) return;
      matmul_q(hb,xb,slot_ptr(m,ci),ci->type,dim,3*dim,row); /* B|C|x en hb[0..3dim) */
      f32 *Bp=hb, *Cp=hb+dim, *Xp=hb+2*dim;
      for(i32 i=0;i<dim;i++) xb2[i]=Bp[i]*Xp[i];
      f32 *cs=m->conv_state+(size_t)L*2*dim;
      load_vec_f32(m,cw,row,(u64)3*dim);
      const f32 *w=row; /* GGUF dims [k,ch]: elemento (k,ch) en ch*3+k */
      for(i32 ch=0;ch<dim;ch++){
        f32 y=w[ch*3+0]*cs[ch] + w[ch*3+1]*cs[dim+ch] + w[ch*3+2]*xb2[ch];
        cs[ch]=cs[dim+ch]; cs[dim+ch]=xb2[ch];
        xb2[ch]=y*Cp[ch];
      }
      matmul_q(xb,xb2,slot_ptr(m,co),co->type,dim,dim,row);
      if(dbg&&pos<1&&L==0){ f32 s=0; for(i32 i=0;i<dim;i++) s+=xb[i]*xb[i];
        fprintf(stderr,"[dbg] conv-branch-out |o|=%g o[0..4]=%g %g %g %g %g\n",sqrtf(s),xb[0],xb[1],xb[2],xb[3],xb[4]); }
      for(i32 i=0;i<dim;i++) x[i]+=xb[i];
    }

    /* FFN común */
    Slot *fn=slot_get(m,R_FFN_NORM,L);
    load_vec_f32(m,fn,row,dim); rmsnorm(xb,x,row,dim,c->eps);
    Slot *wg=slot_get(m,R_FFN_GATE,L);
    Slot *wu=slot_get(m,R_FFN_UP,L);
    Slot *wd=slot_get(m,R_FFN_DOWN,L);
    if(require_slot(wg,"ffn_gate",L)||require_slot(wu,"ffn_up",L)||require_slot(wd,"ffn_down",L)) return;
    if(wg->type==wu->type && slot_ptr(m,wu)==slot_ptr(m,wg)+wg->nbytes)
      matmul_q(hb,xb,slot_ptr(m,wg),wg->type,dim,hid*2,row);
    else {
      matmul_q(hb ,xb,slot_ptr(m,wg),wg->type,dim,hid,row);
      matmul_q(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,row);
    }
    silu_mul(hb, hb2, hid);
    if(m->collect_stats && m->ffn_stats){
      f32 *st=m->ffn_stats+(size_t)L*hid;
      for(i32 i=0;i<hid;i++) st[i]+=fabsf(hb[i]);
    }
    matmul_q(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,row);
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];
    if(m->collect_bi){ const f32 *pre=m->bi_pre+(size_t)L*dim; f32 dot=0,s=0;
      for(i32 i=0;i<dim;i++){ dot+=pre[i]*x[i]; s+=x[i]*x[i]; }
      m->bi_dot[L]+=dot; m->bi_n2[L]+=s; }
  }

  { /* LFM2 no tiene output_norm: reutiliza token_embd_norm al final */
    Slot *on=slot_get(m,R_OUT_NORM,-1);
    if(!on) on=slot_get(m,R_EMBD_NORM,-1);
    load_vec_f32(m,on,row,dim); rmsnorm(x,x,row,dim,c->eps); }
  if(dbg&&pos<4){ f32 s=0,mx=-1e30f,mn=1e30f; for(i32 i=0;i<dim;i++){ s+=x[i]*x[i]; if(x[i]>mx)mx=x[i]; if(x[i]<mn)mn=x[i]; }
    fprintf(stderr,"[dbg] final |x|=%g max=%g min=%g\n",sqrtf(s),mx,mn); }
  if(!want_logits || !logits) return;
  Slot *out=slot_get(m,R_OUTPUT,-1);
  if(!out) out=emb;
  if(require_slot(out,"output",-1)) return;
  if(vk_head_dual(logits,x,slot_ptr(m,out),out->type,dim,c->vocab)) return;
  matmul_q(logits,x,slot_ptr(m,out),out->type,dim,c->vocab,row);
  if(dbg&&pos<4){ f32 mx=-1e30f,mn=1e30f; i32 b1=0,b2=0,b3=0;
    for(i32 i=0;i<c->vocab;i++){ if(logits[i]>mx){mx=logits[i];b3=b2;b2=b1;b1=i;} if(logits[i]<mn)mn=logits[i]; }
    fprintf(stderr,"[dbg] logits min=%g max=%g top=%d,%d,%d (%g %g %g)\n",mn,mx,b1,b2,b3,logits[b1],logits[b2],logits[b3]); }
}

