/* forward_hybrid.c — forward qwen35 gated-delta-net (Fase 3) */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ── qwen35: gated delta net + atención completa cada N capas (CPU secuencial) ──
   Ref: llama.cpp src/models/qwen35.cpp + ggml_gated_delta_net (ops.cpp).
   (HY_NV_MAX se define arriba: lo usa también la validación de geometría en carga) */
static inline float softplusf(float x){
  if(x>20.f) return x;
  return logf(1.f+expf(x));
}
static void l2norm(f32 *x, i32 n, f32 eps){
  double ss=0; for(i32 i=0;i<n;i++) ss+=(double)x[i]*x[i];
  float inv=1.f/sqrtf((float)ss+eps);
  for(i32 i=0;i<n;i++) x[i]*=inv;
}
/* zona scratch híbrida: arranca tras la región del layout denso */
static f32 *hybrid_scratch(Model *m){
  ModelCfg *c=&m->c;
  const i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  i32 maxn=dim>hid?dim:hid; if(c->vocab>maxn) maxn=c->vocab;
  return m->buf + ((size_t)dim*3 + (size_t)hid*2
                   + (size_t)c->n_heads*hd + 2*(size_t)c->n_kv_heads*hd
                   + (size_t)c->n_heads*m->ctx + (size_t)maxn);
}

int forward_hybrid(Model *m, i32 token, i32 pos, f32 *logits, int want_logits){
  ModelCfg *c=&m->c;
  const i32 dim=c->dim, hid=c->hidden_dim, hd=c->head_dim;
  const i32 nq=c->n_heads*hd, nkv=c->n_kv_heads*hd;
  i32 ctx=m->ctx;
  const i32 iv=c->fa_interval;
  const i32 dk=c->ssm_d_state, nv=c->ssm_dt_rank, nk=c->ssm_n_group;
  const i32 dv=c->ssm_inner/nv;
  const i32 kdim=nk*dk;
  const i32 conv_dim=c->ssm_inner+2*kdim;
  const i32 dconv=c->ssm_d_conv;
  if(ctx<=0) ctx=c->seq_len;
  if(pos<0||pos>=ctx||token<0||token>=c->vocab) return -1;
  if(!m->buf||!m->ssm_st||!m->conv_state){ fprintf(stderr,"fwd: runtime not initialized\n"); return -1; }
  static i8 dbg_hyb=-1; if(dbg_hyb==-1){ dbg_hyb=getenv("G2BX_DBG")?1:0; }
  int _dbg_hyb = dbg_hyb;
  if(_dbg_hyb && pos<2) fprintf(stderr,"[C] forward_hybrid token %d pos %d dim %d\n", token, pos, dim);

  f32 *B=m->buf;
  f32 *x=B, *xb=B+dim, *row=B+2*dim;
  f32 *hs=hybrid_scratch(m);
  f32 *qkv=hs;                     /* conv_dim: qkv mezclado / activado */
  f32 *z=qkv+conv_dim;             /* inner: gate z (silu) */
  f32 *gdno=z+c->ssm_inner;        /* inner: salida GDN */
  f32 *wqo=gdno+c->ssm_inner;      /* 2*nq: Q+gate crudos (capas atención) */
  f32 *gate=wqo+2*nq;              /* nq */
  f32 *att=gate+nq;                /* n_heads*ctx: scores */
  assert(att+(size_t)c->n_heads*ctx >= m->buf
      && (size_t)(att+(size_t)c->n_heads*ctx-m->buf) <= m->buf_floats);

  Slot *emb=slot_get(m,R_TOK_EMBD,-1);
  if(require_slot(emb,"tok_embd",-1)) return -1;
  gguf_dequant(emb->type, slot_ptr(m,emb)+(size_t)token*row_stride(emb->type,dim), x, (u64)dim);

  i32 recL=0;
  for(i32 L=0;L<c->n_layers;L++){
    int is_attn=((L+1)%iv)==0;
    Slot *an=slot_get(m,R_ATTN_NORM,L);
    load_vec_f32(m,an,row,dim); rmsnorm(xb,x,row,dim,c->eps);

    if(is_attn){
      /* ── atención GQA con gate sigmoide y rope parcial ── */
      Slot *wq=slot_get(m,R_ATTN_Q,L);
      Slot *wk=slot_get(m,R_ATTN_K,L);
      Slot *wv=slot_get(m,R_ATTN_V,L);
      if(require_slot(wq,"attn_q",L)||require_slot(wk,"attn_k",L)||require_slot(wv,"attn_v",L)) return -1;
      /* Q+gate intercalados por head: [Q_h | G_h] cada 2*hd */
      matmul_q(wqo,xb,slot_ptr(m,wq),wq->type,dim,nq*2,row);
      /* Q va a la zona z (libre en capas de atención: inner==nq aquí) */
      f32 *q=z;
      for(i32 h=0;h<c->n_heads;h++){
        memcpy(q+(size_t)h*hd,    wqo+(size_t)(2*h)*hd,   (size_t)hd*4);
        memcpy(gate+(size_t)h*hd, wqo+(size_t)(2*h+1)*hd, (size_t)hd*4);
      }
      f32 *k=wqo, *v=wqo+nkv; /* reutiliza: q/gate ya extraídos */
      matmul_q(k,xb,slot_ptr(m,wk),wk->type,dim,nkv,row);
      matmul_q(v,xb,slot_ptr(m,wv),wv->type,dim,nkv,row);

      { Slot *qn=slot_get(m,R_ATTN_Q_NORM,L);
        Slot *kn=slot_get(m,R_ATTN_K_NORM,L);
        if(qn){ load_vec_f32(m,qn,row,hd); qk_rmsnorm(q,row,c->n_heads,hd,c->eps); }
        if(kn){ load_vec_f32(m,kn,row,hd); qk_rmsnorm(k,row,c->n_kv_heads,hd,c->eps); } }
      { i32 nr=c->n_rot>0?c->n_rot:hd;
        for(i32 h=0;h<c->n_heads;h++)     rope_th_neox(q+(size_t)h*hd,1,pos,nr,c->rope_theta);
        for(i32 h=0;h<c->n_kv_heads;h++)  rope_th_neox(k+(size_t)h*hd,1,pos,nr,c->rope_theta); }

      kv_store(m,L,pos,k,v);

      const f32 scale=1.f/sqrtf((f32)hd);
      const i32 group=c->n_heads/c->n_kv_heads;
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
          for(i32 g=0;g<ngrp;g++){
            i32 h0=g*group,h1=h0+group; if(h1>c->n_heads) h1=c->n_heads;
            for(i32 t=0;t<=pos;t++){
              kv_key_row_h(m,L,t,h0/group,krow);
              for(i32 h=h0;h<h1;h++)
                att[(size_t)h*ctx+t]=dot_hd(q+(size_t)h*hd,krow,hd)*scale;
            }
            for(i32 h=h0;h<h1;h++) softmax(att+(size_t)h*ctx,pos+1);
            for(i32 h=h0;h<h1;h++){ f32 *qh=q+(size_t)h*hd; for(i32 j=0;j<hd;j++) qh[j]=0.f; }
            for(i32 t=0;t<=pos;t++){
              kv_val_row_h(m,L,t,h0/group,vrow);
              for(i32 h=h0;h<h1;h++)
                fma_hd(q+(size_t)h*hd,vrow,att[(size_t)h*ctx+t],hd);
            }
          }
        }
      }
      /* gate sigmoide sobre la salida de atención */
      for(i32 i=0;i<nq;i++){ f32 gsv=gate[i]; q[i]*=1.f/(1.f+expf(-gsv)); }

      Slot *wo=slot_get(m,R_ATTN_O,L);
      if(require_slot(wo,"attn_o",L)) return -1;
      matmul_q(xb,q,slot_ptr(m,wo),wo->type,dim,dim,row);
      for(i32 i=0;i<dim;i++) x[i]+=xb[i];
    } else {
      /* ── gated delta net ── */
      Slot *wqkv=slot_get(m,R_ATTN_QKV,L);
      Slot *wg =slot_get(m,R_ATTN_GATE,L);
      Slot *wb =slot_get(m,R_SSM_BETA,L);
      Slot *wa =slot_get(m,R_SSM_ALPHA,L);
      Slot *wc =slot_get(m,R_SSM_CONV,L);
      Slot *wdt=slot_get(m,R_SSM_DT,L);
      Slot *wa_=slot_get(m,R_SSM_A,L);
      Slot *wn =slot_get(m,R_SSM_NORM,L);
      Slot *wo =slot_get(m,R_SSM_OUT,L);
      if(require_slot(wqkv,"attn_qkv",L)||require_slot(wg,"attn_gate",L)||
         require_slot(wb,"ssm_beta",L)||require_slot(wa,"ssm_alpha",L)||
         require_slot(wc,"ssm_conv1d",L)||require_slot(wn,"ssm_norm",L)||
         require_slot(wo,"ssm_out",L)||require_slot(wdt,"ssm_dt",L)||require_slot(wa_,"ssm_a",L)) return -1;

      matmul_q(qkv,xb,slot_ptr(m,wqkv),wqkv->type,dim,conv_dim,row);
      if(_dbg_hyb && L==0&&pos<2){ fprintf(stderr,"[C] tok%d normed0=%g %g %g | qkvproj0=%g %g %g\n",pos,
        xb[0],xb[1],xb[2],qkv[0],qkv[1],qkv[2]); }
      matmul_q(z,xb,slot_ptr(m,wg),wg->type,dim,c->ssm_inner,row);

      f32 beta_v[HY_NV_MAX], g_v[HY_NV_MAX], dtb[HY_NV_MAX], av[HY_NV_MAX];
      load_vec_f32(m,wdt,dtb,nv); load_vec_f32(m,wa_,av,nv);
      matmul_q(beta_v,xb,slot_ptr(m,wb),wb->type,dim,nv,row);
      matmul_q(g_v,xb,slot_ptr(m,wa),wa->type,dim,nv,row);
      for(i32 h=0;h<nv;h++){
        beta_v[h]=1.f/(1.f+expf(-beta_v[h]));
        g_v[h]=av[h]*softplusf(g_v[h]+dtb[h]); /* ≈ -exp(A_log)*softplus(a+dt) < 0 */
      }

      /* conv causal depthwise: estado [dc-1][conv_dim]; out → gdno temporalmente */
      f32 *cs=m->conv_state+(size_t)recL*conv_dim*(size_t)(dconv-1);
      {
        const f32 *w=(const f32*)slot_ptr(m,wc);
        for(i32 cc=0;cc<conv_dim;cc++){
          const f32 *wr=w+(size_t)cc*dconv;
          f32 s=0;
          for(i32 kk=0;kk<dconv-1;kk++) s+=wr[kk]*cs[(size_t)kk*conv_dim+cc];
          s+=wr[dconv-1]*qkv[cc];
          gdno[cc]=s;
        }
        memmove(cs,cs+conv_dim,(size_t)(dconv-2)*conv_dim*sizeof(f32));
        memcpy(cs+(size_t)(dconv-2)*conv_dim,qkv,(size_t)conv_dim*sizeof(f32));
        for(i32 cc=0;cc<conv_dim;cc++){ f32 vv=gdno[cc]; qkv[cc]=vv/(1.f+expf(-vv)); }
        if(_dbg_hyb && L==0&&pos<2){
          double _s=0; for(i32 _i=0;_i<128;_i++) _s+=(double)qkv[_i]*qkv[_i];
          fprintf(stderr,"[C] act0..3 %g %g %g %g sum128 %g\n", qkv[0], qkv[1], qkv[2], qkv[3], _s);
        }
        if(_dbg_hyb && L==0&&pos<2){ fprintf(stderr,"[C] tok%d conv0=%g %g %g\n",pos,qkv[0],qkv[1],qkv[2]); }
      }

      /* L2 norm de q/k por cabeza k */
      for(i32 h=0;h<nk;h++){
        l2norm(qkv+(size_t)h*dk,dk,c->eps);
        l2norm(qkv+(size_t)kdim+(size_t)h*dk,dk,c->eps);
      }
      if(_dbg_hyb && L==0&&pos<2){
        double dot=0; for(i32 i=0;i<dk;i++) dot+= (double)qkv[i]*qkv[kdim+i];
        fprintf(stderr,"[C] q0 %g k0 %g v0 %g dot %g\n", qkv[0], qkv[kdim], qkv[2*kdim], dot);
      }
      { /* recurrencia delta rule por v-head; out → wqo (libre aquí) */
        f32 *o=gdno; /* reuso: activado qkv ya consumido al final del bucle */
        const f32 *qc=qkv, *kc=qkv+kdim, *vc=qkv+2*kdim;
        const f32 sc=1.f/sqrtf((f32)dv);
        for(i32 h=0;h<nv;h++){
          const f32 *kh=kc+(size_t)(h%nk)*dk;
          const f32 *qh=qc+(size_t)(h%nk)*dk;
          const f32 *vh=vc+(size_t)h*dv;
          f32 *S=m->ssm_st+((size_t)recL*nv+(size_t)h)*(size_t)dv*(size_t)dv;
          f32 dec=expf(g_v[h]);
          f32 delta[512];
          if((i32)dv>512){ fprintf(stderr,"fwd hybrid: dv=%d >512\n",dv); return -1; }
          for(i32 j=0;j<dv;j++){ f32 *sr=S+(size_t)j*dv; for(i32 i=0;i<dv;i++) sr[i]*=dec; }
          for(i32 j=0;j<dv;j++){
            const f32 *sr=S+(size_t)j*dv; f32 sum=0;
            for(i32 i=0;i<dv;i++) sum+=sr[i]*kh[i];
            delta[j]=(vh[j]-sum)*beta_v[h];
          }
          for(i32 j=0;j<dv;j++){
            f32 *sr=S+(size_t)j*dv; f32 dd=delta[j];
            for(i32 i=0;i<dv;i++) sr[i]+=kh[i]*dd;
          }
          f32 *oh=o+(size_t)h*dv;
          for(i32 j=0;j<dv;j++){
            const f32 *sr=S+(size_t)j*dv; f32 sum=0;
            for(i32 i=0;i<dv;i++) sum+=sr[i]*qh[i];
            oh[j]=sum*sc;
          }
        }
      }
      /* gated rmsnorm: (rmsnorm(o)*w) * silu(z) → resultado en qkv (libre ahora) */
      load_vec_f32(m,wn,row,dv);
      for(i32 h=0;h<nv;h++){
        f32 *oh=gdno+(size_t)h*dv;
        rmsnorm(oh,oh,row,dv,c->eps);
        for(i32 j=0;j<dv;j++) oh[j]*=row[j];
      }
      for(i32 i=0;i<c->ssm_inner;i++){ f32 zz=z[i]; z[i]=zz/(1.f+expf(-zz)); }
      for(i32 i=0;i<c->ssm_inner;i++) qkv[i]=gdno[i]*z[i];
      if(_dbg_hyb && L==0&&pos<2){ fprintf(stderr,"[C] tok%d beta0=%g g0=%g qkva0=%g %g %g | gated0=%g %g %g\n",pos,
        beta_v[0],g_v[0],qkv[0],qkv[1],qkv[2],gdno[0],gdno[1],gdno[2]); }

      matmul_q(xb,qkv,slot_ptr(m,wo),wo->type,c->ssm_inner,dim,row);
      for(i32 i=0;i<dim;i++) x[i]+=xb[i];
      if(_dbg_hyb && L==0&&pos<2){ fprintf(stderr,"[C] tok%d qkv0=%g %g %g | o0=%g %g %g | x0=%g %g %g\n",pos,
        qkv[0],qkv[1],qkv[2], gdno[0],gdno[1],gdno[2], x[0],x[1],x[2]); }
      recL++;
    }

    /* FFN común (post_attention_norm mapeada a R_FFN_NORM) */
    Slot *fn=slot_get(m,R_FFN_NORM,L);
    load_vec_f32(m,fn,row,dim); rmsnorm(xb,x,row,dim,c->eps);
    Slot *wg=slot_get(m,R_FFN_GATE,L);
    Slot *wu=slot_get(m,R_FFN_UP,L);
    Slot *wd=slot_get(m,R_FFN_DOWN,L);
    if(require_slot(wg,"ffn_gate",L)||require_slot(wu,"ffn_up",L)||require_slot(wd,"ffn_down",L)) return -1;
    { /* hb/hb2 sobre el scratch (libre en este punto); cabe 2*hid ≤ scratch útil */
      f32 *hb=hs, *hb2=hs+hid;
      matmul_q(hb ,xb,slot_ptr(m,wg),wg->type,dim,hid,row);
      matmul_q(hb2,xb,slot_ptr(m,wu),wu->type,dim,hid,row);
      silu_mul(hb,hb2,hid);
      matmul_q(xb,hb,slot_ptr(m,wd),wd->type,hid,dim,row);
    }
    for(i32 i=0;i<dim;i++) x[i]+=xb[i];
  }

  Slot *on=slot_get(m,R_OUT_NORM,-1);
  load_vec_f32(m,on,row,dim); rmsnorm(x,x,row,dim,c->eps);
  if(!want_logits||!logits) return 0;
  Slot *out=slot_get(m,R_OUTPUT,-1);
  if(!out) out=emb;
  if(vk_head_dual(logits,x,slot_ptr(m,out),out->type,dim,c->vocab)) return 0;
  matmul_q(logits,x,slot_ptr(m,out),out->type,dim,c->vocab,row);
  return 0;
}
