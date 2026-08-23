/* L4 — GGUF → G2BX (formato propio denso, indexado por rol) — FIXED v3.3 */
#include "g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int parse_name(const char *name, u8 *role, u16 *layer){
  *layer=0xFFFF;
  if(!strcmp(name,"token_embd.weight")){ *role=R_TOK_EMBD; return 0; }
  if(!strcmp(name,"output_norm.weight")){ *role=R_OUT_NORM; return 0; }
  if(!strcmp(name,"output.weight")){ *role=R_OUTPUT; return 0; }
  if(strncmp(name,"blk.",4)) return -1;
  int L=0; const char *p=name+4;
  while(*p>='0'&&*p<='9'){ L=L*10+(*p-'0'); p++; }
  if(*p!='.') return -1; p++;
  *layer=(u16)L;
  if(!strcmp(p,"attn_norm.weight")){ *role=R_ATTN_NORM; return 0; }
  if(!strcmp(p,"attn_q.weight")){ *role=R_ATTN_Q; return 0; }
  if(!strcmp(p,"attn_k.weight")){ *role=R_ATTN_K; return 0; }
  if(!strcmp(p,"attn_v.weight")){ *role=R_ATTN_V; return 0; }
  if(!strcmp(p,"attn_output.weight")){ *role=R_ATTN_O; return 0; }
  if(!strcmp(p,"attn_q_norm.weight")){ *role=R_ATTN_Q_NORM; return 0; }
  if(!strcmp(p,"attn_k_norm.weight")){ *role=R_ATTN_K_NORM; return 0; }
  if(!strcmp(p,"attn_q.bias")){ *role=R_ATTN_Q_BIAS; return 0; }
  if(!strcmp(p,"attn_k.bias")){ *role=R_ATTN_K_BIAS; return 0; }
  if(!strcmp(p,"attn_v.bias")){ *role=R_ATTN_V_BIAS; return 0; }
  if(!strcmp(p,"ffn_norm.weight")){ *role=R_FFN_NORM; return 0; }
  if(!strcmp(p,"ffn_gate.weight")){ *role=R_FFN_GATE; return 0; }
  if(!strcmp(p,"ffn_up.weight")){ *role=R_FFN_UP; return 0; }
  if(!strcmp(p,"ffn_down.weight")){ *role=R_FFN_DOWN; return 0; }
  return -1;
}
static i64 meta_pref(GGUF *g, const char *arch, const char *suf){
  char k[96];
  const char *prefs[4]; int np=0;
  if(arch && arch[0]) prefs[np++]=arch;
  if(!arch || (strcmp(arch,"llama")&&strcmp(arch,"qwen2")&&strcmp(arch,"qwen2moe")&&strcmp(arch,"qwen3"))) { prefs[np++]="qwen3"; prefs[np++]="qwen2"; }
  prefs[np++]="llama";
  for(int i=0;i<np;i++){ snprintf(k,sizeof k,"%s.%s",prefs[i],suf); i64 v=gguf_meta_i64(g,k); if(v) return v; }
  return 0;
}
static f32 meta_preff(GGUF *g, const char *arch, const char *suf){
  char k[96];
  const char *prefs[4]; int np=0;
  if(arch && arch[0]) prefs[np++]=arch;
  if(!arch || (strcmp(arch,"llama")&&strcmp(arch,"qwen2")&&strcmp(arch,"qwen2moe")&&strcmp(arch,"qwen3"))) { prefs[np++]="qwen3"; prefs[np++]="qwen2"; }
  prefs[np++]="llama";
  for(int i=0;i<np;i++){ snprintf(k,sizeof k,"%s.%s",prefs[i],suf); f32 v=gguf_meta_f32(g,k); if(v!=0.f) return v; }
  return 0;
}
static int read_cfg(GGUF *g, ModelCfg *c, u8 *arch, u8 *flags){
  memset(c,0,sizeof *c); *flags=0;
  char aname[64]={0}; gguf_meta_str(g,"general.architecture",aname,sizeof aname);
  /* Stub bailingmoe3 / MLA+MoE+SSM: no soportado en runtime denso Qwen/Llama */
  if(aname[0] && (!strcmp(aname,"bailingmoe3") || !strcmp(aname,"bailingmoe") || strstr(aname,"moe3") || strstr(aname,"ssm"))){
    fprintf(stderr,"g2bx: arquitectura '%s' no soportada (MoE %lld expertos + MLA + SSM). Solo Qwen3/Qwen2/Llama densos.\n",
      aname, (long long)gguf_meta_i64(g,"bailingmoe3.expert_count"));
    fprintf(stderr,"g2bx: Ling-3.0-tiny usa MLA (q_lora/kv_lora) + 128 expertos + IQ1_S; requiere llama.cpp. Archivado en D:\\gguf2bin_models\\\n");
    return -1;
  }
  /* Arquitecturas conocidas LLM/Qwen; cualquier otra (p.ej. dflash) se deduce por heurística */
  const int is_llama=!strcmp(aname,"llama");
  const int is_qwen =!strcmp(aname,"qwen2")||!strcmp(aname,"qwen2moe")||!strcmp(aname,"qwen3");
  if(is_llama) *arch=ARCH_LLAMA;
  else {
    *arch = is_qwen ? ARCH_QWEN2 : ARCH_QWEN3; /* custom -> NEOX/Qwen por defecto */
    if(!is_qwen && aname[0]){
      int looks_qwen = gguf_by_name(g,"blk.0.attn_q_norm.weight")!=NULL
                    || gguf_by_name(g,"blk.0.attn_k_norm.weight")!=NULL
                    || meta_pref(g,aname,"attention.key_length")>0;
      if(!looks_qwen) *arch=ARCH_LLAMA;
    }
  }
  const char *p = aname[0]? aname : (*arch==ARCH_LLAMA?"llama":*arch==ARCH_QWEN2?"qwen2":"qwen3");
  c->dim        =(i32)meta_pref(g,p,"embedding_length");
  c->hidden_dim =(i32)meta_pref(g,p,"feed_forward_length");
  c->n_layers   =(i32)meta_pref(g,p,"block_count");
  c->n_heads    =(i32)meta_pref(g,p,"attention.head_count");
  c->n_kv_heads =(i32)meta_pref(g,p,"attention.head_count_kv");
  c->vocab      =(i32)gguf_meta_arr_len(g,"tokenizer.ggml.tokens");
  if(!c->vocab) c->vocab=(i32)meta_pref(g,p,"vocab_size");
  c->seq_len    =(i32)meta_pref(g,p,"context_length");
  c->eps        =meta_preff(g,p,"attention.layer_norm_rms_epsilon");
  if(c->eps==0.f) c->eps=1e-6f;
  c->head_dim   =(i32)meta_pref(g,p,"attention.key_length");
  if(!c->head_dim && c->n_heads) c->head_dim=c->dim/c->n_heads;
  c->rope_theta =meta_preff(g,p,"rope.freq_base");
  if(c->rope_theta==0.f) c->rope_theta=(*arch==ARCH_LLAMA)?10000.f:1000000.f;
  if(!c->n_kv_heads) c->n_kv_heads=c->n_heads;
  if(!c->vocab){
    GTensor *e=gguf_by_name(g,"token_embd.weight");
    if(e && e->n_dims>=2){ u64 a=e->dims[0], b=e->dims[1]; c->vocab=(i32)(a>b?a:b); if(!c->dim) c->dim=(i32)(a<b?a:b); }
  }
  if(!gguf_by_name(g,"output.weight")) *flags|=F_TIE_EMBD;
  if(gguf_by_name(g,"blk.0.attn_q_norm.weight") || gguf_by_name(g,"blk.0.attn_k_norm.weight")) *flags|=F_QK_NORM;
  else if(*arch!=ARCH_LLAMA) *flags|=F_QK_NORM;
  if(c->dim<=0||c->n_layers<=0||c->n_heads<=0){ fprintf(stderr,"g2bx: cfg incompleta arch=%s dim=%d layers=%d\n",aname,c->dim,c->n_layers); return -1; }
  if(c->seq_len<=0) c->seq_len=2048;
  if(c->seq_len>32768) c->seq_len=32768; /* cap razonable; --ctx lo baja en runtime */
  return 0;
}
static u64 ne_of(const GTensor *t){ u64 n=1; for(u32 i=0;i<t->n_dims;i++) n*=t->dims[i]; return n; }
static int is_weight_role(u8 role){
  switch(role){ case R_TOK_EMBD: case R_OUTPUT: case R_ATTN_Q: case R_ATTN_K: case R_ATTN_V: case R_ATTN_O: case R_FFN_GATE: case R_FFN_UP: case R_FFN_DOWN: return 1; default: return 0; }
}
static void quant_block_q4_0(const f32 *x, u8 *dst){
  f32 amax=0; for(int i=0;i<32;i++){ f32 a=fabsf(x[i]); if(a>amax) amax=a; } f32 d=amax/7.0f; if(d<=0) d=1e-9f; u16 sd=f32_to_half(d); memcpy(dst,&sd,2);
  for(int j=0;j<16;j++){ int q0=(int)lroundf(x[j]/d)+8; int q1=(int)lroundf(x[j+16]/d)+8; if(q0<0)q0=0; else if(q0>15)q0=15; if(q1<0)q1=0; else if(q1>15)q1=15; dst[2+j]=(u8)(q0|(q1<<4)); }
}
static u8 *convert_tensor_q4_0(const u8 *src, u32 srctype, u64 ne){
  if(ne%32) return NULL;
  u64 nb=ne/32; u8 *out=malloc(nb*18); if(!out) return NULL;
  u64 bs = ggml_block_size(srctype); /* 1=F32/F16, 32=Q4/Q8, 256=K-quants */
  if(bs<=1){
    u64 st = ggml_type_bytes(srctype)*32;
    f32 blk[32];
    for(u64 b=0;b<nb;b++){ gguf_dequant(srctype,(u8*)(src+b*st),blk,32); quant_block_q4_0(blk,out+b*18); }
    return out;
  }
  /* bloques de bs elementos (32 o 256): dequant el bloque entero y re-cuantizar en trozos de 32 */
  f32 blk[256];
  u64 src_bytes = ggml_type_bytes(srctype);
  u64 d=0;
  for(u64 sb=0; sb<ne/bs; sb++){
    gguf_dequant(srctype,(u8*)(src+sb*src_bytes),blk,bs);
    for(u64 j=0;j<bs;j+=32){ quant_block_q4_0(blk+j,out+d); d+=18; }
  }
  return out;
}
int g2bx_pack(const char *gguf_path, const char *out_path){
  return g2bx_pack_prune(gguf_path, out_path, 0, 0.f);
}
int g2bx_pack_ex(const char *gguf_path, const char *out_path, int downq4){
  return g2bx_pack_prune(gguf_path, out_path, downq4, 0.f);
}

/* ── Poda estructurada FFN (efecto MoE sin reentrenar) ──
   Con calib_scores [n_layers x hidden] (de model_collect_stats) usa importancia
   real por activación; si es NULL, proxy Σ|gate·up| por pesos.
   Se eliminan grupos de G neuronas alineados a bloques de cuantización. */
int g2bx_pack_prune(const char *gguf_path, const char *out_path, int downq4, float prune){
  return g2bx_pack_prune_scores(gguf_path,out_path,downq4,prune,NULL);
}
int g2bx_pack_prune_scores(const char *gguf_path, const char *out_path,
                           int downq4, float prune, const f32 *calib){
  GGUF g; if(gguf_load(gguf_path,&g)) return -1;
  ModelCfg c; u8 arch, flags; if(read_cfg(&g,&c,&arch,&flags)){ gguf_free(&g); return -1; }
  if(!(prune>0.001f) || prune>=0.9f || c.hidden_dim<=0){ prune=0.f; }
  Slot *slots=calloc(g.n_tensors,sizeof(Slot)); u32 ns=0;
  u64 *ne_arr=calloc(g.n_tensors,sizeof(u64)); u8 **src_ptr=calloc(g.n_tensors,sizeof(u8*)); u32 *src_sz=calloc(g.n_tensors,sizeof(u32)); u8 **conv_ptr=calloc(g.n_tensors,sizeof(u8*));
  if(!slots||!ne_arr||!src_ptr||!src_sz||!conv_ptr){
    fprintf(stderr,"g2bx: OOM\n"); free(slots); free(ne_arr); free(src_ptr); free(src_sz); free(conv_ptr); gguf_free(&g); return -1;
  }
  u32 skipped_unknown=0, skipped_type=0;
  for(u64 i=0;i<g.n_tensors;i++){
    u8 role; u16 layer; if(parse_name(g.t[i].name,&role,&layer)){ skipped_unknown++; continue; }
    u32 ty=g.t[i].type;
    if(ty!=T_F32 && ty!=T_F16 && ty!=T_Q4_0 && ty!=T_Q4_1 && ty!=T_Q5_0 && ty!=T_Q8_0
     && ty!=T_Q2_K && ty!=T_Q3_K && ty!=T_Q4_K && ty!=T_Q5_K && ty!=T_Q6_K && ty!=T_Q8_K){
      fprintf(stderr,"g2bx: tipo %u no soportado en tensor %s (omitido)\n", ty, g.t[i].name);
      skipped_type++; continue;
    }
    u64 ne=ne_of(&g.t[i]); u32 nbytes=(u32)ggml_type_size(g.t[i].type,ne);
    slots[ns].role=role; slots[ns].layer=layer; slots[ns].type=(u8)g.t[i].type; slots[ns].nbytes=nbytes;
    src_ptr[ns]=gguf_tensor_ptr(&g,&g.t[i]); src_sz[ns]=nbytes; ne_arr[ns]=ne; ns++;
  }
  if(skipped_type)
    fprintf(stderr,"g2bx: %u tensores omitidos por tipo no soportado\n", skipped_type);
  if(!ns){ fprintf(stderr,"g2bx: 0 tensores reconocidos\n"); free(slots); free(src_ptr); free(src_sz); free(ne_arr); free(conv_ptr); gguf_free(&g); return -1; }
  { int has_embd=0; for(u32 i=0;i<ns;i++) if(slots[i].role==R_TOK_EMBD){ has_embd=1; break; }
    if(!has_embd)
      fprintf(stderr,"g2bx: AVISO — sin token_embd.weight: modelo draft/decode-only\n");
  }
  for(u32 a=0;a<ns;a++) for(u32 b=a+1;b<ns;b++){
    int la=slots[a].layer==0xFFFF?-1:(int)slots[a].layer; int lb=slots[b].layer==0xFFFF?-1:(int)slots[b].layer;
    if(lb<la || (lb==la && slots[b].role<slots[a].role)){
      Slot ts=slots[a]; slots[a]=slots[b]; slots[b]=ts; u8 *tp=src_ptr[a]; src_ptr[a]=src_ptr[b]; src_ptr[b]=tp; u32 tz=src_sz[a]; src_sz[a]=src_sz[b]; src_sz[b]=tz; u64 tn=ne_arr[a]; ne_arr[a]=ne_arr[b]; ne_arr[b]=tn;
    }
  }

  /* ── poda FFN por capas ── */
  if(prune>0.f){
    const i32 dim=c.dim, hidden=c.hidden_dim;
    u32 G=32;
    for(i32 L=0;L<c.n_layers && G==32;L++){
      /* granularidad = mayor block_size entre los tres tensores */
      for(u32 i=0;i<ns;i++){
        if(slots[i].layer==(u16)L &&
           (slots[i].role==R_FFN_GATE||slots[i].role==R_FFN_UP||slots[i].role==R_FFN_DOWN)){
          u64 bs=ggml_block_size(slots[i].type);
          if(bs>G) G=(u32)bs;
        }
      }
    }
    if(hidden%G || dim%G){
      fprintf(stderr,"prune: hidden=%d no alineado a %u — poda desactivada\n",hidden,G);
      prune=0.f;
    } else {
      u32 nblk_total=hidden/G;
      u32 keep=nblk_total-(u32)(nblk_total*prune+0.5f);
      if(keep<2) keep=2;
      u8 *kept=malloc((size_t)hidden);
      f32 *grow=malloc((size_t)dim*sizeof(f32)), *urw=malloc((size_t)dim*sizeof(f32));
      u64 *bscore=malloc((size_t)nblk_total*sizeof(u64));
      u32 *order=malloc((size_t)nblk_total*sizeof(u32));
      if(!kept||!grow||!urw||!bscore||!order){
        fprintf(stderr,"prune: OOM — desactivada\n"); prune=0.f;
      } else {
        i32 new_hidden=(i32)(keep*(u32)G);
        u64 old_bytes=0,new_bytes=0;
        for(i32 L=0;L<c.n_layers;L++){
          Slot *sg=NULL,*su=NULL,*sd=NULL;
          for(u32 i=0;i<ns;i++){
            if(slots[i].layer!=(u16)L) continue;
            if(slots[i].role==R_FFN_GATE) sg=&slots[i];
            else if(slots[i].role==R_FFN_UP) su=&slots[i];
            else if(slots[i].role==R_FFN_DOWN) sd=&slots[i];
          }
          if(!sg||!su||!sd) continue;
          i32 ig=-1,iu=-1,id_=-1;
          for(u32 i=0;i<ns;i++){
            if(&slots[i]==sg) ig=(i32)i;
            else if(&slots[i]==su) iu=(i32)i;
            else if(&slots[i]==sd) id_=(i32)i;
          }
          const u8 *gp=src_ptr[ig], *up=src_ptr[iu], *dp=src_ptr[id_];
          u32 tg=sg->type, tu=su->type, td=sd->type;
          size_t grs=ggml_type_size(tg,(u64)dim), urs=ggml_type_size(tu,(u64)dim);
          size_t drs_old=ggml_type_size(td,(u64)hidden), drb=ggml_type_bytes(td);
          /* scoring: activaciones calibradas (preferente) o proxy de pesos */
          if(calib){
            for(u32 k=0;k<nblk_total;k++){
              u64 acc=0;
              for(u32 j=0;j<G;j++){
                f32 v=calib[(size_t)L*(size_t)hidden + (size_t)k*G+j];
                acc+=(u64)(v*65536.f);
              }
              bscore[k]=acc;
            }
          } else {
            memset(bscore,0,(size_t)nblk_total*sizeof(u64));
            for(i32 hrow=0;hrow<hidden;hrow++){
              gguf_dequant(tg,gp+(size_t)hrow*grs,grow,(u64)dim);
              gguf_dequant(tu,up+(size_t)hrow*urs,urw,(u64)dim);
              u64 acc=bscore[hrow/G];
              for(i32 j=0;j<dim;j++) acc+=(u64)(fabsf(grow[j]*urw[j])*4096.f);
              bscore[hrow/G]=acc;
            }
          }
          for(u32 k=0;k<nblk_total;k++) order[k]=k;
          /* selección top-keep por score (inserción parcial simple: qsort) */
          for(u32 a=0;a+1<nblk_total;a++){ /* insertion sort descendente por score */
            u32 best=a;
            for(u32 b2=a+1;b2<nblk_total;b2++) if(bscore[order[b2]]>bscore[order[best]]) best=b2;
            if(best!=a){ u32 t=order[a]; order[a]=order[best]; order[best]=t; }
            if(a>=keep) break; /* solo se necesitan los keep primeros */
          }
          memset(kept,0,(size_t)hidden);
          for(u32 k=0;k<keep;k++){ u32 blk=order[k]; for(u32 j=0;j<G;j++) kept[blk*G+j]=1; }
          /* rebuild gate/up: copia de runs de filas */
          u8 *ng=malloc((size_t)new_hidden*grs), *nu=malloc((size_t)new_hidden*urs);
          u8 *nd=malloc((size_t)dim*(size_t)new_hidden/G*drb);
          if(!ng||!nu||!nd){ fprintf(stderr,"prune: OOM en capa %d\n",L); free(ng);free(nu);free(nd); break; }
          /* gate/up: las filas kept se copian por runs contiguos */
          { u32 i=0; u32 dst=0;
            while(i<(u32)hidden){
              while(i<(u32)hidden && !kept[i]) i++;
              if(i>=(u32)hidden) break;
              u32 r=i; while(r<(u32)hidden && kept[r]) r++;
              memcpy(ng+(size_t)dst*grs, gp+(size_t)i*grs, (size_t)(r-i)*grs);
              memcpy(nu+(size_t)dst*urs, up+(size_t)i*urs, (size_t)(r-i)*urs);
              dst+=r-i; i=r;
            }
          }
          /* down [dim,hidden]: por fila, copiar los bloques kept del eje interno */
          { u32 nb_in=(u32)hidden/G;
            for(i32 r=0;r<dim;r++){
              const u8 *srow=dp+(size_t)r*drs_old;
              u8 *drow=nd+(size_t)r*((size_t)keep*drb);
              u32 dst=0;
              for(u32 k=0;k<nb_in;k++) if(kept[(size_t)k*G]) { memcpy(drow+(size_t)dst*drb, srow+(size_t)k*drb, drb); dst++; }
            }
          }
          conv_ptr[ig]=ng; src_ptr[ig]=ng; src_sz[ig]=(u32)((size_t)new_hidden*grs);
          ne_arr[ig]=(u64)new_hidden*(u64)dim;
          sg->nbytes=src_sz[ig];
          conv_ptr[iu]=nu; src_ptr[iu]=nu; src_sz[iu]=(u32)((size_t)new_hidden*urs);
          ne_arr[iu]=(u64)new_hidden*(u64)dim;
          su->nbytes=src_sz[iu];
          conv_ptr[id_]=nd; src_ptr[id_]=nd; src_sz[id_]=(u32)((size_t)dim*(size_t)new_hidden/G*drb);
          ne_arr[id_]=(u64)dim*(u64)new_hidden;
          sd->nbytes=src_sz[id_];
          old_bytes+=grs*(size_t)hidden+urs*(size_t)hidden+drs_old*(size_t)dim;
          new_bytes+=grs*(size_t)new_hidden+urs*(size_t)new_hidden+(size_t)dim*(size_t)new_hidden/G*drb;
        }
        c.hidden_dim=new_hidden;
        fprintf(stderr,"prune: FFN %d -> %d (-%.0f%%): pesos %.0f MB -> %.0f MB\n",
          hidden,new_hidden,100.f*prune,old_bytes/1048576.f,new_bytes/1048576.f);
        free(kept); free(grow); free(urw); free(bscore); free(order);
      }
    }
  }

  /* FIX v3.4: F32/F16->Q4_0 siempre; con --q4 TODOS los tensores de peso
     se convierten a Q4_0 fusionado para el kernel SIMD rápido. */
  for(u32 i=0;i<ns;i++){
    int conv = 0;
    if(is_weight_role(slots[i].role)){
      if(slots[i].type==T_F32 || slots[i].type==T_F16) conv = 1;
      else if(downq4 && slots[i].type!=T_Q4_0) conv = 1;
    }
    if(conv){
      u8 *q4=convert_tensor_q4_0(src_ptr[i], slots[i].type, ne_arr[i]);
      if(q4){ conv_ptr[i]=q4; src_ptr[i]=q4; slots[i].type=T_Q4_0; slots[i].nbytes=(u32)(ne_arr[i]/32*18); src_sz[i]=slots[i].nbytes; }
    }
  }
  u64 cursor=0; for(u32 i=0;i<ns;i++){ slots[i].off=cursor; cursor=ALIGN64(cursor+slots[i].nbytes); } u64 data_size=cursor;
  FILE *o=fopen(out_path,"wb"); if(!o){ free(slots); free(src_ptr); free(src_sz); free(ne_arr); free(conv_ptr); gguf_free(&g); return -1; }
  u16 ver=G2BX_VER; u8 disk_flags = (u8)(flags & ~F_KV_Q8); /* F_KV_Q8 runtime, no on-disk */
  int wr=1; if(fwrite(G2BX_MAGIC,1,4,o)!=4||fwrite(&ver,2,1,o)!=1||fwrite(&arch,1,1,o)!=1||fwrite(&disk_flags,1,1,o)!=1||fwrite(&c,sizeof c,1,o)!=1||fwrite(&ns,4,1,o)!=1) wr=0;
  wr &= fwrite(slots,sizeof(Slot),ns,o)==ns;
  u8 *zeros=calloc(64,1); if(!zeros){ free(slots); free(src_ptr); free(src_sz); free(ne_arr); free(conv_ptr); fclose(o); gguf_free(&g); return -1; }
  u64 written=0;
  for(u32 i=0;i<ns;i++){
    while(written<slots[i].off){ u64 pad=slots[i].off-written; if(pad>64) pad=64; fwrite(zeros,1,(size_t)pad,o); written+=pad; }
    wr &= fwrite(src_ptr[i],1,src_sz[i],o)==src_sz[i]; written+=src_sz[i];
  }
  while(written<data_size){ u64 pad=data_size-written; if(pad>64) pad=64; wr &= fwrite(zeros,1,(size_t)pad,o)==(size_t)pad; written+=pad; }
  free(zeros);
  Tokenizer tk; if(tok_from_gguf(&g,&tk)==0){ tok_write_section(o,&tk); fprintf(stderr,"  tokenizer: %d tokens, %d merges (bos=%d eos=%d)\n",tk.n,tk.nmerges,tk.bos,tk.eos); tok_free(&tk); } else fprintf(stderr,"  tokenizer: NO disponible\n");
  fclose(o);
  if(!wr){ fprintf(stderr,"g2bx: error de escritura en %s (disco lleno?)\n",out_path); }
  fprintf(stderr,"g2bx pack -> %s\n  arch=%u flags=0x%02x layers=%d dim=%d head_dim=%d kv=%d vocab=%d\n  slots=%u weight_bytes=%llu (GGUF era %llu)\n  rope_theta=%.0f qk_norm=%s tie_embd=%s\n",out_path,arch,flags,c.n_layers,c.dim,c.head_dim,c.n_kv_heads,c.vocab,ns,(unsigned long long)data_size,(unsigned long long)g.size,c.rope_theta,(flags&F_QK_NORM)?"yes":"no",(flags&F_TIE_EMBD)?"yes":"no");
  free(slots); free(src_ptr); free(src_sz); free(ne_arr); for(u32 i=0;i<ns;i++) if(conv_ptr[i]) free(conv_ptr[i]); free(conv_ptr); gguf_free(&g); return wr?0:-1;
}
int g2bx_info(const char *path){
  FILE *f=fopen(path,"rb"); if(!f){ fprintf(stderr,"g2bx: no abro %s\n",path); return -1; }
  char magic[4]; u16 ver; u8 arch,flags; ModelCfg c; u32 ns;
  if(fread(magic,1,4,f)!=4||memcmp(magic,G2BX_MAGIC,4)||fread(&ver,2,1,f)!=1||fread(&arch,1,1,f)!=1||fread(&flags,1,1,f)!=1||fread(&c,sizeof c,1,f)!=1||fread(&ns,4,1,f)!=1){ fprintf(stderr,"g2bx: cabecera invalida\n"); fclose(f); return -1; }
  static const char *an[]={"llama","qwen2","qwen3"}; printf("G2BX v%u arch=%s flags=0x%02x\n",ver, arch<3?an[arch]:"?",flags);
  printf("  dim=%d hidden=%d layers=%d heads=%d kv=%d head_dim=%d\n",c.dim,c.hidden_dim,c.n_layers,c.n_heads,c.n_kv_heads,c.head_dim);
  printf("  vocab=%d seq=%d eps=%g rope_theta=%.0f\n",c.vocab,c.seq_len,c.eps,c.rope_theta);
  printf("  slots=%u\n",ns); Slot *sl=malloc(ns*sizeof(Slot));
  if(!sl || fread(sl,sizeof(Slot),ns,f)!=ns){ free(sl); fclose(f); return -1; }
  static const char *rn[]={"tok_embd","out_norm","output","attn_norm","attn_q","attn_k","attn_v",
    "attn_o","attn_q_norm","attn_k_norm","ffn_norm","ffn_gate","ffn_up","ffn_down",
    "attn_q_bias","attn_k_bias","attn_v_bias"};
  u64 total=0; for(u32 i=0;i<ns;i++){ const char *r=sl[i].role<R_COUNT?rn[sl[i].role]:"?"; if(sl[i].layer==0xFFFF) printf("  [%u] %-12s global type=%u %u B off=%llu\n",i,r,sl[i].type,sl[i].nbytes,(unsigned long long)sl[i].off); else printf("  [%u] %-12s L%-4u type=%u %u B off=%llu\n",i,r,sl[i].layer,sl[i].type,sl[i].nbytes,(unsigned long long)sl[i].off); total+=sl[i].nbytes; }
  printf("weight_bytes=%llu file_data~%llu\n",(unsigned long long)total,(unsigned long long)(ns?sl[ns-1].off+sl[ns-1].nbytes:0)); free(sl); fclose(f); return 0;
}
int g2bx_read_cfg_gguf(GGUF *g, ModelCfg *c, u8 *arch, u8 *flags){ return read_cfg(g,c,arch,flags); }
