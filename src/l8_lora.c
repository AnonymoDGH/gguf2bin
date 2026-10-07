/* l8_lora.c — LoRA inference: alloc/apply/save/load (Fase 7).
 * Solo aritmética real sobre pesos reales: SIN entrenamiento, SIN métricas.
 * El experimento de "entrenamiento" CYBER-mRNA (búsqueda estocástica de
 * perturbación, NO basada en gradiente; sus cifras de loss/accuracy eran
 * ilustrativas, no medidas) vive en experimental/cyber-mrna/ y NO se compila. */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static void lora_rand(f32 *p, size_t n){ for(size_t i=0;i<n;i++) p[i]= ((float)rand()/RAND_MAX*2-1)*0.02f; }
static int lora_alloc(Model *m, int r){
 if(r<=0) return -1;
 if(m->lora_r==r) return 0;
 if(m->lora_r){ /* re-alloc para distinto rank */
  for(int L=0; L<m->c.n_layers; L++){ free(m->loraA_q[L]); free(m->loraB_q[L]); free(m->loraA_v[L]); free(m->loraB_v[L]); free(m->loraA_gate[L]); free(m->loraB_gate[L]); free(m->loraM_q[L]); free(m->loraM_v[L]); free(m->loraM_gate[L]); free(m->galore_m[L]); free(m->galore_v[L]); }
  free(m->loraA_q); free(m->loraB_q); free(m->loraA_v); free(m->loraB_v); free(m->loraA_gate); free(m->loraB_gate); free(m->loraM_q); free(m->loraM_v); free(m->loraM_gate); free(m->galore_m); free(m->galore_v);
  m->lora_r=0; m->loraA_q=NULL;
 }
 m->lora_r=r;
 int L=m->c.n_layers;
 m->loraA_q=calloc(L,sizeof(f32*)); m->loraB_q=calloc(L,sizeof(f32*));
 m->loraA_v=calloc(L,sizeof(f32*)); m->loraB_v=calloc(L,sizeof(f32*));
 m->loraA_gate=calloc(L,sizeof(f32*)); m->loraB_gate=calloc(L,sizeof(f32*));
 m->loraM_q=calloc(L,sizeof(f32*)); m->loraM_v=calloc(L,sizeof(f32*)); m->loraM_gate=calloc(L,sizeof(f32*));
 m->galore_m=calloc(L,sizeof(f32*)); m->galore_v=calloc(L,sizeof(f32*));
 if(!m->loraA_q) return -1;
 for(int l=0;l<L;l++){
  int dim=m->c.dim, hd=m->c.head_dim, nq=m->c.n_heads*hd, nkv=m->c.n_kv_heads*hd, hid=m->c.hidden_dim;
  m->loraA_q[l]=malloc((size_t)dim*r*sizeof(f32)); m->loraB_q[l]=malloc((size_t)r*nq*sizeof(f32));
  m->loraA_v[l]=malloc((size_t)dim*r*sizeof(f32)); m->loraB_v[l]=malloc((size_t)r*nkv*sizeof(f32));
  m->loraA_gate[l]=malloc((size_t)dim*r*sizeof(f32)); m->loraB_gate[l]=malloc((size_t)r*hid*sizeof(f32));
  m->loraM_q[l]=malloc((size_t)nq*sizeof(f32)); m->loraM_v[l]=malloc((size_t)nkv*sizeof(f32)); m->loraM_gate[l]=malloc((size_t)hid*sizeof(f32));
  m->galore_m[l]=calloc((size_t)r*hid,sizeof(f32)); m->galore_v[l]=calloc((size_t)r*hid,sizeof(f32));
  if(!m->loraA_q[l]||!m->loraB_q[l]) return -1;
  lora_rand(m->loraA_q[l], (size_t)dim*r); memset(m->loraB_q[l],0,(size_t)r*nq*4);
  lora_rand(m->loraA_v[l], (size_t)dim*r); memset(m->loraB_v[l],0,(size_t)r*nkv*4);
  lora_rand(m->loraA_gate[l], (size_t)dim*r); memset(m->loraB_gate[l],0,(size_t)r*hid*4);
  for(int i=0;i<nq;i++) m->loraM_q[l][i]=1.0f;
  for(int i=0;i<nkv;i++) m->loraM_v[l][i]=1.0f;
  for(int i=0;i<hid;i++) m->loraM_gate[l][i]=1.0f;
 }
 return 0;
}
void lora_add(f32 *out, const f32 *x, const f32 *A, const f32 *B, const f32 *M, int dim, int outdim, int r){
 if(!A||!B) return;
 f32 tmp[128]; if(r>128) r=128;
 for(int k=0;k<r;k++){ f32 s=0; for(int i=0;i<dim;i++) s+= x[i]*A[i*r+k]; tmp[k]=s; }
 for(int j=0;j<outdim;j++){ f32 s=0; for(int k=0;k<r;k++) s+= tmp[k]*B[k*outdim+j];
   float mag = M? M[j]:1.0f;
   float norm = 1.0f + fabsf(s)*0.01f;
   out[j]+= mag * s / norm;
 }
}
int cyber_save_lora(Model *m, const char *path){
 if(!m||!path||!m->lora_r) return -1;
 FILE *f=fopen(path,"wb"); if(!f) return -1;
 int ver=2; fwrite(&ver,4,1,f); fwrite(&m->lora_r,4,1,f); fwrite(&m->c.n_layers,4,1,f);
 for(int l=0;l<m->c.n_layers;l++){
  int dim=m->c.dim, hid=m->c.hidden_dim, nq=m->c.n_heads*m->c.head_dim, nkv=m->c.n_kv_heads*m->c.head_dim, r=m->lora_r;
  fwrite(m->loraA_q[l],4,(size_t)dim*r,f); fwrite(m->loraB_q[l],4,(size_t)r*nq,f);
  fwrite(m->loraA_v[l],4,(size_t)dim*r,f); fwrite(m->loraB_v[l],4,(size_t)r*nkv,f);
  fwrite(m->loraA_gate[l],4,(size_t)dim*r,f); fwrite(m->loraB_gate[l],4,(size_t)r*hid,f);
  fwrite(m->loraM_q[l],4,(size_t)nq,f); fwrite(m->loraM_v[l],4,(size_t)nkv,f); fwrite(m->loraM_gate[l],4,(size_t)hid,f);
 }
 fclose(f); fprintf(stderr,"lora: saved v2 %s r=%d\n",path,m->lora_r); return 0;
}
static void lora_free(Model *m){
 if(!m->lora_r) return;
 f32 **arr[]={m->loraA_q,m->loraB_q,m->loraA_v,m->loraB_v,m->loraA_gate,m->loraB_gate,
              m->loraM_q,m->loraM_v,m->loraM_gate,m->galore_m,m->galore_v};
 for(size_t a=0;a<sizeof arr/sizeof arr[0];a++)
  if(arr[a]) for(int L=0; L<m->c.n_layers; L++) free(arr[a][L]); /* alloc parcial: NULL ok */
 free(m->loraA_q); free(m->loraB_q); free(m->loraA_v); free(m->loraB_v); free(m->loraA_gate); free(m->loraB_gate); free(m->loraM_q); free(m->loraM_v); free(m->loraM_gate); free(m->galore_m); free(m->galore_v);
 m->loraA_q=m->loraB_q=m->loraA_v=m->loraB_v=m->loraA_gate=m->loraB_gate=NULL;
 m->loraM_q=m->loraM_v=m->loraM_gate=m->galore_m=m->galore_v=NULL;
 m->lora_r=0;
}
static int rdf(f32 *p, size_t n, FILE *f){ return p && fread(p,4,n,f)==n ? 0 : -1; }
int cyber_load_lora(Model *m, const char *path){
 FILE *f=fopen(path,"rb"); if(!f) return -1;
 int ver, r, L;
 if(fread(&ver,4,1,f)!=1){ fclose(f); return -1; }
 if(ver==2){ if(fread(&r,4,1,f)!=1 || fread(&L,4,1,f)!=1){ fclose(f); return -1; } }
 else { r=ver; if(fread(&L,4,1,f)!=1){ fclose(f); return -1; } ver=1; }
 if(L!=m->c.n_layers){ fclose(f); return -1; }
 /* lora_add usa un tmp[128]: un rank mayor indexaría A con stride erróneo */
 if(r<=0 || r>128){ fprintf(stderr,"lora: rank %d fuera de [1,128]\n",r); fclose(f); return -1; }
 if(lora_alloc(m,r)){ lora_free(m); fclose(f); return -1; }
 int bad=0;
 for(int l=0;l<L && !bad;l++){
  size_t dim=(size_t)m->c.dim, hid=(size_t)m->c.hidden_dim, nq=(size_t)m->c.n_heads*m->c.head_dim, nkv=(size_t)m->c.n_kv_heads*m->c.head_dim, rr=(size_t)r;
  bad = rdf(m->loraA_q[l],dim*rr,f) || rdf(m->loraB_q[l],rr*nq,f)
     || rdf(m->loraA_v[l],dim*rr,f) || rdf(m->loraB_v[l],rr*nkv,f)
     || rdf(m->loraA_gate[l],dim*rr,f) || rdf(m->loraB_gate[l],rr*hid,f);
  if(bad || !m->loraM_q[l] || !m->loraM_v[l] || !m->loraM_gate[l]){ bad=1; break; }
  if(ver==1){
   for(size_t i=0;i<nq;i++) m->loraM_q[l][i]=1.0f;
   for(size_t i=0;i<nkv;i++) m->loraM_v[l][i]=1.0f;
   for(size_t i=0;i<hid;i++) m->loraM_gate[l][i]=1.0f;
  } else {
   bad = rdf(m->loraM_q[l],nq,f) || rdf(m->loraM_v[l],nkv,f) || rdf(m->loraM_gate[l],hid,f);
  }
 }
 fclose(f);
 if(bad){ fprintf(stderr,"lora: %s truncado o incompatible\n",path); lora_free(m); return -1; }
 fprintf(stderr,"lora: loaded v%d %s r=%d\n",ver,path,r); return 0;
}
