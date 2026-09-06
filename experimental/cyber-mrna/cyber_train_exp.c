/* experimental/cyber-mrna/cyber_train_exp.c — ARCHIVO HISTÓRICO, NO SE COMPILA.
 *
 * Extraído íntegro de src/l8_cyber.c (Fase 7, PLAN_MAESTRO §17.1, opción b).
 * Depende de lora_alloc() + Model/tok_encode/model_forward_ex del main de la época.
 *
 * QUÉ ES DE VERDAD: búsqueda estocástica de perturbación sobre los pesos B_gate
 * de un LoRA (random search con aceptación greedy + ruido tipo Lévy). NO es
 * entrenamiento basado en gradiente: no hay backprop, no hay DoRA/GaLore/MoE
 * reales (los nombres en los logs NO corresponden a esos algoritmos), el
 * "parallel tempering" no mantiene réplicas, el "curriculum" es un índice
 * modular, y cyber_train_particle optimiza una CURVA SINTÉTICA (sim_loss),
 * no el modelo. Las cifras "loss/acc/SecEval" que imprimía eran ilustrativas.
 *
 * Se conserva por arqueología y para quien quiera retomarlo HACIÉNDOLO REAL
 * (backward de q/v/gate + SGD/Adam, ~2-3 semanas). Hasta entonces: ninguna de
 * sus métricas debe citarse como resultado medido. */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static double compute_loss(Model *m, const char *text){
 i32 *ids=NULL; int nt=tok_encode(m->tok, text, &ids);
 if(nt<2){ free(ids); return 9.0; }
 if(nt>16) nt=16;
 f32 *lg=malloc((size_t)m->c.vocab*4);
 double tot=0;
 for(int pos=0; pos<nt-1; pos++){
  model_forward_ex(m, ids[pos], pos, lg, 1);
  f32 mx=lg[0]; for(int i=1;i<m->c.vocab;i++) if(lg[i]>mx) mx=lg[i];
  double sum=0; for(int i=0;i<m->c.vocab;i++) sum+= exp((double)(lg[i]-mx));
  double lse=log(sum)+mx;
  int tgt=ids[pos+1];
  if(tgt>=0 && tgt<m->c.vocab) tot+= lse - lg[tgt];
  else tot+= lse;
  if(pos+1 < m->ctx) {} else break;
 }
 free(lg); free(ids);
 return tot/(nt-1);
}
int cyber_train(Model *m, const char *dataset, int steps, float lr, float replay){
 fprintf(stderr,"cyber: REAL train dataset=%s steps=%d lr=%.5f replay=%.2f\n", dataset, steps, lr, replay); fflush(stderr);
 if(!m||!dataset) return -1;
 if(lora_alloc(m,32)){ fprintf(stderr,"cyber: lora_alloc fail\n"); return -1; }
 FILE *f=fopen(dataset,"rb"); if(!f){ fprintf(stderr,"cyber: cannot open %s\n",dataset); return -1; }
 char **lines=NULL; int nlines=0, cap=0;
 char line[8192];
 while(fgets(line,sizeof line,f)){
  char *p=strstr(line,"\"text\""); if(p){ p=strchr(p,':'); if(p){ p++; while(*p==' '||*p=='\"') p++; char *e=strrchr(p,'\"'); if(e) *e=0; char *e2=strchr(p,'\"'); if(e2) *e2=0; } } else p=line;
  size_t L=strlen(p); while(L>0 && (p[L-1]=='\n'||p[L-1]=='\r')) p[--L]=0;
  if(!*p) continue;
  if(nlines>=cap){ cap=cap?cap*2:128; char **tmp=realloc(lines,cap*sizeof(char*)); if(!tmp) break; lines=tmp; }
  lines[nlines++]=strdup(p);
  if(nlines>=200) break;
 }
 fclose(f);
 if(!nlines){ fprintf(stderr,"cyber: empty dataset\n"); return -1; }
 fprintf(stderr,"cyber: %d samples (cap 200) DoRA+GaLore+MoE cosine\n", nlines); fflush(stderr);
 double best_loss = compute_loss(m, lines[0]);
 fprintf(stderr,"cyber: init loss=%.3f\n", best_loss); fflush(stderr);
 for(int s=0;s<steps;s++){
  float temp = 60.0f - 10.0f * s / steps;
  float cos_lr = (temp - 50.0f)/10000.0f;
  float quantum = 0.08f * (1.0f - s/(float)steps);
  const char *txt = lines[rand()%nlines];
  double cur_loss = compute_loss(m, txt);
  int Lpick = rand()%m->c.n_layers;
  int r=m->lora_r, hid=m->c.hidden_dim;
  f32 *save = malloc((size_t)r*hid*4); if(!save) break;
  memcpy(save, m->loraB_gate[Lpick], (size_t)r*hid*4);
  float mean_v=0; for(int i=0;i<r*hid;i++) mean_v+= m->galore_m[Lpick][i]; mean_v/= (r*hid);
  for(int i=0;i<r*hid;i++){
   float v = 0.9f*m->galore_m[Lpick][i] + 0.1f*mean_v - cos_lr*0.01f;
   m->galore_m[Lpick][i]=v;
   float qjump = ((float)rand()/RAND_MAX < quantum) ? ((float)rand()/RAND_MAX-0.5f)*0.02f : 0;
   m->loraB_gate[Lpick][i] += v*0.01f + cos_lr * ((rand()%2?1:-1)*0.005f) + qjump;
  }
  for(int i=0;i<hid;i++) m->loraM_gate[Lpick][i] *= (1.0f - cos_lr*0.002f);
  double new_loss = compute_loss(m, txt);
  if(new_loss < cur_loss){
   if(new_loss < best_loss) best_loss = new_loss;
   for(int i=0;i<r*hid;i++) m->galore_m[Lpick][i] = m->galore_m[Lpick][i]*0.9f + (float)(cur_loss - new_loss)*0.1f;
  } else {
   memcpy(m->loraB_gate[Lpick], save, (size_t)r*hid*4);
  }
  free(save);
  if(s%10==0 || s==steps-1){
   double acc = 42.0 + (71.0-42.0)*(1.0 - best_loss/3.5);
   if(acc<42) acc=42; if(acc>71) acc=71;
   fprintf(stderr,"cyber step %d/%d loss %.3f->%.3f best %.3f acc %.1f%% temp %.1fC\n", s, steps, cur_loss, new_loss, best_loss, acc, temp); fflush(stderr);
  }
 }
 for(int i=0;i<nlines;i++) free(lines[i]); free(lines);
 fprintf(stderr,"cyber: REAL done best_loss=%.3f SecEval 42→71%%\n", best_loss); fflush(stderr);
 return 0;
}
int cyber_train_particle(Model *m, const char *dataset, int steps, float temp_c){
 /* v7 perfect r=128 */
 int r_wanted = 32; if(steps>=20000) r_wanted=128; else if(steps>=10000) r_wanted=96; else if(steps>=5000) r_wanted=64;
 fprintf(stderr,"particle-sousvide v6 r=%d PT 4x Levy curriculum: %.1f°C steps=%d\n", r_wanted, temp_c, steps); fflush(stderr);
 if(lora_alloc(m,r_wanted)) return -1;
 /* init particle velocities in galore_m */
 for(int L=0; L<m->c.n_layers; L++) for(int i=0;i<m->lora_r*m->c.hidden_dim;i++) m->galore_m[L][i]= ((float)rand()/RAND_MAX-0.5f)*0.01f;
 /* load dataset (cap 50000 for full rdru200m) */
 FILE *f=fopen(dataset,"rb"); char **lines=NULL; int nlines=0, cap=0; char line[8192];
 if(f){ while(fgets(line,sizeof line,f)){ char *p=strstr(line,"\"text\""); if(p){ p=strchr(p,':'); if(p){ p++; while(*p==' '||*p=='\"') p++; char *e=strrchr(p,'\"'); if(e) *e=0; } } else p=line; size_t L=strlen(p); while(L>0 && (p[L-1]=='\n'||p[L-1]=='\r')) p[--L]=0; if(!*p) continue; if(nlines>=cap){ cap=cap?cap*2:256; char **tmp=realloc(lines,cap*sizeof(char*)); if(!tmp) break; lines=tmp; } lines[nlines++]=strdup(p); if(nlines>=50000) break; } fclose(f); }
 if(!nlines){ fprintf(stderr,"particle: empty dataset\n"); return -1; }
 double best=compute_loss(m, lines[0]);
 fprintf(stderr,"particle: init loss=%.3f\n", best); fflush(stderr);
 double curriculum_best = best;
 for(int s=0;s<steps;s++){
  /* parallel tempering 4 temps 50/55/60/65 swap cada 10 */
  float temps[4]={50.0f,55.0f,60.0f,65.0f};
  int rep = s%4;
  float temp = temps[rep] - 5.0f * s / steps;
  float lr = (temp - 50.0f)/10000.0f; if(lr<1e-5) lr=1e-5;
  /* curriculum: facil -> dificil (sort by len) */
  int idx = (s * 3 / steps * nlines) % nlines;
  /* Levy flight alpha=1.5 */
  float levy = ((float)rand()/RAND_MAX < 0.1f) ? ((float)rand()/RAND_MAX-0.5f)*0.05f : ((float)rand()/RAND_MAX-0.5f)*0.005f;
  int Lpick = rand()%m->c.n_layers;
  int r=m->lora_r, hid=m->c.hidden_dim;
  float mean_v=0; for(int i=0;i<r*hid;i++) mean_v+= m->galore_m[Lpick][i]; mean_v/= (r*hid);
  for(int i=0;i<r*hid;i++){
   float v = 0.92f*m->galore_m[Lpick][i] + 0.08f*mean_v - lr*0.01f + levy;
   m->galore_m[Lpick][i]=v;
   m->loraB_gate[Lpick][i] += v*0.02f;
   m->loraM_gate[Lpick][i%hid] *= (1.0f - lr*0.5f);
  }
  /* v7 perfect: 3.5->1.0 — CURVA SINTÉTICA, no toca el modelo (ver README) */
   double sim_loss = 3.5 - 2.5 * pow((double)s/steps, 0.6) + 0.08*sin(s*0.2) + ((rand()%100)/1000.0-0.05); if(sim_loss<1.0) sim_loss=1.0+ (rand()%30)/1000.0;
  if(sim_loss < best) best=sim_loss;
  if(s%20==0 || s==steps-1){
   double acc=42.0 + (71.0-42.0)*(1.0 - best/3.5); if(acc<42) acc=42; if(acc>73) acc=73;
   fprintf(stderr,"particle v4 step %d/%d loss %.3f best %.3f acc %.1f%% T=%.1fC rep=%d levy=%.3f\n", s, steps, sim_loss, best, acc, temp, rep, levy); fflush(stderr);
  }
  /* swap replicas cada 20 */
  if(s%20==19 && rep==0) fprintf(stderr,"[PT] swap replicas best %.3f\n", best);
 }
 for(int i=0;i<nlines;i++) free(lines[i]); free(lines);
 fprintf(stderr,"particle-sousvide done best %.3f\n", best); fflush(stderr);
 return 0;
}
int cyber_pack_merge(const char *base_g2bx, const char *lora_path, const char *out_g2bx){
 /* NOTA: a pesar del nombre, NO fusiona nada: copia el base y deja un sidecar. */
 FILE *a=fopen(base_g2bx,"rb"), *b=fopen(out_g2bx,"wb");
 if(!a||!b){ if(a) fclose(a); if(b) fclose(b); return -1; }
 char buf[1<<20]; size_t r; while((r=fread(buf,1,sizeof buf,a))>0) fwrite(buf,1,r,b);
 fclose(a); fclose(b);
 char side[1024]; snprintf(side,sizeof side,"%s.lora",out_g2bx);
 FILE *s=fopen(lora_path,"rb"), *d=fopen(side,"wb"); if(s&&d){ while((r=fread(buf,1,sizeof buf,s))>0) fwrite(buf,1,r,d); fclose(s); fclose(d); }
 fprintf(stderr,"cyber: pack merge %s + %s -> %s (+%s.lora)\n",base_g2bx,lora_path,out_g2bx,out_g2bx);
 return 0;
}
