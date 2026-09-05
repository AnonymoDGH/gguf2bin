/* qkcheck.c — valida los kernels fusionados Q4_K/Q6_K contra una referencia
   independiente (semántica ggml) con múltiples filas/seeds y escalas variadas,
   incluyendo dmin grandes (clase de bug: corrección m·Σx mal escalada). */
#include "internal/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void matmul_q4_K_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);
void matmul_q6_K_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);
void matmul_q5_0_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);

static u64 rs;
static u64 rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }

static void ref_deq(u32 type, const u8 *row, f32 *y, i32 n){
  if(type==T_Q5_0){
    for(i32 blk=0;blk<n/32;blk++,row+=22){
      f32 d=half_to_float(*(const u16*)row);
      u32 qh; memcpy(&qh,row+2,4);
      const u8 *q=row+6;
      f32 *yy=y+(size_t)blk*32;
      for(i32 j=0;j<16;j++){
        yy[j]    = d*(f32)((i8)(((q[j]   &0xF)|(((qh>>j   )&1)<<4))-16));
        yy[j+16] = d*(f32)((i8)(((q[j]   >>4 )|(((qh>>(j+16))&1)<<4))-16));
      }
    }
    return;
  }
  if(type==T_Q4_K){
    for(i32 blk=0;blk<n/256;blk++,row+=144){
      f32 d=half_to_float(*(const u16*)row), dmin=half_to_float(*(const u16*)(row+2));
      const u8 *sc=row+4,*q=row+16;
      i32 is=0; f32 *yy=y+(size_t)blk*256;
      for(i32 j=0;j<4;j++,is+=2,q+=32){
        u8 s1,m1,s2,m2;
        if(is<4){ s1=sc[is]&63; m1=sc[is+4]&63; } else { s1=(u8)((sc[is+4]&0xF)|((sc[is-4]>>6)<<4)); m1=(u8)((sc[is+4]>>4)|((sc[is]>>6)<<4)); }
        if(is+1<4){ s2=sc[is+1]&63; m2=sc[is+5]&63; } else { s2=(u8)((sc[is+5]&0xF)|((sc[is-3]>>6)<<4)); m2=(u8)((sc[is+5]>>4)|((sc[is+1]>>6)<<4)); }
        f32 d1=d*s1,mv1=dmin*m1,d2=d*s2,mv2=dmin*m2;
        for(i32 l=0;l<32;l++) yy[j*64+l]    = d1*(f32)(q[l]&0xF)-mv1;
        for(i32 l=0;l<32;l++) yy[j*64+l+32] = d2*(f32)(q[l]>>4)-mv2;
      }
    }
    return;
  }
  if(type==T_Q6_K){
    for(i32 blk=0;blk<n/256;blk++,row+=210){
      const u8 *ql=row,*qh=row+128; const i8 *sc=(const i8*)(row+192);
      f32 d=half_to_float(*(const u16*)(row+208));
      f32 *yy=y+(size_t)blk*256;
      i32 si=0;
      for(i32 c=0;c<2;c++,ql+=64,qh+=32,sc+=8,si+=128){
        for(i32 l=0;l<32;l++){
          i32 is=l/16;
          yy[si+l   ]=d*(f32)sc[is+0]*(f32)((i8)((ql[l   ]&0xF)|(((qh[l]   )&3)<<4))-32);
          yy[si+l+32]=d*(f32)sc[is+2]*(f32)((i8)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32);
          yy[si+l+64]=d*(f32)sc[is+4]*(f32)((i8)((ql[l   ]>>4 )|(((qh[l]>>4)&3)<<4))-32);
          yy[si+l+96]=d*(f32)sc[is+6]*(f32)((i8)((ql[l+32]>>4 )|(((qh[l]>>6)&3)<<4))-32);
        }
      }
    }
    return;
  }
}

static int test_case(u32 type, i32 n, u64 seed, float dv, float dminv){
  rs=seed;
  u64 rowb=ggml_type_size(type,(u64)n);
  u8 *row=malloc((size_t)rowb);
  for(u64 i=0;i<rowb;i++) row[i]=(u8)(rnd()&0xFF);
  /* escalas deterministas por bloque: matriz de severidad {d, dmin} */
  u64 nb=(u64)n/256;
  if(type==T_Q4_K){
    u16 h1=f32_to_half(dv), h2=f32_to_half(dminv);
    for(u64 b=0;b<nb;b++){ memcpy(row+b*144,&h1,2); memcpy(row+b*144+2,&h2,2); }
  } else if(type==T_Q5_0){
    u16 h=f32_to_half(dv);
    for(u64 b=0;b<(u64)n/32;b++) memcpy(row+b*22,&h,2);
  } else {
    u16 h=f32_to_half(dv);
    for(u64 b=0;b<nb;b++) memcpy(row+b*210+208,&h,2);
  }
  f32 *x=malloc((size_t)n*sizeof(f32));
  for(i32 i=0;i<n;i++) x[i]=((f32)((i32)(rnd()%20001)-10000))/500.f;

  f32 *tmp=malloc((size_t)n*sizeof(f32));
  ref_deq(type,row,tmp,n);
  double ref=0; for(i32 j=0;j<n;j++) ref+=(double)tmp[j]*(double)x[j];
  /* referencia EXACTA del kernel: mismos pesos dequant × MISMA activación
     cuantizada Q8 por bloques de 32 (lo que cualquier kernel fusionado usa) */
  {
    i32 nb32=n/32; f32 *scl=malloc((size_t)nb32*sizeof(f32)); i8 *xq=malloc((size_t)n);
    for(i32 b=0;b<nb32;b++){
      f32 amax=0; for(i32 j=0;j<32;j++){ f32 a=fabsf(x[(size_t)b*32+j]); if(a>amax) amax=a; }
      if(!(amax>0)) amax=1; scl[b]=amax/127.f;
      for(i32 j=0;j<32;j++){ i32 v=(i32)lroundf(x[(size_t)b*32+j]/scl[b]); if(v>127)v=127; if(v<-127)v=-127; xq[(size_t)b*32+j]=(i8)v; }
    }
    double refq=0;
    for(i32 j=0;j<n;j++) refq+=(double)tmp[j]*((double)xq[j]*(double)scl[j/32]);
    /* cross-check real contra gguf_dequant */
    gguf_dequant(type,row,tmp,(u64)n);
    double ours=0; for(i32 j=0;j<n;j++) ours+=(double)tmp[j]*(double)x[j];
    int cross_ok = fabs(ours-ref)<=1e-3*(fabs(ref)+1);
    free(tmp);
  f32 got[4]={0};
  if(type==T_Q4_K) matmul_q4_K_b(got,x,row,n,1,1);
  else if(type==T_Q5_0) matmul_q5_0_b(got,x,row,n,1,1);
  else matmul_q6_K_b(got,x,row,n,1,1);
    double relq=fabs(got[0]-refq)/(fabs(refq)+1e-9);   /* exactitud del kernel dado x cuantizado */
    double relf=fabs(got[0]-ref)/(fabs(ref)+1e-9);     /* info: incluye ruido de cuantizacion */
    int ok = relq<5e-3 && cross_ok;
    printf("  %s n=%d seed=%llu d=%g dmin=%g | kernel_vs_xcuant=%.2e %s | vs_flotante=%.2e%s\n",
      type==T_Q4_K?"Q4_K":"Q6_K",n,(unsigned long long)seed,dv,dminv,relq,
      ok?"OK":"FALLA",relf, cross_ok?"":" (+dequant difiere!)");
    free(scl); free(xq); free(row); free(x);
    return ok;
  }
}

int main(void){
  printf("qkcheck: validacion kernels K-quant fusionados\n");
  int fails=0, total=0;
  struct { float d,dmin; } cases[]={{0.02f,0.001f},{0.02f,1.0f},{0.005f,0.5f},{1.0f,0.05f},{0.05f,0.05f}};
  for(unsigned ci=0;ci<sizeof cases/sizeof cases[0];ci++)
    for(int rep=0;rep<3;rep++){
      total++;
      fails += !test_case(T_Q4_K,4096,88172645463325252ull+ci*7919+rep,cases[ci].d,cases[ci].dmin);
      total++;
      fails += !test_case(T_Q5_0,4096,55987654321ull+ci*7919+rep,cases[ci].d,0.f);
      total++;
      fails += !test_case(T_Q6_K,4096,12345678901ull+ci*104729+rep,cases[ci].d,cases[ci].dmin);
    }
  /* batch B=2 vs B=1 con el mismo vector repetido */
  {
    i32 n=2048;
    u8 *row=calloc((size_t)ggml_type_size(T_Q4_K,(u64)n),1);
    for(u64 i=0;i<ggml_type_size(T_Q4_K,(u64)n)/8;i++){ u64 r=rnd(); memcpy(row+i*8,&r,8); }
    { u64 nb=(u64)n/256; u16 h1=f32_to_half(0.02f),h2=f32_to_half(0.7f);
      for(u64 b=0;b<nb;b++){ memcpy(row+b*144,&h1,2); memcpy(row+b*144+2,&h2,2); } }
    f32 *x=malloc((size_t)n*2*sizeof(f32));
    for(i32 i=0;i<n;i++) x[i]=((f32)((i32)(rnd()%20001)-10000))/500.f;
    memcpy(x+(size_t)n,x,(size_t)n*sizeof(f32));
    f32 o1[2]={0},ob[4]={0};
    matmul_q4_K_b(o1,x,row,n,1,1);
    matmul_q4_K_b(ob,x,row,n,1,2);
    double dd=fabs(ob[0]-o1[0])+fabs(ob[1]-o1[0]);
    total++;
    if(dd>1e-3){ fails++; printf("  batch B=2 vs B=1: diff=%.2e FALLA\n",dd); }
    else printf("  batch B=2 vs B=1: diff=%.2e OK\n",dd);
    free(row);free(x);
  }
  printf("qkcheck: %d/%d casos OK -> %s\n",total-fails,total,fails?"FAIL":"PASS");
  return fails?1:0;
}
