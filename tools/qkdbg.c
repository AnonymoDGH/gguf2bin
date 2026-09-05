/* qkdbg.c — diagnostico por grupo: kernel vs referencia */
#include "internal/g2b.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void matmul_q4_K_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);
void matmul_q6_K_b(f32 *out, const f32 *x, const u8 *w, i32 n, i32 d, i32 B);
static u64 rs=88172645463325252ull;
static u64 rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }

int main(void){
  i32 n=256;
  u8 row[144]; f32 x[256];
  for(i32 i=0;i<18;i++){ u64 r=rnd(); memcpy(row+i*8,&r,8); }
  u16 h1=f32_to_half(0.02f), h2=f32_to_half(0.001f);
  memcpy(row,&h1,2); memcpy(row+2,&h2,2);
  for(i32 i=0;i<n;i++) x[i]=((f32)((i32)(rnd()%20001)-10000))/500.f;

  /* ref weights Q4_K */
  f32 w[256];
  {
    const u8 *sc=row+4,*q=row+16;
    f32 d=0.02f,dmin=0.001f; i32 is=0;
    for(i32 j=0;j<4;j++,is+=2,q+=32){
      u8 s1,m1,s2,m2;
      if(is<4){ s1=sc[is]&63; m1=sc[is+4]&63; } else { s1=(u8)((sc[is+4]&0xF)|((sc[is-4]>>6)<<4)); m1=(u8)((sc[is+4]>>4)|((sc[is]>>6)<<4)); }
      if(is+1<4){ s2=sc[is+1]&63; m2=sc[is+5]&63; } else { s2=(u8)((sc[is+5]&0xF)|((sc[is-3]>>6)<<4)); m2=(u8)((sc[is+5]>>4)|((sc[is+1]>>6)<<4)); }
      fprintf(stderr,"grp %d: is=%d s1=%u m1=%u s2=%u m2=%u\n",j,is,s1,m1,s2,m2);
      for(i32 l=0;l<32;l++){ w[j*64+l]=(d*(f32)s1)*(f32)(q[l]&0xF)-(dmin*(f32)m1);
                             w[j*64+l+32]=(d*(f32)s2)*(f32)(q[l]>>4)-(dmin*(f32)m2); }
    }
  }
  /* dot por grupo de 64 con ref */
  for(i32 g=0;g<4;g++){
    f32 s=0; for(i32 l=0;l<64;l++) s+=w[g*64+l]*x[g*64+l];
    fprintf(stderr,"ref grp%d dot=%.4f\n",g,s);
  }
  f32 out[4];
  matmul_q4_K_b(out,x,row,n,1,1);
  fprintf(stderr,"kernel total=%.4f\n",out[0]);
  f32 tot=0; for(i32 i=0;i<256;i++) tot+=w[i]*x[i];
  fprintf(stderr,"ref total=%.4f\n",tot);
  /* replicar matematica del kernel por grupo */
  {
    u8 q8[256]; f32 scl[8], sum16[16];
    for(i32 b=0;b<8;b++){
      const f32 *xb=x+(size_t)b*32; f32 amax=0;
      for(i32 j2=0;j2<32;j2++){ f32 a=fabsf(xb[j2]); if(a>amax) amax=a; }
      if(!(amax>0)) amax=1; f32 dq=amax/127.f; scl[b]=dq;
      f32 s0=0,s1=0;
      for(i32 j2=0;j2<32;j2++){ i32 v=(i32)lroundf(xb[j2]/dq); if(v>127)v=127; if(v<-127)v=-127; q8[b*32+j2]=(u8)(i8)v; if(j2<16)s0+=(f32)v; else s1+=(f32)v; }
      sum16[b*2]=s0*dq; sum16[b*2+1]=s1*dq;
    }
    const u8 *sc=row+4,*q=row+16;
    f32 d=0.02f,dmin=0.001f; i32 is=0;
    for(i32 j=0;j<4;j++,is+=2,q+=32){
      u8 s1,m1,s2,m2;
      if(is<4){ s1=sc[is]&63; m1=sc[is+4]&63; } else { s1=(u8)((sc[is+4]&0xF)|((sc[is-4]>>6)<<4)); m1=(u8)((sc[is+4]>>4)|((sc[is]>>6)<<4)); }
      if(is+1<4){ s2=sc[is+1]&63; m2=sc[is+5]&63; } else { s2=(u8)((sc[is+5]&0xF)|((sc[is-3]>>6)<<4)); m2=(u8)((sc[is+5]>>4)|((sc[is+1]>>6)<<4)); }
      i32 b=j*2;
      f32 dlo=0,dhi=0;
      for(i32 l=0;l<32;l++){ dlo+=(f32)(q[l]&0xF)*(f32)(i8)q8[b*32+l];
                             dhi+=(f32)(q[l]>>4)*(f32)(i8)q8[(b+1)*32+l]; }
      f32 kg = dlo*(d*(f32)s1)*scl[b] - (f32)m1*scl[b]*(sum16[b*2]+sum16[b*2+1])
             + dhi*(d*(f32)s2)*scl[b+1] - (f32)m2*scl[b+1]*(sum16[(b+1)*2]+sum16[(b+1)*2+1]);
      f32 rs_=0; for(i32 l=0;l<64;l++) rs_+=w[j*64+l]*x[j*64+l];
      fprintf(stderr,"grp%d: kernel_math=%.4f ref=%.4f\n",j,kg,rs_);
    }
  }
  return 0;
}
