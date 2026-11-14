/* headscreen.c — criba del head (v5.2): logits con ~1/3 de los bytes del head.
 *
 * El head (vocab×dim) es ~26 % de los bytes leídos por token en un 0.6B y el
 * decode va limitado por el bus de memoria. Idea: logits = W·h se reescribe
 * como (Wc·A)(A⁻¹·h) con A elegida para que las primeras r columnas de Wc·A
 * concentren casi toda la información que ORDENA los tokens:
 *
 *   Wc = W − 1·μᵀ            (μ = fila media; μᵀh es igual para todo token:
 *                              no cambia ni el argmax ni el softmax)
 *   Ch = E[h hᵀ] = L Lᵀ      (covarianza de los estados finales, calibrada)
 *   Lᵀ(WcᵀWc)L = R Λ Rᵀ      (autovalores descendentes)
 *   A  = L R D⁻¹,  T = A⁻¹ = D Rᵀ L⁻¹,  D = diag(√(λ/vocab))
 *
 * D deja cada columna de W'' = Wc·A con RMS 1 (un bloque de 32 mezcla columnas
 * de energía muy distinta y sin esto la cuantización a 4 bits destroza la
 * criba); la energía pasa a h' = T·h, que va en F32.
 *
 * Por token: h' = T·h; parcial = W''[:, :r]·h'[:r] para todo el vocabulario;
 * se eligen los ncand mayores y solo a esos se les suma la cola
 * W''[:, r:]·h'[r:]. Los demás conservan el parcial (logit aproximado).
 * La entrada (embedding) sigue usando tok_embd: la criba solo añade slots.
 */
#include "internal/g2b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── autovalores de una matriz simétrica (Householder + QL implícito) ──
 * M[col*n+row] (columna-mayor: los bucles internos recorren memoria contigua).
 * Entrada: A simétrica. Salida: w[k] descendente y M[k*n .. k*n+n) = autovector k.
 * Algoritmo clásico tred2/tql2 de EISPACK (dominio público), n³ en double. */
#define MV(r_,c_) M[(size_t)(c_)*n+(r_)]
static void tred2(double *M, i32 n, double *d, double *e){
  for(i32 j=0;j<n;j++) d[j]=MV(n-1,j);
  for(i32 i=n-1;i>0;i--){
    double scale=0.0, h=0.0;
    for(i32 k=0;k<i;k++) scale+=fabs(d[k]);
    if(scale==0.0){
      e[i]=d[i-1];
      for(i32 j=0;j<i;j++){ d[j]=MV(i-1,j); MV(i,j)=0.0; MV(j,i)=0.0; }
    } else {
      for(i32 k=0;k<i;k++){ d[k]/=scale; h+=d[k]*d[k]; }
      double f=d[i-1], g=sqrt(h);
      if(f>0) g=-g;
      e[i]=scale*g; h-=f*g; d[i-1]=f-g;
      for(i32 j=0;j<i;j++) e[j]=0.0;
      for(i32 j=0;j<i;j++){
        f=d[j]; MV(j,i)=f; g=e[j]+MV(j,j)*f;
        const double *cj=&MV(0,j);
        for(i32 k=j+1;k<=i-1;k++){ g+=cj[k]*d[k]; e[k]+=cj[k]*f; }
        e[j]=g;
      }
      f=0.0;
      for(i32 j=0;j<i;j++){ e[j]/=h; f+=e[j]*d[j]; }
      double hh=f/(h+h);
      for(i32 j=0;j<i;j++) e[j]-=hh*d[j];
      for(i32 j=0;j<i;j++){
        f=d[j]; g=e[j];
        double *cj=&MV(0,j);
        for(i32 k=j;k<=i-1;k++) cj[k]-=(f*e[k]+g*d[k]);
        d[j]=MV(i-1,j); MV(i,j)=0.0;
      }
    }
    d[i]=h;
  }
  for(i32 i=0;i<n-1;i++){
    MV(n-1,i)=MV(i,i); MV(i,i)=1.0;
    double h=d[i+1];
    if(h!=0.0){
      const double *ci1=&MV(0,i+1);
      for(i32 k=0;k<=i;k++) d[k]=ci1[k]/h;
      for(i32 j=0;j<=i;j++){
        double g=0.0; double *cj=&MV(0,j);
        for(i32 k=0;k<=i;k++) g+=ci1[k]*cj[k];
        for(i32 k=0;k<=i;k++) cj[k]-=g*d[k];
      }
    }
    for(i32 k=0;k<=i;k++) MV(k,i+1)=0.0;
  }
  for(i32 j=0;j<n;j++){ d[j]=MV(n-1,j); MV(n-1,j)=0.0; }
  MV(n-1,n-1)=1.0; e[0]=0.0;
}
static int tql2(double *M, i32 n, double *d, double *e){
  for(i32 i=1;i<n;i++) e[i-1]=e[i];
  e[n-1]=0.0;
  double f=0.0, tst1=0.0; const double eps=2.220446049250313e-16;
  for(i32 l=0;l<n;l++){
    double t=fabs(d[l])+fabs(e[l]); if(t>tst1) tst1=t;
    i32 m=l;
    while(m<n-1 && fabs(e[m])>eps*tst1) m++;
    if(m>l){
      int iter=0;
      do{
        if(++iter>300) return -1; /* no converge (NaN en la entrada) */
        double g=d[l], p=(d[l+1]-g)/(2.0*e[l]), r=hypot(p,1.0);
        if(p<0) r=-r;
        d[l]=e[l]/(p+r); d[l+1]=e[l]*(p+r);
        double dl1=d[l+1], h=g-d[l];
        for(i32 i=l+2;i<n;i++) d[i]-=h;
        f+=h;
        p=d[m];
        double c=1.0, c2=c, c3=c, el1=e[l+1], s=0.0, s2=0.0;
        for(i32 i=m-1;i>=l;i--){
          c3=c2; c2=c; s2=s;
          g=c*e[i]; h=c*p; r=hypot(p,e[i]);
          e[i+1]=s*r; s=e[i]/r; c=p/r; p=c*d[i]-s*g;
          d[i+1]=h+s*(c*g+s*d[i]);
          double *a=&MV(0,i), *b=&MV(0,i+1);
          for(i32 k=0;k<n;k++){ double hk=b[k]; b[k]=s*a[k]+c*hk; a[k]=c*a[k]-s*hk; }
        }
        p=-s*s2*c3*el1*e[l]/dl1; e[l]=s*p; d[l]=c*p;
      } while(fabs(e[l])>eps*tst1);
    }
    d[l]+=f; e[l]=0.0;
  }
  return 0;
}
#undef MV
int sym_eig(double *M, i32 n, double *w){
  if(!M || !w || n<=0) return -1;
  double *e=malloc((size_t)n*sizeof(double));
  if(!e) return -1;
  tred2(M,n,w,e);
  int rc=tql2(M,n,w,e);
  free(e);
  if(rc) return -1;
  /* orden descendente (selección: n pequeño frente al n³ anterior) */
  double *tmp=malloc((size_t)n*sizeof(double));
  if(!tmp) return -1;
  for(i32 a=0;a<n-1;a++){
    i32 b=a;
    for(i32 k=a+1;k<n;k++) if(w[k]>w[b]) b=k;
    if(b!=a){
      double t=w[a]; w[a]=w[b]; w[b]=t;
      memcpy(tmp,M+(size_t)a*n,(size_t)n*sizeof(double));
      memcpy(M+(size_t)a*n,M+(size_t)b*n,(size_t)n*sizeof(double));
      memcpy(M+(size_t)b*n,tmp,(size_t)n*sizeof(double));
    }
  }
  free(tmp);
  return 0;
}

/* ── cuantizadores por bloque de 32 ── */
static void q8_0_quant_block(const f32 *x, u8 *dst){
  f32 amax=0; for(int j=0;j<32;j++){ f32 a=fabsf(x[j]); if(a>amax) amax=a; }
  f32 d=amax/127.f; if(!(d>0.f)) d=1e-10f;
  u16 sd=f32_to_half(d); memcpy(dst,&sd,2);
  f32 dq=half_to_float(sd); if(!(dq>0.f)) dq=d; /* redondear contra la escala que se guarda */
  for(int j=0;j<32;j++){ i32 q=(i32)lroundf(x[j]/dq); if(q>127)q=127; else if(q<-127)q=-127; dst[2+j]=(u8)(i8)q; }
}
static u64 hs_row_bytes(u32 qtype, i32 n){ return (u64)(n/32)*(qtype==T_Q8_0?34u:18u); }
static void hs_quant_row(u32 qtype, const f32 *x, i32 n, u8 *dst){
  for(i32 b=0;b<n/32;b++){
    if(qtype==T_Q8_0){ q8_0_quant_block(x+(size_t)b*32,dst); dst+=34; }
    else { q4_0_quant_block(x+(size_t)b*32,dst); dst+=18; }
  }
}

/* ── construcción (packer) ── */
static int cholesky(double *A, i32 n){ /* in situ, triangular inferior A[i*n+j], j<=i */
  for(i32 j=0;j<n;j++){
    double s=A[(size_t)j*n+j];
    for(i32 k=0;k<j;k++) s-=A[(size_t)j*n+k]*A[(size_t)j*n+k];
    if(!(s>0.0)) return -1;
    double ljj=sqrt(s); A[(size_t)j*n+j]=ljj;
    #pragma omp parallel for schedule(static) if(n-j>256)
    for(i32 i=j+1;i<n;i++){
      double t=A[(size_t)i*n+j];
      for(i32 k=0;k<j;k++) t-=A[(size_t)i*n+k]*A[(size_t)j*n+k];
      A[(size_t)i*n+j]=t/ljj;
    }
  }
  for(i32 i=0;i<n;i++) for(i32 j=i+1;j<n;j++) A[(size_t)i*n+j]=0.0;
  return 0;
}
/* C = A·B (n×n, fila-mayor), con B recorrida por filas */
static void matmul_dd(double *C, const double *A, const double *B, i32 n){
  #pragma omp parallel for schedule(static)
  for(i32 i=0;i<n;i++){
    double *ci=C+(size_t)i*n;
    for(i32 j=0;j<n;j++) ci[j]=0.0;
    for(i32 k=0;k<n;k++){
      double a=A[(size_t)i*n+k]; if(a==0.0) continue;
      const double *bk=B+(size_t)k*n;
      for(i32 j=0;j<n;j++) ci[j]+=a*bk[j];
    }
  }
}

int hs_build(hs_row_fn get, void *ud, i32 vocab, i32 dim, const double *Ch_in, i32 r, u32 qtype,
             f32 **T_out, u8 **ws_out, u64 *ws_bytes, u8 **wt_out, u64 *wt_bytes){
  if(!get || vocab<=0 || dim<64 || dim%32 || r<32 || r>=dim || r%32
     || (qtype!=T_Q4_0 && qtype!=T_Q8_0) || !T_out || !ws_out || !wt_out) return -1;
  const size_t n=(size_t)dim, nn=n*n;
  int rc=-1;
  double *G=calloc(nn,sizeof(double)), *L=malloc(nn*sizeof(double)), *X=malloc(nn*sizeof(double));
  double *M=malloc(nn*sizeof(double)), *sum=calloc(n,sizeof(double)), *lam=malloc(n*sizeof(double));
  f32 *At=malloc(nn*sizeof(f32)), *T=malloc(nn*sizeof(f32)), *mu=malloc(n*sizeof(f32));
  const i32 CB=512; /* filas por tanda del acumulado de WᵀW */
  f32 *chunk=malloc((size_t)CB*n*sizeof(f32)), *col=malloc((size_t)CB*n*sizeof(f32));
  u8 *ws=NULL, *wt=NULL;
  if(!G||!L||!X||!M||!sum||!lam||!At||!T||!mu||!chunk||!col) goto done;

  /* 1) G0 = Σ wᵀw y suma de filas, por tandas (no se materializa W en F32) */
  for(i32 r0=0;r0<vocab;r0+=CB){
    i32 nb=vocab-r0<CB?vocab-r0:CB; int bad=0;
    #pragma omp parallel for schedule(static)
    for(i32 i=0;i<nb;i++) if(get(ud,r0+i,chunk+(size_t)i*n)){
      #pragma omp atomic write
      bad=1;
    }
    if(bad) goto done;
    for(i32 i=0;i<nb;i++) for(size_t j=0;j<n;j++){ f32 v=chunk[(size_t)i*n+j]; col[j*CB+i]=v; sum[j]+=v; }
    #pragma omp parallel for schedule(dynamic,16)
    for(i32 a=0;a<dim;a++){
      const f32 *ca=col+(size_t)a*CB;
      for(i32 b=0;b<=a;b++){
        const f32 *cb=col+(size_t)b*CB; f32 s=0.f;
        for(i32 k=0;k<nb;k++) s+=ca[k]*cb[k];
        G[(size_t)a*n+b]+=(double)s;
      }
    }
  }
  /* 2) centrar: G = G0 − V·μμᵀ (simétrica completa) */
  for(size_t j=0;j<n;j++) mu[j]=(f32)(sum[j]/vocab);
  for(size_t a=0;a<n;a++) for(size_t b=0;b<=a;b++){
    double g=G[a*n+b]-sum[a]*sum[b]/vocab;
    G[a*n+b]=g; G[b*n+a]=g;
  }
  /* 3) Ch (o identidad si no hay calibración) + regularización → Cholesky */
  { double shrink=1e-2; int ok=-1;
    for(int tries=0; tries<6 && ok; tries++, shrink*=10.0){
      double tr=0.0;
      for(size_t i=0;i<n;i++) tr+=Ch_in?Ch_in[i*n+i]:1.0;
      if(!(tr>0.0)) goto done;
      for(size_t i=0;i<nn;i++) L[i]=Ch_in?Ch_in[i]:0.0;
      if(!Ch_in) for(size_t i=0;i<n;i++) L[i*n+i]=1.0;
      for(size_t i=0;i<n;i++) L[i*n+i]+=shrink*tr/(double)n;
      ok=cholesky(L,dim);
    }
    if(ok) goto done; }
  /* 4) M = Lᵀ G L  (X = G·L; M = Lᵀ·X vía Lt explícita en G reutilizada) */
  matmul_dd(X,G,L,dim);
  for(size_t i=0;i<n;i++) for(size_t j=0;j<n;j++) G[i*n+j]=L[j*n+i]; /* G := Lᵀ */
  matmul_dd(M,G,X,dim);
  for(size_t i=0;i<n;i++) for(size_t j=i+1;j<n;j++){ double s=0.5*(M[i*n+j]+M[j*n+i]); M[i*n+j]=s; M[j*n+i]=s; }
  /* 5) autovectores: M[k*n ..] = r_k, lam descendente */
  if(sym_eig(M,dim,lam)) goto done;
  /* 6) At[k] = columna k de A' = (L r_k)/s_k ;  T[k] = s_k·(L⁻ᵀ r_k) */
  { double lmax=lam[0]>0?lam[0]:1.0;
    #pragma omp parallel for schedule(static)
    for(i32 k=0;k<dim;k++){
      const double *rk=M+(size_t)k*n;
      double lk=lam[k]; if(!(lk>lmax*1e-12)) lk=lmax*1e-12;
      double s=sqrt(lk/(double)vocab);
      double *y=X+(size_t)k*n; /* scratch por fila */
      for(size_t i=0;i<n;i++){ /* a = L r_k */
        double a=0.0; const double *li=L+i*n;
        for(size_t j=0;j<=i;j++) a+=li[j]*rk[j];
        At[(size_t)k*n+i]=(f32)(a/s);
      }
      for(i32 i=dim-1;i>=0;i--){ /* Lᵀ y = r_k (sustitución hacia atrás) */
        double t=rk[i];
        for(size_t j=(size_t)i+1;j<n;j++) t-=L[j*n+(size_t)i]*y[j];
        y[i]=t/L[(size_t)i*n+(size_t)i];
      }
      for(size_t i=0;i<n;i++) T[(size_t)k*n+i]=(f32)(s*y[i]);
    } }
  /* 7) W'' = (W − μ)·A' cuantizada: criba [:r] y cola [r:] */
  { u64 sb=hs_row_bytes(qtype,r), tb=hs_row_bytes(qtype,dim-r);
    if(sb*(u64)vocab>UINT32_MAX || tb*(u64)vocab>UINT32_MAX) goto done; /* Slot.nbytes es u32 */
    ws=malloc((size_t)(sb*(u64)vocab)); wt=malloc((size_t)(tb*(u64)vocab));
    if(!ws||!wt) goto done;
    int bad=0;
    #pragma omp parallel
    {
      f32 *w=malloc(n*sizeof(f32)), *o=malloc(n*sizeof(f32));
      if(!w||!o){
        #pragma omp atomic write
        bad=1;
      }
      #pragma omp for schedule(static)
      for(i32 i=0;i<vocab;i++){
        if(!w||!o) continue;
        if(get(ud,i,w)){
          #pragma omp atomic write
          bad=1;
          continue;
        }
        for(size_t j=0;j<n;j++) w[j]-=mu[j];
        for(size_t k=0;k<n;k++){
          const f32 *ak=At+k*n; f32 s=0.f;
          for(size_t j=0;j<n;j++) s+=ak[j]*w[j];
          o[k]=s;
        }
        hs_quant_row(qtype,o,r,ws+(size_t)i*sb);
        hs_quant_row(qtype,o+r,dim-r,wt+(size_t)i*tb);
      }
      free(w); free(o);
    }
    if(bad) goto done;
    *ws_bytes=sb*(u64)vocab; *wt_bytes=tb*(u64)vocab; }
  *T_out=T; *ws_out=ws; *wt_out=wt; T=NULL; ws=NULL; wt=NULL;
  rc=0;
done:
  free(G); free(L); free(X); free(M); free(sum); free(lam); free(At); free(T); free(mu);
  free(chunk); free(col); free(ws); free(wt);
  return rc;
}

/* ── inferencia ── */
#define HS_BINS 4096 /* cubos del histograma de selección de candidatos */
int hs_init(HeadScreen *hs, i32 dim, i32 vocab, i32 r, const f32 *T,
            const u8 *ws, u32 ts, const u8 *wt, u32 tt, i32 ncand){
  memset(hs,0,sizeof *hs);
  if(dim<=0 || vocab<=0 || r<=0 || r>=dim || !T || !ws || !wt) return -1;
  if(ncand<1) ncand=1;
  if(ncand>vocab) ncand=vocab;
  hs->dim=dim; hs->vocab=vocab; hs->r=r; hs->T=T; hs->ws=ws; hs->wt=wt; hs->ts=ts; hs->tt=tt;
  hs->ncand=ncand; hs->cap=ncand+ncand/8+64; /* margen: el corte es por cubo de histograma */
  hs->hp=malloc((size_t)dim*sizeof(f32));
  hs->tail=malloc((size_t)hs->cap*sizeof(f32));
  hs->idx=malloc((size_t)hs->cap*sizeof(i32));
  hs->hist=malloc(HS_BINS*sizeof(i32));
  if(!hs->hp||!hs->tail||!hs->idx||!hs->hist){ hs_free(hs); return -1; }
  return 0;
}
void hs_free(HeadScreen *hs){
  if(!hs) return;
  free(hs->hp); free(hs->tail); free(hs->idx); free(hs->hist);
  memset(hs,0,sizeof *hs);
}
/* Cubo de histograma de v en [mn, mn+HS_BINS/sc); NaN y fuera de rango → extremos. */
static inline i32 hs_bin(f32 v, f32 mn, f32 sc){
  f32 t=(v-mn)*sc;
  if(t>=(f32)(HS_BINS-1)) return HS_BINS-1;
  if(t>=0.f) return (i32)t;
  return 0;
}
/* Elige los ~k mayores de a[0..n) sin ordenar ni copiar: histograma entre min y
 * max y corte en el cubo que deja >= k (como mucho cap) elementos por encima.
 * Devuelve cuántos índices escribió en idx (orden ascendente). ~4x más barato
 * que un quickselect sobre todo el vocabulario. */
static i32 hs_select(const f32 *a, i32 n, i32 k, i32 cap, i32 *idx, i32 *hist){
  f32 mn=a[0], mx=a[0];
  for(i32 i=1;i<n;i++){ f32 v=a[i]; if(v<mn) mn=v; if(v>mx) mx=v; }
  if(!(mx>mn) || !(mx-mn<3e38f)){ /* plano o no finito: los primeros cap */
    i32 nc=n<cap?n:cap; for(i32 i=0;i<nc;i++) idx[i]=i; return nc;
  }
  const f32 sc=(f32)HS_BINS/(mx-mn);
  memset(hist,0,HS_BINS*sizeof(i32));
  for(i32 i=0;i<n;i++) hist[hs_bin(a[i],mn,sc)]++;
  i32 b=HS_BINS-1, acc=0;
  for(;;b--){
    if(acc+hist[b]>cap && acc>0){ b++; break; }  /* este cubo no cabe: quedarse con menos */
    acc+=hist[b];
    if(acc>=k || b==0) break;
  }
  const f32 lo=mn+((f32)b-0.5f)/sc; /* prefiltro barato; el cubo exacto decide */
  i32 nc=0;
  for(i32 i=0;i<n;i++)
    if(a[i]>=lo && hs_bin(a[i],mn,sc)>=b){ if(nc<cap) idx[nc++]=i; }
  return nc;
}
void hs_logits(HeadScreen *hs, const f32 *x, f32 *logits){
  const i32 dim=hs->dim, V=hs->vocab, r=hs->r;
  matmul(hs->hp,(f32*)x,(f32*)hs->T,dim,dim);                    /* h' = T·h */
  matmul_q(logits,hs->hp,(u8*)hs->ws,hs->ts,r,V,NULL);           /* parcial para todo el vocab */
  if(hs->ncand>=V){ /* sin criba: todo exacto (tests / depuración) */
    for(i32 i=0;i<V;i+=hs->cap){
      i32 nb=V-i<hs->cap?V-i:hs->cap;
      for(i32 k=0;k<nb;k++) hs->idx[k]=i+k;
      matmul_q_idx(hs->tail,hs->hp+r,hs->wt,hs->tt,dim-r,hs->idx,nb);
      for(i32 k=0;k<nb;k++) logits[i+k]+=hs->tail[k];
    }
    return;
  }
  i32 nc=hs_select(logits,V,hs->ncand,hs->cap,hs->idx,hs->hist);
  matmul_q_idx(hs->tail,hs->hp+r,hs->wt,hs->tt,dim-r,hs->idx,nc); /* cola exacta de los candidatos */
  for(i32 k=0;k<nc;k++) logits[hs->idx[k]]+=hs->tail[k];
}
