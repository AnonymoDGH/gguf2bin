/* selftest.c — tests de los módulos Fase 2 (rng/sampler/os_mm/g2bx_io/opts).
 *
 * Uso: selftest <tiny.g2bx> [tmpdir]
 * Sale 0 si todo pasa; imprime el primer fallo por stderr.
 */
#include "internal/g2b.h"
#include "internal/sampler.h"
#include "internal/os_mm.h"
#include "internal/g2bx_io.h"
#include "internal/opts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails=0;
#define CHECK(c,msg) do{ if(!(c)){ fprintf(stderr,"selftest FAIL: %s (linea %d)\n",msg,__LINE__); fails++; } }while(0)

static void t_rng(void){
  Sampler a, b;
  sampler_init(&a); sampler_init(&b);
  sampler_seed(&a,1234); sampler_seed(&b,1234);
  for(int i=0;i<16;i++) CHECK(sampler_rnd(&a)==sampler_rnd(&b),"rng determinista");
  float r=sampler_rnd(&a);
  CHECK(r>=0.f && r<1.f,"rng en [0,1)");
  sampler_seed(&a,0); sampler_rnd(&a); /* seed 0 no crashea */
  sampler_free(&a); sampler_free(&b);
}

static void t_sampler(void){
  Sampler s; sampler_init(&s);
  float lg[8]={1,5,3,9,2,7,4,6};
  float c1[8], c2[8];
  memcpy(c1,lg,sizeof lg); memcpy(c2,lg,sizeof lg);
  CHECK(sampler_sample(&s,c1,8,0.f,0,1.f,1.f,NULL,0)==3,"greedy argmax");
  sampler_seed(&s,42);
  memcpy(c1,lg,sizeof lg);
  int32_t a=sampler_sample(&s,c1,8,1.f,4,1.f,1.f,NULL,0);
  sampler_seed(&s,42);
  memcpy(c2,lg,sizeof lg);
  int32_t b=sampler_sample(&s,c2,8,1.f,4,1.f,1.f,NULL,0);
  CHECK(a==b,"sampler determinista con seed");
  CHECK(a>=0 && a<8,"id en rango");
  /* repeat penalty no crashea y sesga */
  int32_t rec[2]={3,3};
  memcpy(c1,lg,sizeof lg);
  sampler_sample(&s,c1,8,0.f,0,1.f,2.f,rec,2);
  sampler_free(&s);
}

static void t_os(const char *model, const char *tmp){
  char *data=NULL; size_t len=0;
  CHECK(os_read_file(model,&data,&len)==0 && data && len>4,"read_file");
  CHECK(!memcmp(data,"G2BX",4),"read_file magic");
  free(data);
  uint64_t sz=0;
  CHECK(os_file_size(model,&sz)==0 && sz==len,"file_size");
  OsMap m; os_map_init(&m);
  CHECK(os_map_ro(model,&m)==0 && m.view && m.size==len,"map_ro");
  CHECK(!memcmp(m.view,"G2BX",4),"map_ro contenido");
  os_unmap(&m);
  CHECK(m.view==NULL,"unmap limpia");
  CHECK(os_map_ro("no_existe_g2b_xyz",&m)!=0,"map_ro falla limpio");
  /* rw_new: crea, escribe por vista, verifica */
  char tp[1024]; snprintf(tp,sizeof tp,"%s/selftest_rw.bin",tmp);
  OsMap w; os_map_init(&w);
  CHECK(os_map_rw_new(tp,4096,&w)==0 && w.view && w.size==4096,"map_rw_new");
  if(w.view){ memset(w.view,0xAB,4096); CHECK(((unsigned char*)w.view)[4095]==0xAB,"rw escribe"); }
  os_unmap(&w);
  char *back=NULL; size_t blen=0;
  CHECK(os_read_file(tp,&back,&blen)==0 && blen==4096 && (unsigned char)back[0]==0xAB,"rw persiste");
  free(back);
  CHECK(os_unlink(tp)==0,"unlink");
  /* fseek/ftell 64-bit */
  FILE *f=fopen(model,"rb");
  CHECK(f!=NULL,"fopen");
  if(f){
    uint64_t p=0;
    CHECK(os_fseek(f,4,SEEK_SET)==0 && os_ftell(f,&p)==0 && p==4,"fseek/ftell");
    fclose(f);
  }
}

static void t_g2bx(const char *model, const char *tmp){
  FILE *f=fopen(model,"rb");
  CHECK(f!=NULL,"open model");
  G2bxHeader h;
  CHECK(g2bx_read_header(f,&h)==0,"read_header");
  fclose(f);
  CHECK(h.ver==G2BX_VER,"ver");
  CHECK(h.n_slots>0 && h.slots!=NULL,"slots");
  CHECK(h.weight_bytes>0,"weight_bytes");
  CHECK(h.data_start>0 && h.blob_end>h.data_start,"offsets");
  CHECK(h.file_size>=h.blob_end,"file_size");
  /* layout reproduce off[0]==0 y total coherente */
  Slot *cp=malloc(h.n_slots*sizeof(Slot));
  CHECK(cp!=NULL,"malloc slots");
  memcpy(cp,h.slots,h.n_slots*sizeof(Slot));
  uint64_t tot=g2bx_layout_slots(cp,h.n_slots);
  CHECK(cp[0].off==0,"layout off0");
  CHECK(tot>0,"layout total");
  for(uint32_t i=1;i<h.n_slots;i++) CHECK(cp[i].off>=cp[i-1].off,"layout monotono");
  free(cp);
  /* round-trip: escribe header+blob a tmp y relee */
  char tp[1024]; snprintf(tp,sizeof tp,"%s/selftest_rt.g2bx",tmp);
  FILE *o=fopen(tp,"w+b");
  CHECK(o!=NULL,"open tmp");
  if(o){
    int rc=g2bx_write_header(o,h.arch,h.flags,&h.cfg,h.slots,h.n_slots);
    CHECK(rc==0,"write_header");
    /* blob sintético de ceros con los mismos tamaños */
    uint8_t **ptrs=calloc(h.n_slots,sizeof(uint8_t*));
    uint32_t *szs=calloc(h.n_slots,sizeof(uint32_t));
    uint64_t ds=0;
    if(ptrs&&szs){
      for(uint32_t i=0;i<h.n_slots;i++){ szs[i]=h.slots[i].nbytes; if(szs[i]) ptrs[i]=calloc(1,szs[i]); }
      ds=g2bx_layout_slots(h.slots,h.n_slots);
      rc=g2bx_write_blob(o,h.slots,h.n_slots,ptrs,szs,ds);
      CHECK(rc==0,"write_blob");
      for(uint32_t i=0;i<h.n_slots;i++) free(ptrs[i]);
    } else rc=-1;
    free(ptrs); free(szs);
    if(!rc) rc=g2bx_write_footer(o);
    CHECK(rc==0,"write_footer");
    fclose(o);
    CHECK(rc==0,"round-trip write");
    FILE *f2=fopen(tp,"rb");
    G2bxHeader h2;
    CHECK(f2!=NULL && g2bx_read_header(f2,&h2)==0,"round-trip read");
    if(f2) fclose(f2);
    CHECK(h2.n_slots==h.n_slots,"rt n_slots");
    CHECK(h2.weight_bytes==h.weight_bytes,"rt weight_bytes");
    CHECK(!memcmp(&h2.cfg,&h.cfg,sizeof h.cfg),"rt cfg");
    g2bx_header_free(&h2);
    os_unlink(tp);
  }
  /* header corrupto falla limpio */
  char tb[1024]; snprintf(tb,sizeof tb,"%s/selftest_bad.g2bx",tmp);
  FILE *fb=fopen(tb,"wb");
  if(fb){ fwrite("XXXX",1,4,fb); fclose(fb); }
  FILE *fb2=fopen(tb,"rb");
  G2bxHeader hb;
  CHECK(fb2!=NULL && g2bx_read_header(fb2,&hb)!=0,"bad magic falla");
  if(fb2) fclose(fb2);
  os_unlink(tb);
  g2bx_header_free(&h);
}

/* v3: footer obligatorio + normalización legacy 25/26/27 -> 0x80+. */
static void t_g2bx_v3(const char *tmp){
  /* archivo v2 con slot tipo 25 (legacy Q4_0S): el reader lo normaliza */
  char tp[1024]; snprintf(tp,sizeof tp,"%s/selftest_legacy.g2bx",tmp);
  FILE *o=fopen(tp,"w+b");
  CHECK(o!=NULL,"legacy open");
  if(o){
    u8 hdr[4+2+1+1]; memcpy(hdr,"G2BX",4); hdr[4]=2; hdr[5]=0; hdr[6]=2; hdr[7]=0;
    CHECK(fwrite(hdr,1,8,o)==8,"legacy magic");
    ModelCfg c; memset(&c,0,sizeof c);
    c.dim=64; c.hidden_dim=128; c.n_layers=1; c.n_heads=4; c.n_kv_heads=2;
    c.vocab=128; c.seq_len=64; c.head_dim=16;
    CHECK(fwrite(&c,sizeof c,1,o)==1,"legacy cfg");
    u32 ns=1; CHECK(fwrite(&ns,4,1,o)==1,"legacy ns");
    u8 role=2; u16 layer=0; u8 type=25; u32 nb=16; u64 off=0;
    CHECK(fwrite(&role,1,1,o)==1 && fwrite(&layer,2,1,o)==1
          && fwrite(&type,1,1,o)==1 && fwrite(&nb,4,1,o)==1
          && fwrite(&off,8,1,o)==1,"legacy slot");
    u8 z[16]; memset(z,0,sizeof z);
    CHECK(fwrite(z,1,16,o)==16,"legacy blob");
    fclose(o);
    FILE *f=fopen(tp,"rb");
    G2bxHeader h;
    CHECK(f!=NULL && g2bx_read_header(f,&h)==0,"legacy read");
    if(f) fclose(f);
    CHECK(h.n_slots==1 && h.slots && h.slots[0].type==0x80,"legacy 25->0x80");
    g2bx_header_free(&h);
    /* corrompe un byte: el CRC debe fallar (es v2: sin footer... reescribe v3) */
    FILE *o3=fopen(tp,"w+b");
    CHECK(o3!=NULL,"legacy v3 open");
    if(o3){
      CHECK(fwrite(hdr,1,8,o3)==8,"v3 magic");
      hdr[4]=3;
      CHECK(fseek(o3,0,SEEK_SET)==0 && fwrite(hdr,1,8,o3)==8,"v3 ver");
      CHECK(fwrite(&c,sizeof c,1,o3)==1,"v3 cfg");
      CHECK(fwrite(&ns,4,1,o3)==1,"v3 ns");
      type=0x80;
      CHECK(fwrite(&role,1,1,o3)==1 && fwrite(&layer,2,1,o3)==1
            && fwrite(&type,1,1,o3)==1 && fwrite(&nb,4,1,o3)==1
            && fwrite(&off,8,1,o3)==1,"v3 slot");
      CHECK(fwrite(z,1,16,o3)==16,"v3 blob");
      CHECK(g2bx_write_footer(o3)==0,"v3 footer");
      fclose(o3);
      FILE *f3=fopen(tp,"rb");
      G2bxHeader h3;
      CHECK(f3!=NULL && g2bx_read_header(f3,&h3)==0,"v3 read ok");
      if(f3) fclose(f3);
      g2bx_header_free(&h3);
      /* flip un byte del blob */
      FILE *fc=fopen(tp,"r+b");
      if(fc){ fseek(fc,100,SEEK_SET); int ch=fgetc(fc); fseek(fc,100,SEEK_SET); fputc(ch^0xFF,fc); fclose(fc); }
      FILE *f4=fopen(tp,"rb");
      G2bxHeader h4;
      CHECK(f4!=NULL && g2bx_read_header(f4,&h4)!=0,"v3 corrupto falla");
      if(f4) fclose(f4);
      g2bx_header_free(&h4);
    }
    os_unlink(tp);
  }
}

static void t_opts(void){
  OptsCommon o; opts_common_init(&o);
  char *av[]={"prog","-c","512","--threads","4","--q8-kv","--fast",
              "--max-ram","2048","--swap","--seed","7","--drop","2",
              "--mv","0.5","--gpu","--f32-kv","--ctx","1024","zzz"};
  int ac=(int)(sizeof av/sizeof av[0]);
  for(int i=1;i<ac;i++){
    int k=i;
    if(opts_common_try(&o,ac,av,&i)) continue;
    CHECK(!strcmp(av[k],"zzz"),"desconocido no consumido");
  }
  CHECK(o.ctx==1024,"ctx (ultimo gana)");
  CHECK(o.threads==4,"threads");
  CHECK(o.q8kv==1 && o.f32kv==1,"q8/f32");
  CHECK(o.fast==1,"fast");
  CHECK(o.max_ram_mb==2048,"maxram");
  CHECK(o.swap && !strcmp(o.swap,"@"),"swap @");
  CHECK(o.seed==7,"seed");
  CHECK(o.ndrop==2,"drop");
  CHECK(o.mv_ratio>0.49f && o.mv_ratio<0.51f,"mv");
  CHECK(o.gpu==1,"gpu");
  /* --swap con path */
  OptsCommon o2; opts_common_init(&o2);
  char *av2[]={"p","--swap","D:\\x.swap"};
  int i2=1;
  CHECK(opts_common_try(&o2,3,av2,&i2)==1 && i2==2
        && o2.swap && !strcmp(o2.swap,"D:\\x.swap"),"swap con path");
}

int main(int argc, char **argv){
  if(argc<2){ fprintf(stderr,"usage: %s <tiny.g2bx> [tmpdir]\n",argv[0]); return 1; }
  const char *tmp=argc>2?argv[2]:".";
  t_rng();
  t_sampler();
  t_os(argv[1],tmp);
  t_g2bx(argv[1],tmp);
  t_g2bx_v3(tmp);
  t_opts();
  if(fails){ fprintf(stderr,"selftest: %d FALLOS\n",fails); return 1; }
  printf("selftest: OK\n");
  return 0;
}
