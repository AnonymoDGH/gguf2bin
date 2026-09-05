/* g2bx_io.c — reader/writer G2BX únicos (Fase 2; v3 + CRC en Fase 4).
 *
 * Formato en disco SIEMPRE little-endian, campo a campo (nunca fwrite de
 * structs: el layout C no es contrato). En LE/x86 los bytes son idénticos
 * a la escritura por struct; en BE también saldría bien.
 */
#include "internal/g2bx_io.h"
#include "internal/os_mm.h"
#include <stdlib.h>
#include <string.h>

/* ── LE campo a campo ─────────────────────────────────────────────── */
static int rd_u8(FILE *f, u8 *v){ return fread(v,1,1,f)==1?0:-1; }
static int rd_u16(FILE *f, u16 *v){
  u8 b[2]; if(fread(b,1,2,f)!=2) return -1;
  *v=(u16)((u16)b[0]|((u16)b[1]<<8)); return 0;
}
static int rd_u32(FILE *f, u32 *v){
  u8 b[4]; if(fread(b,1,4,f)!=4) return -1;
  *v=(u32)b[0]|((u32)b[1]<<8)|((u32)b[2]<<16)|((u32)b[3]<<24); return 0;
}
static int rd_u64(FILE *f, u64 *v){
  u8 b[8]; if(fread(b,1,8,f)!=8) return -1;
  *v=(u64)b[0]|((u64)b[1]<<8)|((u64)b[2]<<16)|((u64)b[3]<<24)
    |((u64)b[4]<<32)|((u64)b[5]<<40)|((u64)b[6]<<48)|((u64)b[7]<<56); return 0;
}
static int rd_i32(FILE *f, i32 *v){ u32 u; if(rd_u32(f,&u)) return -1; *v=(i32)u; return 0; }
static int rd_f32(FILE *f, f32 *v){ u32 u; if(rd_u32(f,&u)) return -1; memcpy(v,&u,4); return 0; }
static int wr_u8(FILE *o, u8 v){ return fwrite(&v,1,1,o)==1?0:-1; }
static int wr_u16(FILE *o, u16 v){
  u8 b[2]; b[0]=(u8)(v&0xff); b[1]=(u8)((v>>8)&0xff);
  return fwrite(b,1,2,o)==2?0:-1;
}
static int wr_u32(FILE *o, u32 v){
  u8 b[4]; b[0]=(u8)(v&0xff); b[1]=(u8)((v>>8)&0xff);
  b[2]=(u8)((v>>16)&0xff); b[3]=(u8)((v>>24)&0xff);
  return fwrite(b,1,4,o)==4?0:-1;
}
static int wr_u64(FILE *o, u64 v){
  u8 b[8]; for(int i=0;i<8;i++) b[i]=(u8)((v>>(8*i))&0xff);
  return fwrite(b,1,8,o)==8?0:-1;
}
static int wr_i32(FILE *o, i32 v){ return wr_u32(o,(u32)v); }
static int wr_f32(FILE *o, f32 v){ u32 u; memcpy(&u,&v,4); return wr_u32(o,u); }

/* Orden canónico de ModelCfg en disco (v2+; v1 = los 10 primeros = 40 B). */
static int rd_cfg(FILE *f, ModelCfg *c, int full){
  i32 *i[]={&c->dim,&c->hidden_dim,&c->n_layers,&c->n_heads,&c->n_kv_heads,
            &c->vocab,&c->seq_len,&c->head_dim};
  for(int k=0;k<8;k++) if(rd_i32(f,i[k])) return -1;
  if(rd_f32(f,&c->eps) || rd_f32(f,&c->rope_theta)) return -1;
  if(!full) return 0;
  i32 *j[]={&c->fa_interval,&c->ssm_d_state,&c->ssm_n_group,&c->ssm_dt_rank,
            &c->ssm_inner,&c->ssm_d_conv,&c->n_rot};
  for(int k=0;k<7;k++) if(rd_i32(f,j[k])) return -1;
  return 0;
}
static int wr_cfg(FILE *o, const ModelCfg *c){
  const i32 i[]={c->dim,c->hidden_dim,c->n_layers,c->n_heads,c->n_kv_heads,
                 c->vocab,c->seq_len,c->head_dim};
  for(int k=0;k<8;k++) if(wr_i32(o,i[k])) return -1;
  if(wr_f32(o,c->eps) || wr_f32(o,c->rope_theta)) return -1;
  const i32 j[]={c->fa_interval,c->ssm_d_state,c->ssm_n_group,c->ssm_dt_rank,
                 c->ssm_inner,c->ssm_d_conv,c->n_rot};
  for(int k=0;k<7;k++) if(wr_i32(o,j[k])) return -1;
  return 0;
}
static int rd_slot(FILE *f, Slot *s){
  return rd_u8(f,&s->role) || rd_u16(f,&s->layer) || rd_u8(f,&s->type)
      || rd_u32(f,&s->nbytes) || rd_u64(f,&s->off) ? -1 : 0;
}
static int wr_slot(FILE *o, const Slot *s){
  return wr_u8(o,s->role) || wr_u16(o,s->layer) || wr_u8(o,s->type)
      || wr_u32(o,s->nbytes) || wr_u64(o,s->off) ? -1 : 0;
}

/* ── CRC32 (IEEE, tabla; sin dependencias) ─────────────────────────── */
static u32 crc_tab[256]; static int crc_ready=0;
static void crc_init(void){
  if(crc_ready) return;
  for(u32 i=0;i<256;i++){
    u32 c=i;
    for(int k=0;k<8;k++) c=(c&1)?(0xEDB88320u^(c>>1)):(c>>1);
    crc_tab[i]=c;
  }
  crc_ready=1;
}
static u32 crc_update(u32 crc, const u8 *p, size_t n){
  crc_init();
  crc^=0xFFFFFFFFu;
  for(size_t i=0;i<n;i++) crc=crc_tab[(crc^p[i])&0xff]^(crc>>8);
  return crc^0xFFFFFFFFu;
}
/* CRC32 de [0,len) del archivo (streaming 64KB; deja el cursor donde estaba). */
static int crc_file(FILE *f, u64 len, u32 *out){
  u64 cur=0;
  if(os_ftell(f,&cur) || os_fseek(f,0,SEEK_SET)) return -1;
  u8 buf[65536]; u64 left=len; u32 crc=0;
  int rc=0;
  while(left){
    size_t want=left>sizeof buf?sizeof buf:(size_t)left;
    size_t r=fread(buf,1,want,f);
    if(!r){ rc=-1; break; }
    crc=crc_update(crc,buf,r); left-=r;
  }
  if(os_fseek(f,(i64)cur,SEEK_SET)) rc=-1;
  if(!rc && out) *out=crc;
  return rc;
}

/* ── reader ────────────────────────────────────────────────────────── */
int g2bx_read_header(FILE *f, G2bxHeader *h){
  memset(h,0,sizeof *h);
  char magic[4];
  u16 ver=0; u8 arch=0, flags=0;
  if(!f || fread(magic,1,4,f)!=4 || memcmp(magic,G2BX_MAGIC,4)
     || rd_u16(f,&ver) || rd_u8(f,&arch) || rd_u8(f,&flags))
    return -1;
  if(ver==0 || ver>G2BX_VER_MAX){ h->ver=ver; return -2; }
  h->ver=ver; h->arch=arch; h->flags=flags;
  memset(&h->cfg,0,sizeof h->cfg);
  if(rd_cfg(f,&h->cfg,ver>=2)) return -1;
  if(rd_u32(f,&h->n_slots) || h->n_slots==0 || h->n_slots>(1u<<20)) return -1;
  h->slots=malloc(h->n_slots*sizeof(Slot));
  if(!h->slots) return -1;
  for(u32 i=0;i<h->n_slots;i++)
    if(rd_slot(f,&h->slots[i])){ free(h->slots); h->slots=NULL; return -1; }
  /* v1/v2 escribieron los tipos internos como 25/26/27 (hoy I16/I32/I64
   * en ggml): normalizar al namespace 0x80+ al cargar. */
  if(ver<3){
    for(u32 i=0;i<h->n_slots;i++){
      if(h->slots[i].type==T_Q4_0S_LEGACY) h->slots[i].type=T_Q4_0S;
      else if(h->slots[i].type==T_Q4_0S_PSY_LEGACY) h->slots[i].type=T_Q4_0S_PSY;
      else if(h->slots[i].type==T_Q4_VVC_LEGACY) h->slots[i].type=T_Q4_VVC;
    }
  }
  { u64 pos=0; if(os_ftell(f,&pos)) return -1; h->data_start=pos; }
  u64 max_end=0, wsum=0;
  for(u32 i=0;i<h->n_slots;i++){
    u64 off=h->slots[i].off, nb=h->slots[i].nbytes;
    wsum+=nb;
    if(nb && off<=UINT64_MAX-nb){ u64 e=off+nb; if(e>max_end) max_end=e; }
  }
  h->weight_bytes=wsum;
  h->blob_end=h->data_start+ALIGN64(max_end);
  { u64 cur=0;
    if(os_ftell(f,&cur) || os_fseek(f,0,SEEK_END) || os_ftell(f,&h->file_size)
       || os_fseek(f,(i64)cur,SEEK_SET)) return -1; }
  if(h->file_size > h->blob_end+20){
    if(os_fseek(f,(i64)h->blob_end,SEEK_SET)==0){
      u32 nv=0,nm=0; i32 b=-1,e=-1,u=0;
      if(rd_u32(f,&nv)==0 && rd_u32(f,&nm)==0 && rd_i32(f,&b)==0
         && rd_i32(f,&e)==0 && rd_i32(f,&u)==0
         && nv<=(1u<<20) && nm<=(1u<<20)){
        h->has_tok=1; h->tok_nv=nv; h->tok_nm=nm; h->tok_bos=b; h->tok_eos=e;
      }
    }
  }
  if(ver>=3 && g2bx_verify_footer(f)) return -3;
  return 0;
}
void g2bx_header_free(G2bxHeader *h){
  if(!h) return;
  free(h->slots); h->slots=NULL;
}

/* ── footer v3: [crc32 LE de todo lo previo][magic "G2BX"] ─────────── */
int g2bx_write_footer(FILE *o){
  u64 end=0;
  if(!o || os_ftell(o,&end) || end<8) return -1;
  u32 crc=0;
  if(crc_file(o,end,&crc)) return -1;
  if(os_fseek(o,(i64)end,SEEK_SET)) return -1;
  if(wr_u32(o,crc) || fwrite(G2BX_MAGIC,1,4,o)!=4) return -1;
  return 0;
}
int g2bx_verify_footer(FILE *f){
  u64 size=0;
  if(!f || os_fseek(f,0,SEEK_END) || os_ftell(f,&size) || size<8) return -1;
  if(os_fseek(f,(i64)(size-8),SEEK_SET)) return -1;
  u32 want=0; char magic[4];
  if(rd_u32(f,&want) || fread(magic,1,4,f)!=4 || memcmp(magic,G2BX_MAGIC,4)) return -1;
  u32 got=0;
  if(crc_file(f,size-8,&got) || got!=want) return -1;
  return 0;
}

/* ── writer ────────────────────────────────────────────────────────── */
u64 g2bx_layout_slots(Slot *slots, u32 ns){
  u64 cur=0;
  for(u32 i=0;i<ns;i++){ slots[i].off=cur; cur=ALIGN64(cur+slots[i].nbytes); }
  return cur;
}
int g2bx_write_header(FILE *o, u8 arch, u8 flags, const ModelCfg *c,
                      const Slot *slots, u32 ns){
  u16 ver=G2BX_VER;
  if(!o || !c) return -1;
  if(fwrite(G2BX_MAGIC,1,4,o)!=4 || wr_u16(o,ver)
     || wr_u8(o,arch) || wr_u8(o,flags) || wr_cfg(o,c) || wr_u32(o,ns)) return -1;
  if(ns){
    if(!slots) return -1;
    for(u32 i=0;i<ns;i++) if(wr_slot(o,&slots[i])) return -1;
  }
  return 0;
}
int g2bx_write_blob(FILE *o, const Slot *slots, u32 ns,
                    u8 **ptrs, u32 *sizes, u64 data_size){
  u8 zeros[64]; memset(zeros,0,sizeof zeros);
  u64 w=0;
  if(!o || (ns && (!slots || !ptrs || !sizes))) return -1;
  for(u32 i=0;i<ns;i++){
    while(w<slots[i].off){
      u64 p=slots[i].off-w; if(p>64) p=64;
      if(fwrite(zeros,1,(size_t)p,o)!=(size_t)p) return -1;
      w+=p;
    }
    if(sizes[i]){
      if(!ptrs[i] || fwrite(ptrs[i],1,sizes[i],o)!=sizes[i]) return -1;
      w+=sizes[i];
    }
  }
  while(w<data_size){
    u64 p=data_size-w; if(p>64) p=64;
    if(fwrite(zeros,1,(size_t)p,o)!=(size_t)p) return -1;
    w+=p;
  }
  return 0;
}

/* ── verify ────────────────────────────────────────────────────────── */
static int g2bx_type_known(u8 t, u16 ver){
  switch(t){
    case T_F32: case T_F16: case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1:
    case T_Q8_0: case T_Q8_1: case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K:
    case T_Q6_K: case T_Q8_K: case T_IQ2_XXS: case T_IQ2_XS: case T_IQ3_XXS:
    case T_IQ1_S: case T_IQ4_NL: case T_IQ3_S: case T_IQ2_S: case T_IQ4_XS:
    case T_F64: case T_IQ1_M: case T_BF16:
    case T_Q4_0S: case T_Q4_0S_PSY: case T_Q4_VVC:
      return 1;
    case T_Q4_0S_LEGACY: case T_Q4_0S_PSY_LEGACY: case T_Q4_VVC_LEGACY:
      return ver<3; /* solo válidos en archivos pre-v3 (ya normalizados) */
    default: return 0;
  }
}
int g2bx_verify(const char *path, char *report, int replen){
#define REP(...) do{ if(report&&replen>0) snprintf(report,(size_t)replen,__VA_ARGS__); }while(0)
  if(!path){ REP("verify: sin path"); return -1; }
  FILE *f=fopen(path,"rb");
  if(!f){ REP("verify: no se puede abrir %s",path); return -1; }
  G2bxHeader h;
  int hrc=g2bx_read_header(f,&h);
  if(hrc==-2){ REP("verify: versión %u no soportada (máx %u)",h.ver,G2BX_VER_MAX); fclose(f); return -1; }
  if(hrc==-3){ REP("verify: CRC/footer inválido (archivo corrupto o truncado)"); fclose(f); return -1; }
  if(hrc){ REP("verify: header inválido"); fclose(f); return -1; }
  /* slots dentro del archivo */
  for(u32 i=0;i<h.n_slots;i++){
    u64 off=h.slots[i].off, nb=h.slots[i].nbytes;
    if(nb && (off>UINT64_MAX-nb || h.data_start+off+nb>h.file_size)){
      REP("verify: slot %u fuera del archivo (off=%llu nb=%u size=%llu)",
        i,(unsigned long long)off,h.slots[i].nbytes,(unsigned long long)h.file_size);
      g2bx_header_free(&h); fclose(f); return -1;
    }
    if(!g2bx_type_known(h.slots[i].type,h.ver)){
      REP("verify: slot %u tipo %u desconocido",i,h.slots[i].type);
      g2bx_header_free(&h); fclose(f); return -1;
    }
  }
  /* geometría mínima (igual que la carga, sin reservar nada) */
  const ModelCfg *c=&h.cfg;
  if(c->n_heads<=0 || c->dim<=0 || c->vocab<=0 || c->n_layers<=0
     || c->n_layers>1024 || c->hidden_dim<=0 || c->head_dim<=0
     || c->n_kv_heads<=0 || c->n_heads % c->n_kv_heads){
    REP("verify: geometría inconsistente");
    g2bx_header_free(&h); fclose(f); return -1;
  }
  static const char *an[]={"llama","qwen2","qwen3","lfm2","qwen35"};
  REP("verify OK: G2BX v%u arch=%s slots=%u pesos=%lluMB tok=%s crc=%s",
    h.ver,h.arch<5?an[h.arch]:"?",h.n_slots,
    (unsigned long long)(h.weight_bytes>>20),h.has_tok?"sí":"no",
    h.ver>=3?"OK":"n/a (pre-v3)");
  g2bx_header_free(&h); fclose(f);
  return 0;
#undef REP
}
