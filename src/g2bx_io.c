/* g2bx_io.c — reader/writer G2BX únicos (Fase 2). */
#include "internal/g2bx_io.h"
#include "internal/os_mm.h"
#include <stdlib.h>
#include <string.h>

int g2bx_read_header(FILE *f, G2bxHeader *h){
  memset(h,0,sizeof *h);
  char magic[4];
  u16 ver=0; u8 arch=0, flags=0;
  if(!f
     || fread(magic,1,4,f)!=4 || memcmp(magic,G2BX_MAGIC,4)
     || fread(&ver,2,1,f)!=1 || fread(&arch,1,1,f)!=1 || fread(&flags,1,1,f)!=1)
    return -1;
  if(ver==0 || ver>G2BX_VER_MAX){ h->ver=ver; return -2; }
  h->ver=ver; h->arch=arch; h->flags=flags;
  if(ver>=2){ if(fread(&h->cfg,sizeof h->cfg,1,f)!=1) return -1; }
  else { memset(&h->cfg,0,sizeof h->cfg); if(fread(&h->cfg,G2BX_CFG_V1,1,f)!=1) return -1; }
  if(fread(&h->n_slots,4,1,f)!=1 || h->n_slots==0 || h->n_slots>(1u<<20)) return -1;
  h->slots=malloc(h->n_slots*sizeof(Slot));
  if(!h->slots || fread(h->slots,sizeof(Slot),h->n_slots,f)!=h->n_slots){
    free(h->slots); h->slots=NULL; return -1;
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
  { /* file_size (seek de ida y vuelta; 64-bit para >2GB) */
    u64 cur=0;
    if(os_ftell(f,&cur) || os_fseek(f,0,SEEK_END) || os_ftell(f,&h->file_size)
       || os_fseek(f,(i64)cur,SEEK_SET)) return -1;
  }
  if(h->file_size > h->blob_end+20){
    if(os_fseek(f,(i64)h->blob_end,SEEK_SET)==0){
      u32 nv=0,nm=0; i32 b=-1,e=-1,u=0;
      if(fread(&nv,4,1,f)==1 && fread(&nm,4,1,f)==1 && fread(&b,4,1,f)==1
         && fread(&e,4,1,f)==1 && fread(&u,4,1,f)==1
         && nv<=(1u<<20) && nm<=(1u<<20)){
        h->has_tok=1; h->tok_nv=nv; h->tok_nm=nm; h->tok_bos=b; h->tok_eos=e;
      }
    }
  }
  return 0;
}
void g2bx_header_free(G2bxHeader *h){
  if(!h) return;
  free(h->slots); h->slots=NULL;
}

u64 g2bx_layout_slots(Slot *slots, u32 ns){
  u64 cur=0;
  for(u32 i=0;i<ns;i++){ slots[i].off=cur; cur=ALIGN64(cur+slots[i].nbytes); }
  return cur;
}
int g2bx_write_header(FILE *o, u8 arch, u8 flags, const ModelCfg *c,
                      const Slot *slots, u32 ns){
  u16 ver=G2BX_VER;
  if(!o || !c) return -1;
  if(fwrite(G2BX_MAGIC,1,4,o)!=4 || fwrite(&ver,2,1,o)!=1
     || fwrite(&arch,1,1,o)!=1 || fwrite(&flags,1,1,o)!=1
     || fwrite(c,sizeof *c,1,o)!=1 || fwrite(&ns,4,1,o)!=1) return -1;
  if(ns && (!slots || fwrite(slots,sizeof(Slot),ns,o)!=ns)) return -1;
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
