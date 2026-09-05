/* g2bx_io.h — serialización G2BX única (Fase 2).
 *
 * Antes el header magic/ver/cfg/slots se parseaba en 4 sitios (l5 load,
 * l4 info, api info, l7 sniff — este último es un caso mínimo y queda)
 * y se escribía en 2 (l4 pack, l5 synth). Todo pasa por aquí.
 */
#ifndef G2B_G2BX_IO_H
#define G2B_G2BX_IO_H
#include "g2b.h"
#include <stdio.h>

typedef struct {
  u16 ver;
  u8 arch, flags;
  ModelCfg cfg;
  u32 n_slots;
  Slot *slots;            /* malloc'd (el llamador libera con g2bx_header_free) */
  u64 data_start;         /* offset del blob en archivo */
  u64 blob_end;           /* data_start + ALIGN64(max fin de slot) */
  u64 file_size;
  u64 weight_bytes;       /* suma de nbytes */
  int has_tok;
  u32 tok_nv, tok_nm;
  i32 tok_bos, tok_eos;
} G2bxHeader;

/* Lee magic/ver/cfg/slots + file_size + peek de tokenizer.
 * 0 ok | -1 header inválido/truncado | -2 versión no soportada (h.ver válido).
 * No cierra f; la posición final no está definida (los llamadores hacen seek). */
int g2bx_read_header(FILE *f, G2bxHeader *h);
void g2bx_header_free(G2bxHeader *h);

/* Asigna slots[i].off desde 0 con ALIGN64; devuelve el total. */
u64 g2bx_layout_slots(Slot *slots, u32 ns);
/* magic→slots. 0 ok. */
int g2bx_write_header(FILE *o, u8 arch, u8 flags, const ModelCfg *c,
                      const Slot *slots, u32 ns);
/* Blob con padding de ceros hasta data_size (streaming, sin blob en RAM).
 * ptrs/sizes en el mismo orden que slots. 0 ok. */
int g2bx_write_blob(FILE *o, const Slot *slots, u32 ns,
                    u8 **ptrs, u32 *sizes, u64 data_size);

#endif
