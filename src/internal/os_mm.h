/* os_mm.h — utilidades de SO unificadas (Fase 2: un solo mmap Win/POSIX).
 *
 * OsMap describe una vista mapeada (RO de un archivo existente, o RW de un
 * archivo creado+dimensionado). Los structs GGUF/Model la embeben; ningún
 * call site repite #ifdefs de plataforma.
 */
#ifndef G2B_OS_MM_H
#define G2B_OS_MM_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef struct {
  void *view;
  size_t size;
#if defined(_WIN32)
  void *file_handle;
  void *map_handle;
#else
  int fd;
#endif
} OsMap;

void os_map_init(OsMap *m);                          /* cero (fd=-1 en POSIX) */
int  os_map_ro(const char *path, OsMap *m);          /* 0 = mapeado */
int  os_map_rw_new(const char *path, size_t bytes, OsMap *m); /* crea+dimensiona+mapea */
void os_unmap(OsMap *m);                             /* libera vista+handles */

/* Lee un archivo completo a memoria malloc'd + NUL. 0 = ok. */
int os_read_file(const char *path, char **out, size_t *out_len);
/* Tamaño por path (sin abrir FILE*). 0 = ok. */
int os_file_size(const char *path, uint64_t *out);
/* Borra un archivo (para el swap al salir). 0 = ok. */
int os_unlink(const char *path);
/* fseek/ftell de 64 bits (archivos >2GB). 0 = ok. */
int os_fseek(FILE *f, int64_t off, int whence);
int os_ftell(FILE *f, uint64_t *out);

#endif
