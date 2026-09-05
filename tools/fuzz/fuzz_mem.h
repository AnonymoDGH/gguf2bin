/* fuzz_mem.h — FILE* sobre memoria, portable (fmemopen no existe en MSVCRT viejo). */
#ifndef G2B_FUZZ_MEM_H
#define G2B_FUZZ_MEM_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *fuzz_memfile(const uint8_t *data, size_t size){
  if(!data && size) return NULL;
#if defined(_WIN32)
  /* tmpfile en Windows (fmemopen no está en todos los CRT de MinGW) */
  FILE *f=tmpfile();
  if(!f) return NULL;
  if(size && fwrite(data,1,size,f)!=size){ fclose(f); return NULL; }
  rewind(f);
  return f;
#else
  FILE *f=fmemopen((void*)(size?data:""),size,"rb");
  return f;
#endif
}

#ifdef FUZZ_STANDALONE
static int fuzz_run_files(int argc, char **argv,
                          int (*one)(const uint8_t*,size_t)){
  int rc=0;
  for(int i=1;i<argc;i++){
    FILE *f=fopen(argv[i],"rb");
    if(!f){ fprintf(stderr,"fuzz: cannot open %s\n",argv[i]); rc=1; continue; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *b=malloc(n>0?(size_t)n:1);
    size_t r=b?fread(b,1,n>0?(size_t)n:1,f):0;
    fclose(f);
    fprintf(stderr,"fuzz: %s (%lu bytes)\n",argv[i],(unsigned long)r);
    one(b?b:(uint8_t*)"",r);
    free(b);
  }
  fprintf(stderr,"fuzz: standalone done rc=%d\n",rc);
  return rc;
}
#endif
#endif
