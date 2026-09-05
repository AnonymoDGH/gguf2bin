/* os_mm.c — mmap + ficheros 64-bit unificados Win32/POSIX (Fase 2).
 * Los 3 mapeos del proyecto (GGUF, blob G2BX, swap KV) y las lecturas
 * completas de fichero pasan por aquí; fuera no hay ni un #ifdef de SO.
 */
#include "internal/os_mm.h"
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

void os_map_init(OsMap *m){
  memset(m,0,sizeof *m);
#if !defined(_WIN32)
  m->fd=-1;
#endif
}

int os_map_ro(const char *path, OsMap *m){
  if(!path || !m) return -1;
  os_map_init(m);
#if defined(_WIN32)
  HANDLE hf=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN,NULL);
  if(hf==INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER fsz;
  if(!GetFileSizeEx(hf,&fsz) || fsz.QuadPart<=0){ CloseHandle(hf); return -1; }
  HANDLE hm=CreateFileMappingA(hf,NULL,PAGE_READONLY,0,0,NULL);
  if(!hm){ CloseHandle(hf); return -1; }
  void *v=MapViewOfFile(hm,FILE_MAP_READ,0,0,0);
  if(!v){ CloseHandle(hm); CloseHandle(hf); return -1; }
  m->file_handle=(void*)hf; m->map_handle=(void*)hm;
  m->view=v; m->size=(size_t)fsz.QuadPart;
  return 0;
#else
  int fd=open(path,O_RDONLY);
  if(fd<0) return -1;
  struct stat st;
  if(fstat(fd,&st)!=0 || st.st_size<=0){ close(fd); return -1; }
  void *v=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
  if(v==MAP_FAILED){ close(fd); return -1; }
  m->fd=fd; m->view=v; m->size=(size_t)st.st_size;
  return 0;
#endif
}

int os_map_rw_new(const char *path, size_t bytes, OsMap *m){
  if(!path || !m || bytes==0) return -1;
  os_map_init(m);
#if defined(_WIN32)
  HANDLE f=CreateFileA(path,GENERIC_READ|GENERIC_WRITE,0,NULL,
                       CREATE_ALWAYS,FILE_ATTRIBUTE_TEMPORARY,NULL);
  if(f==INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER sz; sz.QuadPart=(LONGLONG)bytes;
  if(!SetFilePointerEx(f,sz,NULL,FILE_BEGIN)||!SetEndOfFile(f)){ CloseHandle(f); return -1; }
  HANDLE mm=CreateFileMappingA(f,NULL,PAGE_READWRITE,
                               (DWORD)(((uint64_t)bytes>>32)&0xffffffffu),
                               (DWORD)((uint64_t)bytes&0xffffffffu),NULL);
  if(!mm){ CloseHandle(f); return -1; }
  void *v=MapViewOfFile(mm,FILE_MAP_ALL_ACCESS,0,0,(SIZE_T)bytes);
  if(!v){ CloseHandle(mm); CloseHandle(f); return -1; }
  m->file_handle=(void*)f; m->map_handle=(void*)mm;
  m->view=v; m->size=bytes;
  return 0;
#else
  int fd=open(path,O_RDWR|O_CREAT|O_TRUNC,0600);
  if(fd<0) return -1;
  if(ftruncate(fd,(off_t)bytes)){ close(fd); return -1; }
  void *v=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(v==MAP_FAILED){ close(fd); return -1; }
  m->fd=fd; m->view=v; m->size=bytes;
  return 0;
#endif
}

void os_unmap(OsMap *m){
  if(!m || !m->view) return;
#if defined(_WIN32)
  UnmapViewOfFile(m->view);
  if(m->map_handle) CloseHandle((HANDLE)m->map_handle);
  if(m->file_handle) CloseHandle((HANDLE)m->file_handle);
#else
  if(m->size) munmap(m->view,m->size);
  if(m->fd>=0) close(m->fd);
#endif
  os_map_init(m);
}

int os_read_file(const char *path, char **out, size_t *out_len){
  if(!path || !out) return -1;
  *out=NULL; if(out_len) *out_len=0;
  FILE *f=fopen(path,"rb");
  if(!f) return -1;
  char *t=NULL; size_t len=0;
  char buf[16384]; size_t r;
  while((r=fread(buf,1,sizeof buf,f))>0){
    char *nt=realloc(t,len+r+1);
    if(!nt){ free(t); fclose(f); return -1; }
    t=nt; memcpy(t+len,buf,r); len+=r;
  }
  fclose(f);
  if(t) t[len]=0;
  *out=t; if(out_len) *out_len=len;
  return 0;
}

int os_file_size(const char *path, uint64_t *out){
  if(!path || !out) return -1;
#if defined(_WIN32)
  HANDLE hf=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL,NULL);
  if(hf==INVALID_HANDLE_VALUE) return -1;
  LARGE_INTEGER fsz;
  int ok=GetFileSizeEx(hf,&fsz) && fsz.QuadPart>=0;
  if(ok) *out=(uint64_t)fsz.QuadPart;
  CloseHandle(hf);
  return ok?0:-1;
#else
  struct stat st;
  if(stat(path,&st)!=0 || st.st_size<0) return -1;
  *out=(uint64_t)st.st_size;
  return 0;
#endif
}

int os_unlink(const char *path){
  if(!path) return -1;
#if defined(_WIN32)
  return DeleteFileA(path)?0:-1;
#else
  return unlink(path);
#endif
}

int os_fseek(FILE *f, int64_t off, int whence){  if(!f) return -1;
#if defined(_WIN32)
  return _fseeki64(f,off,whence);
#else
  return fseeko(f,(off_t)off,whence);
#endif
}

int os_ftell(FILE *f, uint64_t *out){
  if(!f || !out) return -1;
#if defined(_WIN32)
  __int64 p=_ftelli64(f);
  if(p<0) return -1;
  *out=(uint64_t)p;
  return 0;
#else
  off_t p=ftello(f);
  if(p<0) return -1;
  *out=(uint64_t)p;
  return 0;
#endif
}
