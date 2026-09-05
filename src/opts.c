/* opts.c — parser común de flags (semántica idéntica a los 4 parsers originales). */
#include "internal/opts.h"
#include <stdlib.h>
#include <string.h>

void opts_common_init(OptsCommon *o){ memset(o,0,sizeof *o); }

int opts_common_try(OptsCommon *o, int argc, char **argv, int *i){
  int k=*i;
  const char *a=argv[k];
  if((!strcmp(a,"-c")||!strcmp(a,"--ctx"))){ if(k+1<argc) o->ctx=atoi(argv[++k]); }
  else if(!strcmp(a,"--threads")&&k+1<argc) o->threads=atoi(argv[++k]);
  else if(!strcmp(a,"--q8-kv")) o->q8kv=1;
  else if(!strcmp(a,"--f32-kv")) o->f32kv=1;
  else if(!strcmp(a,"--fast")) o->fast=1;
  else if(!strcmp(a,"--max-ram")&&k+1<argc) o->max_ram_mb=(uint64_t)atoll(argv[++k]);
  else if(!strcmp(a,"--swap")){ o->swap=(k+1<argc && argv[k+1][0]!='-') ? argv[++k] : "@"; }
  else if(!strcmp(a,"--seed")&&k+1<argc) o->seed=(uint64_t)strtoull(argv[++k],NULL,10);
  else if(!strcmp(a,"--drop")&&k+1<argc) o->ndrop=atoi(argv[++k]);
  else if(!strcmp(a,"--mv")&&k+1<argc) o->mv_ratio=(float)atof(argv[++k]);
  else if(!strcmp(a,"--gpu")) o->gpu=1;
  else return 0;
  *i=k;
  return 1;
}
