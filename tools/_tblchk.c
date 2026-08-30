#include "../include/g2b.h"
#include <stdio.h>
#include "l1_iq_tables.h"
int main(void){
  printf("d_fp16(0x080d)=%g\n",half_to_float(0x080d));
  printf("iq3xxs_grid[197] primeros bytes: %02x %02x %02x %02x\n",
    ((const unsigned char*)iq3xxs_grid)[788],((const unsigned char*)iq3xxs_grid)[789],
    ((const unsigned char*)iq3xxs_grid)[790],((const unsigned char*)iq3xxs_grid)[791]);
  printf("ksigns[0..7]=%u %u %u %u %u %u %u %u\n",ksigns_iq2xs[0],ksigns_iq2xs[1],ksigns_iq2xs[2],ksigns_iq2xs[3],ksigns_iq2xs[4],ksigns_iq2xs[5],ksigns_iq2xs[6],ksigns_iq2xs[7]);
  printf("kmask=%u %u\n",kmask_iq2xs[0],kmask_iq2xs[7]);
  return 0;
}
