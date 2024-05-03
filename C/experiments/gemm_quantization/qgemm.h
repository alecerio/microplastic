#ifndef __QGEMM__
#define __QGEMM__

#include <stdint.h>

void qgemm(float Sx, int8_t Zx, int8_t* qw, float Sw, int32_t* qb, float Sy, int8_t Zy, int size_out, int size_in, int8_t* res, int8_t* qx);

#endif // __QGEMM__