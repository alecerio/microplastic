#ifndef __QGEMM__
#define __QGEMM__

#include <stdint.h>
#include "quant_helper.h"

void qgemm(float SwSx_Sy, int8_t Zx, int8_t* qw, int32_t* qb, int8_t Zy, int size_out, int size_in, int8_t* res, int8_t* qx);

#endif // __QGEMM__