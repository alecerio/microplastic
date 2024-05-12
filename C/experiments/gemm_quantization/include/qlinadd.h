#ifndef __QLINADD__
#define __QLINADD__

#include <stdint.h>
#include "quant_helper.h"

void qlinadd(int8_t* qa, float SaSy, int8_t Za,
            int8_t* qb, float SbSy, int8_t Zb,
            int8_t* out_qy, int8_t Zy, int size);

#endif // __QLINADD__