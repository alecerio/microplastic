#ifndef __QLINMUL__
#define __QLINMUL__

#include <stdint.h>
#include "quant_helper.h"

void qlinmul(int8_t* qa, int8_t Za,
            int8_t* qb, int8_t Zb,
            int8_t* out_qy, int8_t Zy, 
            float SaSbSy, int size);

#endif // __QLINMUL__