#include "qgemm.h"

void qgemm(float SwSx_Sy, int8_t Zx, int8_t* qw, int32_t* qb, int8_t Zy, int size_out, int size_in, int8_t* res, int8_t* qx) {
    int32_t temp;
    for(int i=0; i<size_out; i++) {
        temp = 0;
        for(int j=0; j<size_in; j++) {
            temp += (int32_t)qw[i*size_in+j] * ((int32_t)qx[j]-(int32_t)Zx);        
        }
        temp += qb[i];
        int32_t res_i = (int32_t)(SwSx_Sy * temp) + Zy;

        #ifdef QUANT_DEBUG
        CHECK_OVERFLOW_INT8(res_i)
        #endif

        CLIP_INT8(res_i)

        res[i] = (int8_t)res_i;
    }
}