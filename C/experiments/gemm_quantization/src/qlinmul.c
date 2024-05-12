#include "qlinmul.h"

void qlinmul(int8_t* qa, int8_t Za, int8_t* qb, int8_t Zb, int8_t* out_qy, int8_t Zy, float SaSbSy, int size) {
    for(int i=0; i<size; i++) {
        int32_t qaz_i = (int32_t)qa[i] - (int32_t)Za;
        int32_t qbz_i = (int32_t)qb[i] - (int32_t)Zb;
        int32_t res_i = (int32_t)(SaSbSy * qaz_i * qbz_i) + (int32_t)Zy;

        #ifdef QUANT_DEBUG
        CHECK_OVERFLOW_INT8(res_i)
        #endif

        CLIP_INT8(res_i)
        out_qy[i] = (int8_t)res_i;
    }
}