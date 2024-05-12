#include "qlinadd.h"

void run_inference(int8_t* input1, int8_t* input2, int8_t* output) {
    for(int i=0; i<QLINADD_OUTPUT_SIZE; i++) {
        int32_t qaz_i = (int32_t)input1[i] - (int32_t)QLINADD_ZA;
        int32_t qbz_i = (int32_t)input2[i] - (int32_t)QLINADD_ZB;
        int32_t res_i = (int32_t)(QLINADD_SAY*qaz_i) + (int32_t)(QLINADD_SBY*qbz_i) + QLINADD_ZY;
        
        NCAST_CLIP_INT8(res_i)
        output[i] = (int8_t)res_i;
    }
}