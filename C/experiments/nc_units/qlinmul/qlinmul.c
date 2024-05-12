#include "qlinmul.h"

void run_inference(int8_t* tensor_a, int8_t* tensor_b, int8_t* tensor_y) {
    for(int i=0; i<QLINMUL_OUTPUT_SIZE; i++) {
        int32_t qaz_i = (int32_t)tensor_a[i] - (int32_t)QLINMUL_ZA;
        int32_t qbz_i = (int32_t)tensor_b[i] - (int32_t)QLINMUL_ZB;
        int32_t res_i = (int32_t)(QLINMUL_SABY * qaz_i * qbz_i) + (int32_t)QLINMUL_ZY;

        NCAST_CLIP_INT8(res_i)
        tensor_y[i] = (int8_t)res_i;
    }
}
