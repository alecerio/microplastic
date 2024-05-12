#include "softmax.h"

void run_inference(float* input, float* output) {

    float max_x; NCAST_REL_OP(input, SOFTMAX_OUTPUT_SIZE, max_x, >)

    float sum_exps = 0.0f;
    for(int i=0; i<SOFTMAX_OUTPUT_SIZE; i++) {
        float input_val = input[i] - max_x;
        NCAST_ACCESS_LUT(NCAST_EXP_LUT, input_val, softmax_exps_temp[i], NCAST_EXP_MINRANGE, NCAST_EXP_MAXRANGE, NCAST_EXP_UPPER, NCAST_EXP_LOWER, SOFTMAX_OUTPUT_SIZE)
        sum_exps += softmax_exps_temp[i];
    }
    
    for(int i=0; i<SOFTMAX_OUTPUT_SIZE; i++) {
        output[i] = softmax_exps_temp[i] / sum_exps;
    }

}