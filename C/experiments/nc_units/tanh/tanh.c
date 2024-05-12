#include "tanh.h"

void run_inference(float* input, float* output) {

    for(int i=0; i<TANH_OUTPUT_SIZE; i++) {
        float y;
        NCAST_ACCESS_LUT(NCAST_TANH_LUT, input[i], y, NCAST_TANH_LUT_MINRANGE, NCAST_TANH_LUT_MAXRANGE, NCAST_TANH_LUT_UPPER, NCAST_TANH_LUT_LOWER, NCAST_TANH_LUT_SIZE)
        output[i] = y;
    }

}