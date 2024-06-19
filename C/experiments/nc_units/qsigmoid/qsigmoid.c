#include "qsigmoid.h"

void run_interference(int8_t* input, int8_t* output) {
    NCAST_DQUANT8(input, qsigmoid_temp, QSIGMOID_OUTPUT_SIZE, QSIGMOID_SX, QSIGMOID_ZX)

    for(int i=0; i<QSIGMOID_OUTPUT_SIZE; i++) {
        float y;
        NCAST_ACCESS_LUT(NCAST_SIGMOID_LUT, qsigmoid_temp[i], y, NCAST_SIGMOID_LUT_MINRANGE, NCAST_SIGMOID_LUT_MAXRANGE, NCAST_SIGMOID_LUT_UPPER, NCAST_SIGMOID_LUT_LOWER, NCAST_SIGMOID_LUT_SIZE)
        qsigmoid_temp[i] = y;
    }

    NCAST_QUANT8(qsigmoid_temp, output, QSIGMOID_OUTPUT_SIZE, QSIGMOID_SY, QSIGMOID_ZY)
}