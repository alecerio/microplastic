#include "qsigmoid.h"

int main() {
    int8_t input[QSIGMOID_OUTPUT_SIZE] = {
        -5, 0, 10, 20
    };

    int8_t output[QSIGMOID_OUTPUT_SIZE];

    run_interference(input, output);
    
    NCAST_PRINT_MAT(output, 1, QSIGMOID_OUTPUT_SIZE, %d, "qsigmoid output")
}