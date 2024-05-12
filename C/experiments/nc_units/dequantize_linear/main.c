#include "dqlinear.h"

int main() {
    int8_t input[DEQUANT_OUTPUT_SIZE] = {
        -50, -20, 10, 30
    };

    float output[DEQUANT_OUTPUT_SIZE];

    run_inference(input, output);

    NCAST_PRINT_MAT(output, 1, DEQUANT_OUTPUT_SIZE, %f, "dequant output")

    NCAST_QUANT8(input, output, DEQUANT_OUTPUT_SIZE, DEQUANT_S, DEQUANT_Z)
    NCAST_PRINT_MAT(input, 1, DEQUANT_OUTPUT_SIZE, %d, "input reconstructed")
}