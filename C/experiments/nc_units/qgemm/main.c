#include "qgemm.h"

int main() {
    int8_t input[QGEMM_INPUT_SIZE] = {
        35, -20, 10
    };

    int8_t output[QGEMM_OUTPUT_SIZE];

    run_inference(input, output);

    NCAST_PRINT_MAT(output, 1, QGEMM_OUTPUT_SIZE, %d, "quantized output");
}