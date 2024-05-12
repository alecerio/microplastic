#include "qlinmul.h"

int main() {

    int8_t input1[QLINMUL_OUTPUT_SIZE] = {
        -50, -20, 4, 20
    };

    int8_t input2[QLINMUL_OUTPUT_SIZE] = {
        -20, 12, 20, 50
    };

    int8_t output[QLINMUL_OUTPUT_SIZE];

    run_inference(input1, input2, output);

    NCAST_PRINT_MAT(output, 1, QLINMUL_OUTPUT_SIZE, %d, "qlinmul output")

}