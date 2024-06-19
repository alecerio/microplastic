#include "softmax.h"

int main() {
    float input[SOFTMAX_OUTPUT_SIZE] = {
        0.05f, 3.0f, -0.003f, -0.002f
    };

    float output[SOFTMAX_OUTPUT_SIZE];

    run_inference(input, output);

    NCAST_PRINT_MAT(output, 1, SOFTMAX_OUTPUT_SIZE, %f, "softmax output")
}