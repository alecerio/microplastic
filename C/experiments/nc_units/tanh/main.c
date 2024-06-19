#include "tanh.h"

int main() {
    float input[TANH_OUTPUT_SIZE] = {
        -1.5f, 0.5f, 0.0f, 0.5f
    };

    float output[TANH_OUTPUT_SIZE];

    run_inference(input, output);

    NCAST_PRINT_MAT(output, 1, TANH_OUTPUT_SIZE, %f, "tanh output")
}