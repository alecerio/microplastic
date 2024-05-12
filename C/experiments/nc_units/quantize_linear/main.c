#include "qlinear.h"

int main() {
    float input[4] = {
        0.01f, -0.02f, 0.03f, 0.04f
    };

    int8_t output[4];

    run_inference(input, output);

    NCAST_PRINT_MAT(output, 1, 4, %d, "q_output")
    NCAST_DQUANT8(output, input, 4, QLIN_S, QLIN_Z)
    NCAST_PRINT_MAT(input, 1, 4, %f, "rec output")
}