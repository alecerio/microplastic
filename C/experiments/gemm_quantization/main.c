#include "quant_helper.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define SIZE_X (4)
#define SIZE_Y (4)
#define SIZE_LIN (SIZE_X*SIZE_Y)

static float W[SIZE_LIN] = {
    2.09f, -0.98f, 1.48f, 0.09f,
    0.05f, -0.14f, -1.08f, 2.12f,
    -0.91f, 1.92f, 0.0f, -1.03f,
    1.87f, 0.0f, 1.53f, 1.49f
};

int main() {

    PRINT_MAT(W, SIZE_X, SIZE_Y, %f, "FLOAT32 MATRIX");

    float r_max, r_min; RANGE_R(W, SIZE_LIN, r_max, r_min)
    int8_t q_min, q_max; NBITS_RANGE(2, q_min, q_max)
    float S; SCALF(r_max, r_min, q_max, q_min, S)
    int8_t Z; ZERO(q_min, r_min, S, Z)

    int8_t Wq[SIZE_LIN]; quantize(W, Wq, SIZE_LIN, S, Z);
    PRINT_MAT(Wq, SIZE_X, SIZE_Y, %d, "QUANTIZED INT8 MATRIX")

    float W_rec[SIZE_LIN];
    dequantize(Wq, W_rec, SIZE_LIN, S, Z);
    PRINT_MAT(W_rec, SIZE_X, SIZE_Y, %f, "FLOAT32 MATRIX RECONSTRUCTED")

    float W_err[SIZE_LIN];
    quant_error(W_rec, W, W_err, SIZE_LIN);
    PRINT_MAT(W_err, SIZE_X, SIZE_Y, %f, "QUANTIZATION ERROR")

    return 0;
}