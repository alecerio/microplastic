#include "quant_helper.h"
#include "qsoftmax.h"

#define INPUT_SIZE (4)

int main() {
    
    float x[INPUT_SIZE] = {
        0.0f, -0.01f, 0.02f, -0.03f
    };

    int8_t qx[INPUT_SIZE];
    quantize(x, qx, INPUT_SIZE, 0.003917609341442585, 0);

    float sm[INPUT_SIZE];
    float exps[INPUT_SIZE];
    float dq[INPUT_SIZE];
    qsoftmax(qx, sm, exps, dq, INPUT_SIZE);

    dequantize(qx, dq, INPUT_SIZE, 0.003917609341442585, 0);
    PRINT_MAT(dq, 1, INPUT_SIZE, %f, "OUTPUT");

    return 0;
}
