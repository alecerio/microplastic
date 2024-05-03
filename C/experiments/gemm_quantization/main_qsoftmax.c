#include "quant_helper.h"
#include "qsoftmax.h"

#define INPUT_SIZE (4)

void qsoftmax(int8_t* qx, float* sm, float* exps, float* dq, int size);

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

float access_softmax_lut(float x) {
    if(x < EXP_LUT_X[0])
        return EXP_LUT_Y[0];
    else if(x > EXP_LUT_X[EXP_LUT_SIZE-1])
        return EXP_LUT_Y[EXP_LUT_SIZE-1];
    else {
        for(int i=0; i<EXP_LUT_SIZE; i++) {
            if(x >= EXP_LUT_X[i] && x < EXP_LUT_X[i+1]) {
                float x0 = EXP_LUT_X[i];
                float x1 = EXP_LUT_X[i+1];
                float y0 = EXP_LUT_Y[i];
                float y1 = EXP_LUT_Y[i+1];
                return y0 + (y1 - y0) * ((x - x0) / (x1 - x0));
            }
        }
    }
}

void qsoftmax(int8_t* qx, float* sm, float* exps, float* dq, int size) {
    
    dequantize(qx, dq, size, 0.003917609341442585, 0);

    float max_x = dq[0];
    for(int i=1; i<size; i++) {
        if(dq[i] > max_x)
            max_x = dq[i];
    }

    float sum_exps = 0.0f;
    for(int i=0; i<size; i++) {
        exps[i] = access_softmax_lut(dq[i] - max_x);
        sum_exps += exps[i];
    }
    
    for(int i=0; i<size; i++) {
        sm[i] = exps[i] / sum_exps;
    }

    quantize(sm, qx, size, 0.003917609341442585, 0);
}