#include "quant_helper.h"
#include "qgemm.h"
#include <math.h>

#define INPUT_SIZE (4)
#define OUTPUT_SIZE (3)
#define SIZE_LIN (INPUT_SIZE * OUTPUT_SIZE)

static int8_t qw1[SIZE_LIN] = {
    -60, 59, -102, 81,
    127, -91, 103, 52,
    31, -18, -51, -50
};

static int8_t qw2[2*3] = {
    127, 10, -56, 
    -95, -100, 96
};

static int32_t qb1[OUTPUT_SIZE] = {
    -6286, -27727, -18030
};

static int32_t qb2[2] = {
    -56203, -53056
};

int main() {
    float input[INPUT_SIZE] = {
        0.0f, -0.01f, 0.02f, -0.03f
    };

    int8_t q_input[INPUT_SIZE]; 
    quantize(input, q_input, INPUT_SIZE, 0.003917609341442585, 0);

    int8_t qgemm1_res[OUTPUT_SIZE];
    qgemm(0.003903175937011838, 0, qw1, 0.003805271815508604, qb1, 0.00199624034576118, 0, 3, 4, qgemm1_res, q_input);
    
    int8_t qgemm2_res[2];
    qgemm(0.00199624034576118, 0, qw2, 0.003978593274950981, qb2, 0.0024469024501740932, 255, 2, 3, qgemm2_res, qgemm1_res);

    float res[2];
    dequantize(qgemm2_res, res, 2, 0.0024469024501740932, 255);

    // softmax
    float output[2];
    float exps[2];
    float sum_exps = 0.0f;
    for(int i=0; i<2; i++) {
        exps[i] = expf(res[i]-res[1]);
        printf("%f ", exps[i]);
        sum_exps += exps[i];
    }
    printf("\n");
    for(int i=0; i<2; i++) {
        output[i] = exps[i] / sum_exps;
    }

    PRINT_MAT(output, 1, 2, %f, "OUTPUT");
    
    return 0;
}