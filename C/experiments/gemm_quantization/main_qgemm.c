#include "quant_helper.h"
#include "qgemm.h"

#define INPUT_SIZE (4)
#define OUTPUT_SIZE (3)
#define SIZE_LIN (INPUT_SIZE * OUTPUT_SIZE)

static int8_t qw[SIZE_LIN] = {
    122, -28, 66, -87,
    0, 76, -50, 119,
    0, -127, 94, -85
};

static int32_t qb[OUTPUT_SIZE] = {
    -35547, 23187, 10704
};

int main() {
    float input[INPUT_SIZE] = {
        0.0f, -0.01f, 0.02f, -0.03f
    };

    int8_t q_input[INPUT_SIZE]; 
    quantize(input, q_input, INPUT_SIZE, 0.003917609341442585, 0);

    int8_t qgemm_res[OUTPUT_SIZE];
    qgemm(0.003917609341442585, 0, qw, 0.0033770152367651463, qb, 0.006249960046261549, 117, OUTPUT_SIZE, INPUT_SIZE, qgemm_res, q_input);

    float res[OUTPUT_SIZE];
    dequantize(qgemm_res, res, OUTPUT_SIZE, 0.006249960046261549, 117);

    PRINT_MAT(res, 1, OUTPUT_SIZE, %f, "DEQUANTIZED OUTPUT")
    
    return 0;
}