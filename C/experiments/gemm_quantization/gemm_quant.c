#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define SIZE_X (4)
#define SIZE_Y (4)
#define SIZE_LIN (SIZE_X*SIZE_Y)

#define PRINT_MAT(X, SX, SY, T, HEADER) \
printf(" ########################### \n"); \
printf(HEADER); \
printf("\n"); \
for(int i=0; i<SX; i++) { \
    for(int j=0; j<SY; j++) { \
        printf(""#T" ", X[i*SY+j]); \
    } \
    printf("\n"); \
}

#define REL_OP(X, SL, Y, OP) \
Y = X[0]; \
for(int i=1; i<SL; i++) { \
    if(X[i] OP Y) \
        Y = X[i]; \
}

#define NBITS_RANGE(N, MIN, MAX) \
MIN = -powl(2, N-1); \
MAX = powl(2, N-1)-1;

#define ROUND(X) \
(X >= 0) ? (int8_t)(X + 0.5) : (int8_t)(X - 0.5);

#define ABS(X) \
(X > 0) ? X : (-X)

static float W[SIZE_LIN] = {
    2.09f, -0.98f, 1.48f, 0.09f,
    0.05f, -0.14f, -1.08f, 2.12f,
    -0.91f, 1.92f, 0.0f, -1.03f,
    1.87f, 0.0f, 1.53f, 1.49f
};

void quantize(float* W, int8_t* Wq, int size, float S, int8_t Z);
void dequantize(int8_t* Wq, float* W, int size, float S, int8_t Z);
void quant_error(float* W, float* Wq, float* We, int size);

int main() {

    PRINT_MAT(W, SIZE_X, SIZE_Y, %f, "FLOAT32 MATRIX");

    float r_max; REL_OP(W, SIZE_LIN, r_max, >)
    float r_min; REL_OP(W, SIZE_LIN, r_min, <)
    int8_t q_min, q_max; NBITS_RANGE(2, q_min, q_max)
    float S = (r_max - r_min) / (q_max - q_min);
    int8_t Z = ROUND(q_min - (r_min/S));

    int8_t Wq[SIZE_LIN]; quantize(W, Wq, SIZE_LIN, S, Z);
    PRINT_MAT(Wq, SIZE_X, SIZE_Y, %d, "QUANTIZED INT8 MATRIX")

    float W_rec[SIZE_LIN];
    dequantize(Wq, W_rec, SIZE_LIN, S, Z);
    PRINT_MAT(W_rec, SIZE_X, SIZE_Y, %f, "FLOAT32 MATRIX RECONSTRUCTED")

    float W_err[SIZE_LIN];
    quant_error(W_rec, W, W_err, SIZE_LIN);
    PRINT_MAT(W_err, SIZE_X, SIZE_Y, %f, "QUANTIZATION ERROR")


    printf("r max: %f\n", r_max);
    printf("r min: %f\n", r_min);
    printf("q max: %d\n", q_max);
    printf("q min: %d\n", q_min);
    printf("S: %f\n", S);
    printf("Z: %d\n", Z);

    return 0;
}

void quantize(float* W, int8_t* Wq, int size, float S, int8_t Z) {
    for(int i=0; i<size; i++) {
        Wq[i] = ROUND((W[i]/S)+Z);
    }
}

void dequantize(int8_t* Wq, float* W, int size, float S, int8_t Z) {
    for(int i=0; i<size; i++) {
        W[i] = (Wq[i]-Z)*S;
    }
}

void quant_error(float* W_rec, float* W, float* We, int size) {
    for(int i=0; i<size; i++) {
        float e = W_rec[i] - W[i];
        We[i] = ABS(e);
    }
}