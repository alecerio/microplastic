#ifndef __QUANT_HELPER__
#define __QUANT_HELPER__

#define QUANT_DEBUG

#include <stdio.h>
#include <stdint.h>
#include <limits.h>

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

#define RANGE_R(W, SIZE_LIN, r_max, r_min) \
REL_OP(W, SIZE_LIN, r_max, >) \
REL_OP(W, SIZE_LIN, r_min, <)

#define NBITS_RANGE(N, MIN, MAX) \
MIN = -powl(2, N-1); \
MAX = powl(2, N-1)-1;

#define ROUND(X) \
(X >= 0) ? (int32_t)(X + 0.5) : (int32_t)(X - 0.5);

#define ABS(X) \
(X > 0) ? X : (-X)

#define SCALF(r_max, r_min, q_max, q_min, S) \
S = (r_max - r_min) / (q_max - q_min);

#define ZERO(q_min, r_min, S, Z) \
Z = ROUND(q_min - (r_min/S));

#define CLIP_INT8(X) \
if(X < INT8_MIN) \
    X = INT8_MIN; \
else if(X >= INT8_MAX-1) \
    X = INT8_MAX;

#define CHECK_OVERFLOW(X, MIN, MAX) \
if(X < MIN || X >= MAX) \
    printf("OVERFLOW INT8 DETECTED: %d\n", X);

#define CHECK_OVERFLOW_INT8(X) \
CHECK_OVERFLOW(X, INT8_MIN, INT8_MAX)

void quantize(float* W, int8_t* Wq, int size, float S, int8_t Z);
void dequantize(int8_t* Wq, float* W, int size, float S, int8_t Z);
void quant_error(float* W, float* Wq, float* We, int size);

#endif // __QUANT_HELPER__