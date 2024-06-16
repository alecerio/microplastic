#ifndef __NCQUANT__
#define __NCQUANT__

#include "ncutils.h"

#define NCAST_RANGE_R(W, SIZE_LIN, r_max, r_min) \
REL_OP(W, SIZE_LIN, r_max, >) \
REL_OP(W, SIZE_LIN, r_min, <)

#define NCAST_NBITS_RANGE(N, MIN, MAX) \
MIN = -powl(2, N-1); \
MAX = powl(2, N-1)-1;

#define NCAST_SCALF(r_max, r_min, q_max, q_min, S) \
S = (r_max - r_min) / (q_max - q_min);

#define NCAST_ZERO(q_min, r_min, S, Z) \
Z = ROUND(q_min - (r_min/S));

#define NCAST_CLIP_INT8(X) \
if(X < INT8_MIN) \
    X = INT8_MIN; \
else if(X >= INT8_MAX-1) \
    X = INT8_MAX;

#define NCAST_CHECK_OVERFLOW(X, MIN, MAX) \
if(X < MIN || X >= MAX) \
    printf("OVERFLOW INT8 DETECTED: %d\n", X);

#define NCAST_CHECK_OVERFLOW_INT8(X) \
NCAST_CHECK_OVERFLOW(X, INT8_MIN, INT8_MAX)

#define NCAST_DBGQUANT8(W, WQ, SIZE, S, Z) \
for(int i=0; i<SIZE; i++) { \
    int32_t temp = NCAST_ROUND((W[i]/S)+Z); \
    NCAST_CHECK_OVERFLOW_INT8(temp) \
    NCAST_CLIP_INT8(temp) \
    WQ[i] = (int8_t) temp; \
}

#define NCAST_QUANT8(W, WQ, SIZE, S, Z) \
for(int i=0; i<SIZE; i++) { \
    int32_t temp = NCAST_ROUND((W[i]/S)+Z); \
    NCAST_CLIP_INT8(temp) \
    WQ[i] = (int8_t) temp; \
}

#define NCAST_DQUANT8(WQ, W, SIZE, S, Z) \
for(int i=0; i<SIZE; i++) { \
    W[i] = S * ((int32_t)WQ[i] - Z); \
} 

#define NCAST_QERR(WREC, W, WE, SIZE) \
for(int i=0; i<SIZE; i++) { \
    float e = WREC[i] - W[i]; \
    WE[i] = NC_ABS(e); \
}

#endif // __NCQUANT__