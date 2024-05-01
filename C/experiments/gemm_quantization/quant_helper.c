#include "quant_helper.h"

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
