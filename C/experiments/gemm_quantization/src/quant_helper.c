#include "quant_helper.h"

void quantize(float* W, int8_t* Wq, int size, float S, int8_t Z) {
    for(int i=0; i<size; i++) {
        int32_t temp = ROUND((W[i]/S)+Z);
        
        #ifdef QUANT_DEBUG
        CHECK_OVERFLOW_INT8(temp)
        #endif
        
        CLIP_INT8(temp)
        Wq[i] = (int8_t) temp;
    }
}

void dequantize(int8_t* Wq, float* W, int size, float S, int8_t Z) {
    for(int i=0; i<size; i++) {
        W[i] = S * ((int32_t)Wq[i] - Z);
    }
} 

void quant_error(float* W_rec, float* W, float* We, int size) {
    for(int i=0; i<size; i++) {
        float e = W_rec[i] - W[i];
        We[i] = ABS(e);
    }
}
