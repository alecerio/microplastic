#include "qgemm.h"

void run_inference(int8_t* input, int8_t* output) {
    {
        int32_t temp;
        for(int i=0; i<QGEMM_OUTPUT_SIZE; i++) {
            temp = 0;
            for(int j=0; j<QGEMM_INPUT_SIZE; j++) {
                temp += (int32_t)tensor_qgemm_qw[i*QGEMM_INPUT_SIZE+j] * ((int32_t)input[j]-(int32_t)QGEMM_ZX);        
            }
            temp += tensor_qgemm_qb[i];
            int32_t res_i = (int32_t)(QGEMM_SWXY * temp) + QGEMM_ZY;

            NCAST_CLIP_INT8(res_i)
            output[i] = (int8_t)res_i;
        }
    }
    
}