#include "qmlp.h"

void run_inference(float* tensor_input, float* tensor_output) {

    for(int i=0; i<QUANTIZE1_SIZE; i++) {
        tensor_quantize1_output[i] = QUANT_ROUND((tensor_input[i]/QUANTIZE1_S)+QUANTIZE1_Z);
    }
    
    // GENERATE CODE FOR QGEMM1

    {
        int temp;
        float tempf;
        float c = (QGEMM1_SW * QGEMM1_SX) / QGEMM1_SY;
        for(int i=0; i<QGEMM1_SIZEOUT; i++) {
            temp = 0;
            for(int j=0; j<QGEMM1_SIZEIN; j++) {
                temp += tensor_qgemm1_wq[i*QGEMM1_SIZEIN+j] * (tensor_quantize1_output[j]-QGEMM1_ZX);        
            }
            temp += tensor_qgemm1_qb[i];
            tempf = c * (float)temp;
            tensor_qgemm1_output[i] = (int8_t)tempf + QGEMM1_ZY;
        }
    }

    // GENERATE CODE FOR QGEMM2

    {
        int temp;
        float tempf;
        float c = (QGEMM2_SW * QGEMM2_SX) / QGEMM2_SY;
        for(int i=0; i<QGEMM2_SIZEOUT; i++) {
            temp = 0;
            for(int j=0; j<QGEMM2_SIZEIN; j++) {
                temp += tensor_qgemm2_wq[i*QGEMM2_SIZELIN+j] * (tensor_qgemm1_output[j]-QGEMM2_ZX);        
            }
            temp += tensor_qgemm2_qb[i];
            tempf = c * (float)temp;
            tensor_qgemm2_output[i] = (int8_t)tempf + QGEMM2_ZY;
        }
    }

    // GENERATE CODE FOR DEQUANTIZE1

    for(int i=0; i<DEQUANTIZE1_SIZEOUT; i++) {
        int subzero = (int)tensor_qgemm2_output[i]-DEQUANTIZE1_Z;
        if (subzero < INT8_MIN) {
            subzero += UINT8_MAX+1;
            subzero %= UINT8_MAX+1;
        }
        else if(subzero > INT8_MAX)
            subzero %= UINT8_MAX+1;
        tensor_dequantize1_output[i] = subzero*DEQUANTIZE1_S;
    }

    for(int i=0; i<DEQUANTIZE1_SIZEOUT; i++) {
        printf("%f, ", tensor_dequantize1_output[i]);
    }
    printf("\n");

    // GENERATE CODE FOR SOFTMAX1

    {
        
        // dequantization
        //float tensor_dequantized[QSOFTMAX1_SIZEOUT];
        //for(int i=0; i<QSOFTMAX1_SIZEOUT; i++) {
        //    int subzero = (int)tensor_qgemm2_output[i]-QSOFTMAX1_Z;
        //    if (subzero < INT8_MIN) {
        //        subzero += UINT8_MAX+1;
        //        subzero %= UINT8_MAX+1;
        //    }
        //    else if(subzero > INT8_MAX)
        //        subzero %= UINT8_MAX+1;
        //    tensor_dequantized[i] = subzero*QSOFTMAX1_S;
        //}

        // find maximum dequantized value
        float max_x = tensor_dequantize1_output[0];
        for(int i=1; i<QSOFTMAX1_SIZEOUT; i++) {
            if(tensor_dequantize1_output[i] > max_x)
                max_x = tensor_dequantize1_output[i];
        }

        // access exp values from exp_lut
        float sum_exps = 0.0f;
        float exps[QSOFTMAX1_SIZEOUT];
        for(int i=0; i<QSOFTMAX1_SIZEOUT; i++) {
            ACCESS_EXP_LUT(tensor_dequantize1_output[i] - max_x, exps[i])
            sum_exps += exps[i];
        }
    
        // compute softmax
        //float sm[QSOFTMAX1_SIZEOUT];
        for(int i=0; i<QSOFTMAX1_SIZEOUT; i++) {
            tensor_output[i] = exps[i] / sum_exps;
        }

        for(int i=0; i<QSOFTMAX1_SIZEOUT; i++) {
            printf("%f, ", tensor_output[i]);
        }
        printf("\n");

        // re-quantize
        //for(int i=0; i<QSOFTMAX1_SIZEOUT; i++) {
        //    tensor_qsoftmax1_output[i] = QUANT_ROUND((sm[i]/QSOFTMAX1_S)+QSOFTMAX1_Z);
        //}
    }
}