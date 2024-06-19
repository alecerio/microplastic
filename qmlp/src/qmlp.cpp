// *****************************************************************************
// 	THIS CODE WAS AUTOMATICALLY GENERATED ON 2024-05-27 11:03:12
// *****************************************************************************

#include "qmlp.h"

static int8_t tensor_netnet0Gemm_output_0_zero_point[1] = {-128, };

static float32_t tensor_netnet0Gemm_output_0_scale[1] = {0.021465769, };

static int8_t tensor_input_zero_point[1] = {4, };

static float32_t tensor_input_scale[1] = {0.0141514605, };

static float32_t tensor_net0weight_scale[1] = {0.008824298, };

static int8_t tensor_net0weight_zero_point[1] = {0, };

static int8_t tensor_net0weight_quantized[328] = {27, 38, 55, 43, 40, -46, 1, 5, 6, 50, 68, -2, -69, -18, -66, -8, -79, -74, -72, -63, -55, 15, -22, -89, 80, -29, -18, -19, -67, -50, -4, -52, -21, 3, 9, -58, 22, -11, 29, 76, -37, 13, 43, -1, -62, 69, 5, 50, 30, 38, -24, 7, -43, -14, 15, -35, 17, -3, 16, -98, -71, -39, -61, -60, -53, -72, -66, 34, -39, -34, -76, 5, 32, -22, -70, -11, 10, -23, -38, -17, -58, 40, 3, -87, -10, 1, -33, 14, -22, -47, 32, 31, -9, 23, 13, 5, 1, -93, -30, -107, -1, -17, -83, -80, -87, -25, -85, 31, 64, 50, -32, -45, 70, 2, 35, 29, 63, 68, 33, 63, 50, -4, 52, -32, 21, -8, -44, -23, 40, -40, 62, 1, -5, -3, -10, 39, -31, 38, -65, 5, -71, -76, -42, -44, -59, -23, -35, -42, 64, -31, -64, 8, 79, -53, 54, 50, -46, 74, -54, -74, -39, -30, -15, -19, 17, -31, 20, -26, 18, -52, -4, 47, -65, -37, -6, 50, 82, -28, -49, -49, -71, 46, -9, -127, -71, -58, -20, -44, -78, 48, -47, -14, 17, 19, 45, -19, -17, -20, -13, 38, 46, 58, -25, 16, -64, 47, -50, 15, -7, -5, 9, -4, 20, 33, 3, -3, -43, -47, -28, 52, -15, -7, -103, -32, -76, -122, -95, -47, -23, 9, -6, 17, 30, 67, -7, -59, -11, 46, 4, -20, 62, -10, -40, 100, 39, 27, 10, -63, -13, 48, -4, -57, 15, -26, -65, 39, 15, 51, -11, 2, -29, -19, -60, -50, -80, -25, -12, -39, -127, -2, 20, -20, -54, -7, -49, 19, -33, -37, -20, 59, -35, 9, 10, -13, 77, 0, 85, -50, 9, -70, -25, -74, 27, -8, 1, -43, 17, 15, 8, -7, -49, 20, 76, -120, -24, -40, -92, -67, -62, -33, -38, -43, -62, -36, -1, 119, -10, 25, -21, 27, 33, 43, -44, 65, 17, -51, 35, 21, };

static int32_t tensor_net0bias_quantized[8] = {9816, 13928, 11636, 6045, 8849, 13160, 10534, 8131, };

static int8_t tensor_output_zero_point[1] = {-4, };

static float32_t tensor_output_scale[1] = {0.54042983, };

static float32_t tensor_net2weight_scale[1] = {0.024885697, };

static int8_t tensor_net2weight_zero_point[1] = {0, };

static int8_t tensor_net2weight_quantized[16] = {74, 115, 106, 120, 107, 114, 95, 127, -89, -121, -91, -122, -92, -103, -89, -123, };

static int32_t tensor_net2bias_quantized[2] = {-5902, 6673, };


// -------------------------------------------------------
//                         QLIN input_QuantizeLinear
// -------------------------------------------------------

#include "ncquant.h"

#define QLIN_input_QuantizeLinear_OUTPUT_SIZE (41)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_input_quantized[QLIN_input_QuantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM netnet0Gemm_quant
// -------------------------------------------------------

#define QGEMM_netnet0Gemm_quant_OUTPUT_SIZE (8)
#define QGEMM_netnet0Gemm_quant_INPUT_SIZE (41)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_netnet0Gemm_output_0_quantized[QGEMM_netnet0Gemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//                         QGEMM netnet2Gemm_quant
// -------------------------------------------------------

#define QGEMM_netnet2Gemm_quant_OUTPUT_SIZE (2)
#define QGEMM_netnet2Gemm_quant_INPUT_SIZE (8)

#define CONNECTED_OUTPUT
#ifdef CONNECTED_OUTPUT
static int8_t tensor_output_quantized[QGEMM_netnet2Gemm_quant_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif

// -------------------------------------------------------
//             DEQUANTIZE LINEAR output_DequantizeLinear
// -------------------------------------------------------

#define DEQUANT_output_DequantizeLinear_OUTPUT_SIZE (2)


#ifdef CONNECTED_OUTPUT
static float tensor_output[DEQUANT_output_DequantizeLinear_OUTPUT_SIZE];
#undef CONNECTED_OUTPUT
#endif
void run_inference(float32_t* tensor_input, float32_t* tensor_output) {



// -------------------------------------------------------
//                         QLIN input_QuantizeLinear
// -------------------------------------------------------

NCAST_QUANT8(tensor_input, tensor_input_quantized, QLIN_input_QuantizeLinear_OUTPUT_SIZE, tensor_input_scale[0], tensor_input_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT $NAME -----------------\n");
for(int i=0; i<QLIN_input_QuantizeLinear_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_input_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif






// QGEMM OPERATOR netnet0Gemm_quant

{
    float sw = tensor_net0weight_scale[0];
    float sx = tensor_input_scale[0];
    float sy = tensor_netnet0Gemm_output_0_scale[0];
    float swxy = (sw*sx)/sy;
    int32_t temp;
    for(int i=0; i<QGEMM_netnet0Gemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_netnet0Gemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_net0weight_quantized[i*QGEMM_netnet0Gemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_input_quantized[j]-(int32_t)tensor_input_zero_point[0]);        
        }
        temp += tensor_net0bias_quantized[i];

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_netnet0Gemm_output_0_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_netnet0Gemm_output_0_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT netnet0Gemm_quant -----------------\n");
for(int i=0; i<QGEMM_netnet0Gemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_netnet0Gemm_output_0_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif






// QGEMM OPERATOR netnet2Gemm_quant
{
    float sw = tensor_net2weight_scale[0];
    float sx = tensor_netnet0Gemm_output_0_scale[0];
    float sy = tensor_output_scale[0];
    float swxy = (sw*sx)/sy;
    int32_t temp;
    for(int i=0; i<QGEMM_netnet2Gemm_quant_OUTPUT_SIZE; i++) {
        temp = 0;
        for(int j=0; j<QGEMM_netnet2Gemm_quant_INPUT_SIZE; j++) {
            temp += (int32_t)tensor_net2weight_quantized[i*QGEMM_netnet2Gemm_quant_INPUT_SIZE+j] * ((int32_t)tensor_netnet0Gemm_output_0_quantized[j]-(int32_t)tensor_netnet0Gemm_output_0_zero_point[0]);        
        }
        temp += tensor_net2bias_quantized[i];

        int32_t res_i = ((int32_t)NCAST_ROUND(swxy * temp)) + (tensor_output_zero_point[0]);

        NCAST_CLIP_INT8(res_i)
        tensor_output_quantized[i] = (int8_t)res_i;
    }
}



#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT netnet2Gemm_quant -----------------\n");
for(int i=0; i<QGEMM_netnet2Gemm_quant_OUTPUT_SIZE; i++) {
    printf("%d ", tensor_output_quantized[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif

// -------------------------------------------------------
//             DEQUANTIZE LINEAR output_DequantizeLinear
// -------------------------------------------------------

NCAST_DQUANT8(tensor_output_quantized, tensor_output, DEQUANT_output_DequantizeLinear_OUTPUT_SIZE, tensor_output_scale[0], tensor_output_zero_point[0])

#ifdef COMPILER_DEBUG
printf("----------------- DEBUG OUTPUT output_DequantizeLinear -----------------\n");
for(int i=0; i<2; i++) {
    printf("%f ", tensor_output[i]);
}
printf("\n");
printf("------------------------------------------------------\n\n");
#endif
}