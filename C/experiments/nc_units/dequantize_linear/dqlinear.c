#include "dqlinear.h"

void run_inference(int8_t* tensor_input, float* tensor_output) {

    NCAST_DQUANT8(tensor_input, tensor_output, DEQUANT_OUTPUT_SIZE, DEQUANT_S, DEQUANT_Z)

}