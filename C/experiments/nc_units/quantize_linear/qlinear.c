#include "qlinear.h"

void run_inference(float* tensor_input, int8_t* tensor_output) {
    
    // -------------------------------------------------------
    //                         QLIN
    // -------------------------------------------------------

    NCAST_QUANT8(tensor_input, tensor_output, QLIN_OUTPUT_SIZE, QLIN_S, QLIN_Z)

    // -------------------------------------------------------

}