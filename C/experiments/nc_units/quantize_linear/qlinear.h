
// -------------------------------------------------------
//                         QLIN
// -------------------------------------------------------

#include "ncquant.h"

#define QLIN_OUTPUT_SIZE (4)
#define QLIN_S (0.002f)
#define QLIN_Z (0)

static int8_t tensor_qlin_output[QLIN_OUTPUT_SIZE];

// -------------------------------------------------------

void run_inference(float* tensor_input, int8_t* tensor_output);