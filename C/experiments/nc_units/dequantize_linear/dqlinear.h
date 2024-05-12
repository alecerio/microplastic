// -------------------------------------------------------
//                         DEQUANT
// -------------------------------------------------------

#include "ncquant.h"

#define DEQUANT_S (0.002f)
#define DEQUANT_Z (0)
#define DEQUANT_OUTPUT_SIZE (4)

static float tensor_dequant_output[DEQUANT_OUTPUT_SIZE];

// -------------------------------------------------------

void run_inference(int8_t* tensor_input, float* tensor_output);
