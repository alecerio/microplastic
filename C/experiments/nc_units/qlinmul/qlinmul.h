// -------------------------------------------------------
//                         QLINMUL
// -------------------------------------------------------

#include "ncquant.h"

#define QLINMUL_OUTPUT_SIZE (4)
#define QLINMUL_ZA (0)
#define QLINMUL_ZB (0)
#define QLINMUL_ZY (0)
#define QLINMUL_SA (0.02f)
#define QLINMUL_SB (0.02f)
#define QLINMUL_SY (0.02f)
#define QLINMUL_SABY ((QLINMUL_SA * QLINMUL_SB) / QLINMUL_SY)

static int8_t qlinmul_output[QLINMUL_OUTPUT_SIZE];

// -------------------------------------------------------

void run_inference(int8_t* tensor_a, int8_t* tensor_b, int8_t* tensor_y);