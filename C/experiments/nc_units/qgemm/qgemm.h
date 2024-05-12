
// -------------------------------------------------------
//                         QGEMM
// -------------------------------------------------------

#include "ncquant.h"

#define QGEMM_OUTPUT_SIZE (4)
#define QGEMM_INPUT_SIZE (3)

#define QGEMM_ZX (0)
#define QGEMM_ZY (0)
#define QGEMM_SX (0.002f)
#define QGEMM_SY (0.002f)
#define QGEMM_SW (0.002f)
#define QGEMM_SWXY ((QGEMM_SW * QGEMM_SX) / QGEMM_SY)

static int8_t tensor_qgemm_qw[QGEMM_OUTPUT_SIZE * QGEMM_INPUT_SIZE] = {
    40, 40, 40,
    50, 50, 50,
    60, 60, 60,
    70, 70, 70,
};

static int32_t tensor_qgemm_qb[QGEMM_OUTPUT_SIZE] = {
    10000, 20000, 30000
};

static int8_t tensor_qgemm_output[QGEMM_OUTPUT_SIZE];

// ---------------------------------------

void run_inference(int8_t* input, int8_t* output);