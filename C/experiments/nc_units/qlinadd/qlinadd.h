// -------------------------------------------------------
//                         QLINADD
// -------------------------------------------------------

#include "ncquant.h"

#define QLINADD_OUTPUT_SIZE (4)
#define QLINADD_ZA (0)
#define QLINADD_ZB (0)
#define QLINADD_ZY (0)
#define QLINADD_SA (0.002f)
#define QLINADD_SB (0.002f)
#define QLINADD_SY (0.002f)
#define QLINADD_SAY (QLINADD_SA / QLINADD_SY)
#define QLINADD_SBY (QLINADD_SB / QLINADD_SY)

static int8_t qlinadd_output[QLINADD_OUTPUT_SIZE];

// -------------------------------------------------------

void run_inference(int8_t* input1, int8_t* input2, int8_t* output);
