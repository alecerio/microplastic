
// -------------------------------------------------------
//                         QGEMM
// -------------------------------------------------------

#include "ncquant.h"

#define QGEMM_$(NAME)_OUTPUT_SIZE (4)
#define QGEMM_$(NAME)_INPUT_SIZE (3)

static int8_t tensor_$(OUTPUT_NAME)[QGEMM_$(NAME)_OUTPUT_SIZE];
