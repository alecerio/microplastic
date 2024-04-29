// INCLUDE

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


typedef float float32_t;

// MACROS

#define NUM_THREADS (4)

void run_inference(float32_t* tensor_onnxGemm_0, float32_t* tensor_8);
