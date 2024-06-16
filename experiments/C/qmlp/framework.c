#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include "qmlp.h"

#define NUM_ITERATIONS (100000)

void setup();
void postprocessing();
void run();

// ##########################################################
//        THIS SECTION IS FOR I/O VARIABLES DECLARATIONS
// ##########################################################

#define INPUT_SIZE (41)
float tensor_input[INPUT_SIZE];

#define OUTPUT_SIZE (2)
float tensor_output[OUTPUT_SIZE];

double exp_collector[NUM_ITERATIONS];

clock_t start_time, end_time;
double cpu_time_used;

// ##########################################################

int main() {
    setup();
    run();
    postprocessing();
}

void setup() {
    srand(time(NULL));

    for(int i=0; i<INPUT_SIZE; i++) {
        float val = rand()/ RAND_MAX;
        tensor_input[i] = val;
    }

}

void run() {
    for(int i=0; i<NUM_ITERATIONS; i++) {
        start_time = clock();
        run_inference(tensor_input, tensor_output);
        end_time = clock();
        cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
        exp_collector[i] = cpu_time_used;
    }
}

void postprocessing() {
    double sum = 0.0;
    double max = 0.0;
    for(int i=0; i<NUM_ITERATIONS; i++) {
        //printf("%.20f, ", exp_collector[i]);
        sum += exp_collector[i];
        if(exp_collector[i] > max)
            max = exp_collector[i];
    }
    printf("average: %.20f\n", sum / NUM_ITERATIONS);
    printf("max: %.20f\n", max);
}



