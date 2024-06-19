#include <omp.h>

#define NUM_ITERATIONS (10000)

void setup();
void postprocessing();
void run();

// ##########################################################
//        THIS SECTION IS FOR I/O VARIABLES DECLARATIONS
// ##########################################################

// ##########################################################

int main() {
    setup();
    run();
    postprocessing();
}

void setup() {

}

void run() {
    for(int i=0; i<NUM_ITERATIONS; i++) {
        double start_benchmark = omp_get_wtime();
        run_inference(in_noisy, in_hidden1, in_hidden2, output, out_hidden1, out_hidden2);
        double end_benchmark = omp_get_wtime();
        double time = end_benchmark - start_benchmark;
    }
}

void postprocessing() {
    
}



