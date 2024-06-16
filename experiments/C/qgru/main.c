#include <stdio.h>
#include "qgru_model.h"

#define INPUT_SIZE (41)
#define HIDDEN_SIZE (8)
#define OUTPUT_SIZE (2)

int main() {
    float input[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++)
        input[i] = 1.0f;
    
    float hidden[HIDDEN_SIZE];
    for(int i=0; i<HIDDEN_SIZE; i++)
        hidden[i] = 1.0f;
    
    float output[OUTPUT_SIZE];

    run_inference(input, hidden, output);

    printf("output 0: %f\n", output[0]);
    printf("output 1: %f\n", output[1]);

    return 0;
}