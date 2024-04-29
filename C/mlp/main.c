#include "mlp.h"

#define INPUT_SIZE (41)
#define OUTPUT_SIZE (2)

int main() {
    float input_tensor[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++)
        input_tensor[i] = 1.0;
    
    float output_tensor[OUTPUT_SIZE];
    for(int i=0; i<OUTPUT_SIZE; i++)
        output_tensor[i] = 0.0f;

    run_inference(input_tensor, output_tensor);

    for(int i=0; i<OUTPUT_SIZE; i++)
        printf("%f ", output_tensor[i]);

    printf("\n");
    
    return 0;
}