#include "qmlp.h"

int main() {
    float input[4] = {
        0.0f, -0.01f, 0.02f, -0.03f
    };

    float output[2];
    run_inference(input, output);

    for(int i=0; i<2; i++) {
        printf("%f, ", output[i]);
    }
    printf("\n");
}