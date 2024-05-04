#include "qmlp.h"

int main() {
    float input[4] = {
        0.0f, -0.01f, 0.02f, -0.03f
    };

    float output[3];
    run_inference(input, output);
}