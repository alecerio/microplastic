#include "qlinadd.h"

int main() {
    int8_t a[QLINADD_OUTPUT_SIZE] = {
        -5, 0, 10, 15
    };

    int8_t b[QLINADD_OUTPUT_SIZE] = {
        -10, -5, 0, 5
    };

    int8_t c[QLINADD_OUTPUT_SIZE];

    run_inference(a, b, c);

    NCAST_PRINT_MAT(c, 1, QLINADD_OUTPUT_SIZE, %d, "qlinadd output")
}