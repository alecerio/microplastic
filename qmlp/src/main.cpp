#include <Arduino.h>
#include <time.h>
#include <stdlib.h>
#include "qmlp.h"
#include <math.h>

#define NUM_ITERATIONS (10000)

// ##########################################################
//        THIS SECTION IS FOR I/O VARIABLES DECLARATIONS
// ##########################################################

#define INPUT_SIZE (4) //41
float tensor_input[INPUT_SIZE];

#define OUTPUT_SIZE (2)
float tensor_output[OUTPUT_SIZE];

double exp_collector[NUM_ITERATIONS];

unsigned long start_time, end_time; // instead of clock_t
double cpu_time_used;


// ##########################################################

void setup() {
    Serial.begin(9600);
    srand(time(NULL));
  
   
}

void loop() {

    for (int i = 0; i < INPUT_SIZE; i++) {
        float val = (float)rand() / RAND_MAX;
        tensor_input[i] = val;
    }

    // run();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start_time = micros();
        run_inference(tensor_input, tensor_output);
        end_time = micros();
        cpu_time_used = ((double)(end_time - start_time)) / 1000000.0;
        exp_collector[i] = cpu_time_used;
    }

    //postprocessing();
   double sum = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double sum_of_squares = 0.0;
    double standard_deviation = 0.0;

    // Calcolo della somma e del massimo
    for(int i=0; i<NUM_ITERATIONS; i++) {
        sum += exp_collector[i];
        if(exp_collector[i] > max)
            max = exp_collector[i];
    }

    // Calcolo della media
    mean = sum / NUM_ITERATIONS;

    // Calcolo della somma dei quadrati delle differenze dalla media
    for(int i=0; i<NUM_ITERATIONS; i++) {
        sum_of_squares += pow(exp_collector[i] - mean, 2);
    }

    // Calcolo della deviazione standard
    standard_deviation = sqrt(sum_of_squares / NUM_ITERATIONS);

    //Serial.print("average: "); max dev_st
    Serial.print(mean, 20);
    Serial.print(" ");
    Serial.print(max, 20);
    Serial.print(" ");
    Serial.println(standard_deviation, 20);
}


 

