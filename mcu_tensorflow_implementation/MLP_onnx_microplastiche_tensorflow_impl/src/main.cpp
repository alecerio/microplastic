#include <Arduino.h>
#undef DEFAULT // solo per M5SticKCPlus altrimenti commentare
#include <math.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "mlp_quantized_model.h" // Include il modello convertito

#define INPUT_SIZE 41
#define OUTPUT_SIZE 2
#define NUM_ITERATIONS 10000
#define TENSOR_ARENA_SIZE 4 * 1024  // 4KB per TensorFlow Lite Micro

// Buffer per TensorFlow Lite
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

double exp_collector[NUM_ITERATIONS];
unsigned long start_time, end_time;
double cpu_time_used;

double mean = 0.0;
double max_time = 0.0;
double standard_deviation = 0.0;

double sum = 0.0;
double sum_of_squares = 0.0;

void setup() {
    Serial.begin(9600);
    Serial.println("Setup TensorFlow Lite");

    model = tflite::GetModel(mlp_quantized_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Modello non compatibile!");
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Errore allocazione tensori!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Setup completato");
}

void loop() {
    for (int i = 0; i < INPUT_SIZE; i++) {
        input->data.f[i] = random(0, 100) / 100.0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start_time = micros();
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Errore esecuzione modello!");
            return;
        }
        end_time = micros();
        cpu_time_used = ((double)(end_time - start_time)) / 1000000.0;
        exp_collector[i] = cpu_time_used;
    }

    sum = 0.0;
    max_time = 0.0;
    sum_of_squares = 0.0;

    for(int i = 0; i < NUM_ITERATIONS; i++) {
        sum += exp_collector[i];
        if(exp_collector[i] > max_time)
            max_time = exp_collector[i];
    }

    mean = sum / NUM_ITERATIONS;

    for(int i = 0; i < NUM_ITERATIONS; i++) {
        sum_of_squares += pow(exp_collector[i] - mean, 2);
    }

    standard_deviation = sqrt(sum_of_squares / NUM_ITERATIONS);
    

    Serial.print("Tempo medio: ");
    Serial.print(mean, 20);
    Serial.print(" s\t Tempo massimo: ");
    Serial.print(max_time, 20);
    Serial.print(" s\t Deviazione standard: ");
    Serial.println(standard_deviation, 20);

    delay(1000);
}
