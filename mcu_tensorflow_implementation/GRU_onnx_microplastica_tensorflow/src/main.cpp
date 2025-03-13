#include <Arduino.h>
#undef DEFAULT // solo per M5SticKCPlus altrimenti commentare
#include <time.h>
#include <stdlib.h>
#include "qgru_model.h"
#include <math.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_allocator.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#define NUM_ITERATIONS (10000)
#define INPUT_SIZE (41)
#define HIDDEN_SIZE (8)
#define OUTPUT_SIZE (2)
#define TENSOR_ARENA_SIZE (10 * 1024)  // Dimensione dell'arena di memoria

float tensor_input[INPUT_SIZE];
float tensor_hidden[HIDDEN_SIZE];
float tensor_output[OUTPUT_SIZE];

double exp_collector[NUM_ITERATIONS];
unsigned long start_time, end_time;
double cpu_time_used;

// Buffer di memoria per il modello TensorFlow Lite
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Dichiarazione dell'interprete TensorFlow Lite
static tflite::MicroInterpreter *interpreter = nullptr;

void setup() {
    Serial.begin(9600);
    srand(time(NULL));

    // Configurazione del reporter degli errori
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Caricare il modello
    const tflite::Model* model = tflite::GetModel(qgru_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Versione del modello non compatibile!");
        while (1);
    }

    // Risolutore degli operatori per il modello
    static tflite::AllOpsResolver resolver;

    // Creazione dell'allocatore
    static tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(tensor_arena, TENSOR_ARENA_SIZE, error_reporter);

    // Creazione dell'interprete
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, allocator, error_reporter);
    interpreter = &static_interpreter;

    // Allocazione dei tensori
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Errore nell'allocazione dei tensori!");
        while (1);
    }
}

void loop() {
    // Inizializzazione degli input casuali
    for (int i = 0; i < INPUT_SIZE; i++) {
        tensor_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        tensor_hidden[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Ottieni i dettagli del tensore di input
    TfLiteTensor* input_tensor = interpreter->input(0);
    memcpy(input_tensor->data.f, tensor_input, INPUT_SIZE * sizeof(float));

    // Esegui l'inferenza NUM_ITERATIONS volte e registra i tempi
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start_time = millis();
        interpreter->Invoke();
        end_time = millis();
        cpu_time_used = static_cast<double>(end_time - start_time) / 1000.0;
        exp_collector[i] = cpu_time_used;
    }

    // Ottieni i dettagli del tensore di output
    TfLiteTensor* output_tensor = interpreter->output(0);
    memcpy(tensor_output, output_tensor->data.f, OUTPUT_SIZE * sizeof(float));

    // Post-processing per calcolare statistiche sui tempi
    double sum = 0.0, max = 0.0, mean = 0.0;
    double sum_of_squares = 0.0, standard_deviation = 0.0;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        sum += exp_collector[i];
        if (exp_collector[i] > max)
            max = exp_collector[i];
    }
    
    mean = sum / NUM_ITERATIONS;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        sum_of_squares += pow(exp_collector[i] - mean, 2);
    }
    standard_deviation = sqrt(sum_of_squares / NUM_ITERATIONS);

    // Stampare i risultati
    Serial.print("Average: ");
    Serial.print(mean, 6);
    Serial.print(" s, Max: ");
    Serial.print(max, 6);
    Serial.print(" s, Std Dev: ");
    Serial.println(standard_deviation, 6);
}
