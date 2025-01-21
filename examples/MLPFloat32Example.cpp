#include <Arduino.h>
#include <esp32_tflm_wrapper.h> // include this wrapper which brings all necessary tflm deps
#include "dummy_mlp_float32.h" // included (comes with this lib) dummy float 32 MLP model

#define ARENA_SIZE 4700 // arena size allocation
#define TF_NUM_OPS 2 // number of ops this model uses
#define TF_NUM_INPUTS 13 // number of inputs the model expects
#define TF_NUM_OUTPUTS 4 // number of classes the model classify

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> simpleMLP;

void setup() {
    Serial.begin(115200);
    Serial.println("__TENSORFLOW FP 32 MLP__");
    delay(3000);


    simpleMLP.setNumInputs(TF_NUM_INPUTS);
    simpleMLP.setNumOutputs(TF_NUM_OUTPUTS);

    // network configuration: the two ops the model requires to run
    simpleMLP.resolver.AddFullyConnected();
    simpleMLP.resolver.AddSoftmax();
    // done with network configuration


    while (!simpleMLP.begin(GeneratedCHeaderFile_simple_mlp_for_testing).isOk())
        Serial.println(simpleMLP.exception.toString());
}

float inputs[] = {
    60.000000, 1.000000, 58.000000, 58.000000, 72.000000, 454.000000, 445.000000, 23.000000, 30.000000, 10.000000,
    219.000000, 255.000000, 446.000000
};

void loop() {
    if (!simpleMLP.predict(inputs).isOk()) {
        Serial.println(simpleMLP.exception.toString());
        return;
    }
    Serial.print("predicted class ");
    Serial.println(simpleMLP.classification);
    delay(1000);
}
