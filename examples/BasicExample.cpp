#include <Arduino.h>
#include "model.h" // Your model
#include "esp32_tflm_wrapper.h" // include this wrapper

// this is the memory (in bytes) that will be allocated to
// tflm interpreter
#define ARENA_SIZE 5800
// number of TF ops your model utilizes
#define TF_NUM_OPS 2


Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> simpleMLP;

void setup() {
    Serial.begin(115200);
    Serial.println("__TENSORFLOW MLP__");
    // network configuration as per your model
    simpleMLP.setNumInputs(10);
    simpleMLP.setNumOutputs(5);
    simpleMLP.resolver.AddFullyConnected();
    simpleMLP.resolver.AddSoftmax();
    // Done
    delay(3000);
    // Here `g_model` is the name of the C char array generated via xxD
    while (!simpleMLP.begin(g_model).isOk())
        Serial.println(simpleMLP.exception.toString());
}

// Your input array.
// In this example we are utilizing a 8-bit integer quantized model. Hence `int8_t` as input
int8_t x1[] = {1, 3, 2, 3, 4, 5, 6, 7, 9, 1};

void loop() {
    if (!simpleMLP.predict(x1).isOk()) {
        Serial.println(simpleMLP.exception.toString());
        return;
    }

    Serial.print("it should print some number");
    Serial.println(simpleMLP.classification);
    // Now you can do something with the classification
    delay(1000);
}
