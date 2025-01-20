//
// Created by Inmoresentum on 1/20/25.
//

#include <unity.h>
#include <Arduino.h>
#include <esp32_tflm_wrapper.h>
#include "dummy_mlp_float32.h"

#define ARENA_SIZE 4700
#define TF_NUM_OPS 2
#define TF_NUM_INPUTS 13
#define TF_NUM_OUTPUTS 4

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> testMLP;

// Test inputs
float test_inputs[] = {
    60.000000, 1.000000, 58.000000, 58.000000, 72.000000, 454.000000, 445.000000,
    23.000000, 30.000000, 10.000000, 219.000000, 255.000000, 446.000000
};

void setUp(void) {
    // Initialize the model before each test
    testMLP.setNumInputs(TF_NUM_INPUTS);
    testMLP.setNumOutputs(TF_NUM_OUTPUTS);
    testMLP.resolver.AddFullyConnected();
    testMLP.resolver.AddSoftmax();
    TEST_ASSERT_TRUE(testMLP.begin(GeneratedCHeaderFile_simple_mlp_for_testing).isOk());
}

void tearDown(void) {
    // Clean up after each test if needed
}

void test_model_initialization() {
    TEST_ASSERT_EQUAL(TF_NUM_INPUTS, testMLP.numInputs);
    TEST_ASSERT_EQUAL(TF_NUM_OUTPUTS, testMLP.numOutputs);
}

void test_prediction_success() {
    TEST_ASSERT_TRUE(testMLP.predict(test_inputs).isOk());
}

void test_prediction_class() {
    testMLP.predict(test_inputs);
    TEST_ASSERT_EQUAL(2, testMLP.classification);
}

void test_prediction_output_range() {
    testMLP.predict(test_inputs);
    // Check if all output probabilities are between 0 and 1 (SOFTMAX Activiation Function)
    for (u_int16_t i = 0; i < TF_NUM_OUTPUTS; i++) {
        TEST_ASSERT_TRUE(testMLP.outputs[i] >= 0.0f && testMLP.outputs[i] <= 1.0f);
    }
}

void setup() {
    delay(2000);  // Allow some time for board initialization
    UNITY_BEGIN();

    RUN_TEST(test_model_initialization);
    RUN_TEST(test_prediction_success);
    RUN_TEST(test_prediction_class);
    RUN_TEST(test_prediction_output_range);

    UNITY_END();
}

void loop() {
    // Nothing to do here
}
