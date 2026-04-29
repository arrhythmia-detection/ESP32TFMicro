# Tutorial: Deploying MLP to ESP32 with TFLM

This tutorial guides you through the end-to-end
pipeline of deploying a Multi-Layer Perceptron (MLP)
model to an ESP32 microcontroller.
We will cover data generation,
model training, quantization,
and final deployment using the
`ESP32TFLMWrapper` library.

---

## 1. Model Training and Conversion (Python)

To deploy a model, we first need to train it on a
computer and convert it to a
format compatible with microcontrollers (TensorFlow Lite Micro).

### 1.1 Multi-Layer Perceptron (MLP)

An MLP is a type of feed-forward artificial neural network.
It consists of at least three layers of nodes: an input
layer, a hidden layer, and an output layer.
We will use a synthetic dataset for this demonstration.

### 1.2 Training a Float32 Model

A Float32 model uses 32-bit floating-point numbers for weights and activations.
It is accurate but consumes more memory.

```python
import os
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
N_SAMPLES = 1000
N_FEATURES = 20
N_CLASSES = 3
EPOCHS = 20
BATCH_SIZE = 32
MODEL_DIR = "model_output"
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.tflite")
HEADER_FILE_PATH = os.path.join(MODEL_DIR, "mlp_model.h")


def generate_data():
    X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES, n_informative=15,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)
    return X_train, X_test, y_train_onehot, y_test


def build_and_train_model(X_train, y_train_onehot):
    model = Sequential([
        Input(shape=(N_FEATURES,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_onehot, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    return model


def convert_to_tflite(model, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    return tflite_model


# Execution
X_train, X_test, y_train_onehot, y_test = generate_data()
model = build_and_train_model(X_train, y_train_onehot)
convert_to_tflite(model, TFLITE_MODEL_PATH)
```

### 1.3 Training an INT8 Quantized Model

Quantization reduces the model size by ~4x by converting
32-bit floats to 8-bit integers. This is crucial for
resource-constrained devices like the ESP32.

```python
def convert_to_quantized_tflite(model, X_train, save_path):
    def representative_data_gen():
        for i in range(100):
            yield [np.expand_dims(X_train[i], axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # For compatibility with ESP32 TFLM
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True

    tflite_quant_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_quant_model)
    return tflite_quant_model
```

---

## 2. Preparing the C Header File

Microcontrollers do not have a standard file system
to load `.tflite` files.
We must embed the model as a C array in the flash memory.

### 2.1 Conversion using `xxd`

Run the following command in your terminal:

```bash
xxd -i mlp_model_quant.tflite > mlp_model_quant.h
```

### 2.2 Optimizing the Header File

To ensure efficient memory usage and alignment, modify the generated header file as follows:

```cpp
#pragma once

// Data alignment for efficient CPU access
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

// Add 'const' to ensure the model is stored in Flash memory, not SRAM
const unsigned char mlp_model_quant_tflite[] DATA_ALIGN_ATTRIBUTE = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
```

**Why these changes?**

* **Data Alignment:** TensorFlow Lite Micro requires the 
  model data to be aligned (typically to a 4-byte boundary). This
  allows the processor to fetch data more
  efficiently and prevents alignment faults.
* **Const Attribute:** By marking the array as `const`,
  the compiler places it in the **Flash memory** (IROM). Without
  it, the model would be copied into **SRAM** at startup,
  which is very limited on the ESP32 and could lead to a crash.

---

## 3. ESP32 Deployment (PlatformIO)

### 3.1 Installation

Add the library to your `platformio.ini`:

```ini
lib_deps =
    https://github.com/arrhythmia-detection/ESP32TFMicro.git
```

### 3.2 Main Implementation

In your `main.cpp`, we use the `Eloquent::TF::Sequential` wrapper to manage the TFLM interpreter.

```cpp
#include <Arduino.h>
#include <esp32_tflm_wrapper.h>
#include "mlp_model_quant.h"

// 1. Define configuration
#define ARENA_SIZE 1800      // Memory allocated for tensors
#define TF_NUM_OPS 2         // We use FULLY_CONNECTED and SOFTMAX
#define TF_NUM_INPUTS 20      // Number of features in our synthetic data
#define TF_NUM_OUTPUTS 3      // Number of classes

// 2. Initialize the model
Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> model;

void setup() {
    Serial.begin(115200);
    delay(3000);

    model.setNumInputs(TF_NUM_INPUTS);
    model.setNumOutputs(TF_NUM_OUTPUTS);

    // 3. Register necessary operations
    // These must match the ops used in your TFLite model
    model.resolver.AddFullyConnected();
    model.resolver.AddSoftmax();

    // 4. Load the model from Flash
    if (!model.begin(mlp_model_quant_tflite).isOk()) {
        Serial.println("Model initialization failed!");
        Serial.println(model.exception.toString());
        while (true);
    }
    Serial.println("Model initialized successfully!");
}

int8_t inputs[TF_NUM_INPUTS];

void loop() {
    // Fill inputs with dummy data (scaled to INT8 range)
    for (int i = 0; i < TF_NUM_INPUTS; i++) {
        inputs[i] = static_cast<int8_t>(random(-128, 127));
    }

    // 5. Run Inference
    const auto before = micros();
    if (!model.predict(inputs).isOk()) {
        Serial.println(model.exception.toString());
        return;
    }
    const auto after = micros();

    // 6. Output result
    Serial.print("Predicted Class: ");
    Serial.println(model.classification);
    Serial.print("Inference Time: ");
    Serial.print(after - before);
    Serial.println(" us");

    delay(2000);
}
```

### 3.3 Key Parameters

* **ARENA_SIZE:** This is the "Tensor Arena," a pool of memory where TFLM stores its intermediate tensors. If this is
  too small, `model.begin()` will fail.
  Hence, increase the size if it fails due to memory constraints.
* **Ops Registration:** Only register the operations your model actually uses (e.g., `AddFullyConnected`, `AddSoftmax`)
  to save binary size.
* **INT8 Inputs:** Since we quantized the model to INT8, our input array must also be `int8_t`. Ensure your input
  features are scaled correctly based on the quantization parameters used during conversion.

---

## 4. Conclusion

By following this pipeline, you can successfully transition from a high-level Python model to a highly optimized C++
implementation running on the ESP32. This approach allows for real-time AI at the edge with minimal power consumption
and memory footprint.
