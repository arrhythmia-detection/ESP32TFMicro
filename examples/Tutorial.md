# Tutorial: End-to-End Deployment of Deep Learning Models to ESP32

This tutorial provides a rigorous,
researcher-oriented guide to the full
pipeline of deploying TensorFlow-based Neural
Networks to the ESP32 microcontroller.
We utilize the `ESP32TFLMWrapper` library
to abstract the low-level complexities
of TensorFlow Lite for Microcontrollers (TFLM).

Deployment is approached in two distinct phases:

1. **Baseline Deployment**: Utilizing standard Float32 precision to validate
   the architectural integrity and logic of
   the model on the target hardware.
2. **Optimization Phase**: Implementing Full Integer Quantization (INT8) to
   reduce memory footprint and increase
   computational efficiency, critical for
   resource-constrained edge devices.

---

## 1. Phase I: Multi-Layer Perceptron (MLP) with Float32 Precision

The Multi-Layer Perceptron (MLP) serves as our foundational architecture.
Before moving to complex optimizations, we
must establish a functional baseline.

### 1.1 Model Synthesis and Training (Python)

We utilize a synthetic dataset to maintain focus on the pipeline mechanics.
The following script handles data
generation, model architecture definition, training,
and conversion to a standard TFLite format.

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
KERAS_REPORT_PATH = os.path.join(MODEL_DIR, "keras_classification_report.txt")
TFLITE_REPORT_PATH = os.path.join(MODEL_DIR, "tflite_classification_report.txt")
TFLITE_OPS_PATH = os.path.join(MODEL_DIR, "tflite_ops_list.txt")


def generate_data():
    """
    Generates a synthetic dataset for multi-class classification.
    We use make_classification from scikit-learn because it allows us to quickly generate
    a robust dataset with a specified number of classes and informative features,
    perfect for validating our model's logic before trying real-world data.
    """
    print("Generating synthetic data...")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        n_informative=15,
        random_state=42
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding (e.g., class 2 out of 3 becomes [0, 0, 1])
    # This format is required when using the 'categorical_crossentropy' loss function during training.
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)

    return X_train, X_test, y_train_onehot, y_test


def build_and_train_model(X_train, y_train_onehot):
    """
    Builds and trains a simple Multi-Layer Perceptron (MLP) model.
    The architecture uses fully connected (Dense) layers with ReLU activations.
    The final layer uses 'softmax' activation to output probabilities across the N_CLASSES.
    """
    print("\nBuilding and training MLP model...")
    model = Sequential([
        Input(shape=(N_FEATURES,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(N_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',  # Adam is a standard, efficient optimizer for most general tasks
        loss='categorical_crossentropy',  # Appropriate loss function for multi-class classification
        metrics=['accuracy']
    )

    # Train the model on the generated data
    model.fit(X_train, y_train_onehot, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    return model


def evaluate_keras_model(model, X_test, y_test, report_path):
    """
    Evaluates the standard Keras model using sklearn's classification_report
    and saves the result to a text file for reference.
    """
    print("\nEvaluating Keras Model...")

    # model.predict gives us the probabilities for each class
    y_pred_probs = model.predict(X_test)

    # np.argmax gets the index of the highest probability, which represents our predicted class
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, digits=5)
    print(report)

    # Save the classification report to a text file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Keras Model Classification Report\n")
        f.write("=================================\n")
        f.write(report)
    print(f"Saved Keras classification report to: {report_path}")


def convert_to_tflite(model, save_path):
    """
    Converts a Keras model to a Float32 TFLite model.
    TFLite models are highly optimized and have a smaller footprint, making them
    ideal for deployment on edge devices like the ESP32.
    """
    print("\nConverting model to TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # We do not apply any optimizations (quantization) here

    try:
        tflite_model = converter.convert()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to: {save_path}")
        return tflite_model
    except Exception as e:
        print(f"Failed to convert the model to TFLite: {e}")
        return None


def evaluate_tflite_model(tflite_model_content, X_test, y_test, report_path):
    """
    Evaluates the TFLite model using sklearn's classification_report and saves it to a file.
    This step is crucial to ensure that the conversion process did not heavily degrade
    the model's performance (especially important when later experimenting with quantization).
    """
    print("\nEvaluating TFLite Model...")
    try:
        # Initialize the TFLite interpreter with the model in memory
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)

        # Allocate memory for the model's tensors (nodes). This must be called before any inference.
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        y_pred = []
        # Run inference on each test sample iteratively
        for i in range(len(X_test)):
            # TFLite expects input data to be strictly typed and shaped as [batch_size, features]
            input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)

            # Feed the input data to the model
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run the prediction
            interpreter.invoke()

            # Extract the prediction result
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data)
            y_pred.append(predicted_class)

        report = classification_report(y_test, y_pred, digits=5)
        print(report)

        # Save the classification report to a text file
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("TFLite Model Classification Report\n")
            f.write("==================================\n")
            f.write(report)
        print(f"Saved TFLite classification report to: {report_path}")

    except Exception as e:
        print(f"Failed to evaluate TFLite model: {e}")


def print_tflite_ops(tflite_model_content, ops_path):
    """
    Prints and saves the unique TFLite operations used by the model.
    In C++ (TensorFlow Lite for Microcontrollers), you must explicitly register
    each operation used by your model to save memory. This list tells you exactly
    what ops to include in your micro_mutable_op_resolver.
    """
    print("\nExtracting TFLite Operations...")
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        ops_details = interpreter._get_ops_details()

        # Extract unique operations names using a set
        ops = set(op["op_name"] for op in ops_details)
        ops_list = sorted(list(ops))

        # Console output
        print(f"Total number of unique ops: {len(ops_list)}")
        print("TensorFlow Lite Ops utilized by the model:")
        for op in ops_list:
            print(f" - {op}")

        # Save output to a text file
        os.makedirs(os.path.dirname(ops_path), exist_ok=True)
        with open(ops_path, "w") as f:
            f.write(f"Total number of unique ops: {len(ops_list)}\n")
            f.write("TensorFlow Lite Ops utilized by the model:\n")
            for op in ops_list:
                f.write(f" - {op}\n")
        print(f"Saved TFLite ops list to: {ops_path}")

    except Exception as e:
        print(f"Failed to extract TFLite ops: {e}")


def convert_tflite_to_c_header(tflite_path, header_path):
    """
    Converts the compiled TFLite model to a C header file using the 'xxd' tool.
    Microcontrollers (like ESP32) don't have standard file systems. The model must
    be embedded directly into the flash memory as a constant C array.
    """
    print("\nConverting TFLite model to C header file...")
    try:
        # Execute the xxd command to generate the hex dump as a C array
        subprocess.run(f"xxd -i {tflite_path} > {header_path}", shell=True, check=True)
        print(f"Successfully generated C header file: {header_path}")
    except subprocess.CalledProcessError as e:
        print(f"xxd command failed: {e}. Please ensure 'xxd' is installed on your system.")
    except Exception as e:
        print(f"Failed to convert TFLite model to C header: {e}")


def main():
    # 1. Generate data
    X_train, X_test, y_train_onehot, y_test = generate_data()

    # 2. Build and train Keras model
    keras_model = build_and_train_model(X_train, y_train_onehot)

    # 3. Evaluate Keras model and save report
    evaluate_keras_model(keras_model, X_test, y_test, KERAS_REPORT_PATH)

    # 4. Convert to TFLite
    tflite_model = convert_to_tflite(keras_model, TFLITE_MODEL_PATH)

    if tflite_model:
        # 5. Evaluate TFLite model and save report
        evaluate_tflite_model(tflite_model, X_test, y_test, TFLITE_REPORT_PATH)

        # 6. Print utilized TFLite ops and save to file
        print_tflite_ops(tflite_model, TFLITE_OPS_PATH)

        # 7. Convert to C header using xxd
        convert_tflite_to_c_header(TFLITE_MODEL_PATH, HEADER_FILE_PATH)


if __name__ == "__main__":
    main()
```

### 1.2 Preparing the Embedded Model

Microcontrollers lack standard file systems; thus,
the model must be compiled into
the executable as a constant array.
We use `xxd` for this purpose,
but several critical modifications
are required for production-grade deployment.

#### 1.2.1 Data Alignment and Flash Placement

The TFLM interpreter requires the model buffer
to be 4-byte aligned for efficient CPU access
and to prevent alignment faults.
Furthermore, we must ensure the model
resides in **IROM (Flash)** rather
than **SRAM** to preserve the limited
runtime memory.

```cpp
#pragma once

// Macro-based attribute detection for cross-compiler compatibility
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif

// Define alignment attribute: 4-byte alignment is standard for TFLM on ESP32
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

// The 'const' keyword is mandatory to force the linker to place the data in Flash memory.
// Without it, the ESP32 bootloader will copy the entire model into SRAM at startup,
// potentially leading to Stack/Heap overflows.
const unsigned char mlp_model_tflite[] DATA_ALIGN_ATTRIBUTE = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
```

### 1.3 Embedded Implementation (PlatformIO)

The following `main.cpp` demonstrates the deployment of
the Float32 model using the `Eloquent::TF::Sequential` wrapper.

```cpp
#include <Arduino.h>
#include <esp32_tflm_wrapper.h>
#include "mlp_model.h"

// Hyperparameters for the TFLM Interpreter
#define ARENA_SIZE 2048      // Tensor Arena size in bytes. Adjust based on model complexity.
#define TF_NUM_OPS 2         // Total unique operations utilized by the model
#define TF_NUM_INPUTS 20      // Input vector dimensionality
#define TF_NUM_OUTPUTS 3      // Output class count

// Instance of the Sequential wrapper
Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> model;

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("Initializing Float32 MLP...");

    // Configure model dimensions
    model.setNumInputs(TF_NUM_INPUTS);
    model.setNumOutputs(TF_NUM_OUTPUTS);

    // Register Operations: Must exactly match the ops identified during Python evaluation.
    // Failure to register an op will result in a runtime error during model.begin().
    model.resolver.AddFullyConnected();
    model.resolver.AddSoftmax();

    // Initialize the interpreter with the model data from Flash
    if (!model.begin(mlp_model_tflite).isOk()) {
        Serial.println("Initialization Error:");
        Serial.println(model.exception.toString());
        while (true) delay(1000); // Halt execution on failure
    }
    Serial.println("Model Ready.");
}

float input_vector[TF_NUM_INPUTS];

void loop() {
    // Generate synthetic input data for verification
    for (int i = 0; i < TF_NUM_INPUTS; i++) {
        input_vector[i] = static_cast<float>(random(-100, 100)) / 100.0f;
    }

    const uint32_t start_time = micros();
    
    // Execute Inference
    if (!model.predict(input_vector).isOk()) {
        Serial.println(model.exception.toString());
        return;
    }
    
    const uint32_t end_time = micros();

    // Output Performance Metrics and Predictions
    Serial.print("Class: ");
    Serial.print(model.classification);
    Serial.print(" | Latency: ");
    Serial.print(end_time - start_time);
    Serial.println(" us");

    delay(1000);
}
```

---

## 2. Phase II: Optimization via Full Integer Quantization (INT8)

Once the Float32 baseline is verified, we apply quantization.
This process converts the model weights and activations
from 32-bit floats to 8-bit integers,
typically resulting in a 4x reduction
in size and significant latency improvements
on hardware lacking a high-performance FPU.

### 2.1 Quantization Strategy (Python)

The following script implements Post-Training Quantization (PTQ)
with a representative dataset to calibrate the dynamic
ranges of the tensors.

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
MODEL_DIR = "quantized_model_output"
TFLITE_QUANT_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model_quant.tflite")
HEADER_FILE_PATH = os.path.join(MODEL_DIR, "mlp_model_quant.h")
KERAS_REPORT_PATH = os.path.join(MODEL_DIR, "keras_classification_report.txt")
TFLITE_QUANT_REPORT_PATH = os.path.join(MODEL_DIR, "tflite_quant_classification_report.txt")
TFLITE_QUANT_OPS_PATH = os.path.join(MODEL_DIR, "tflite_quant_ops_list.txt")


def generate_data():
    """Generates synthetic dataset for classification."""
    print("Generating synthetic data...")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        n_informative=15,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)

    return X_train, X_test, y_train_onehot, y_test


def build_and_train_model(X_train, y_train_onehot):
    """Builds and trains the standard Float32 MLP model."""
    print("\nBuilding and training MLP model...")
    model = Sequential([
        Input(shape=(N_FEATURES,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(N_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train_onehot, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    return model


def evaluate_keras_model(model, X_test, y_test, report_path):
    """Evaluates the Keras model."""
    print("\nEvaluating Keras Model...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, digits=5)
    print(report)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Keras Model Classification Report\n")
        f.write("=================================\n")
        f.write(report)


def convert_to_quantized_tflite(model, X_train, save_path):
    """
    Converts a Keras model to a Fully Integer Quantized (INT8) TFLite model.
    This reduces the model size by ~4x and speeds up inference on microcontrollers
    like the ESP32 that lack dedicated floating-point units (FPU) for all operations.
    """
    print("\nConverting model to Fully Integer Quantized TFLite (INT8)...")

    # 1. Representative Dataset Generator
    # Required for INT8 quantization. It feeds a subset of training data through the model
    # to measure the dynamic ranges (min/max values) of activations in each layer.
    def representative_data_gen():
        # Provide ~100 samples to calibrate the quantization parameters
        for i in range(100):
            # The yield statement returns a single sample shaped as [batch_size, features]
            yield [np.expand_dims(X_train[i], axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 2. Set optimizations to DEFAULT (which enables quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 3. Provide the representative dataset generator
    converter.representative_dataset = representative_data_gen

    # 4. Restrict supported operations to strictly INT8 built-ins
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # 5. Enforce INT8 input and output interfaces
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # 6. Crucial for TensorFlow Lite for Microcontrollers (TFLM)
    # TFLM's FullyConnected op currently lacks broad support for per-channel quantization.
    # This experimental flag forces per-tensor quantization for Dense layers,
    # ensuring compatibility with the ESP32 TFLM interpreter.
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True

    try:
        tflite_quant_model = converter.convert()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(tflite_quant_model)
        print(f"Saved Quantized TFLite model to: {save_path}")
        return tflite_quant_model
    except Exception as e:
        print(f"Failed to quantize the model: {e}")
        return None


def evaluate_quantized_tflite_model(tflite_model_content, X_test, y_test, report_path):
    """
    Evaluates the Quantized TFLite model.
    Since we enforced INT8 input/output, we must manually scale our Float32 test data
    into INT8 before feeding it to the interpreter.
    """
    print("\nEvaluating Quantized TFLite Model...")
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Extract quantization parameters (Scale and Zero Point) for the input tensor
        input_scale, input_zero_point = input_details[0]["quantization"]

        y_pred = []
        for i in range(len(X_test)):
            # 1. Take the float32 sample
            test_sample = np.expand_dims(X_test[i], axis=0).astype(np.float32)

            # 2. Quantize the input: (Float_Value / Scale) + Zero_Point
            if input_scale != 0.0:
                quantized_input = test_sample / input_scale + input_zero_point
                # Cast to int8 and clamp to valid range [-128, 127]
                quantized_input = np.clip(np.round(quantized_input), -128, 127).astype(np.int8)
            else:
                quantized_input = test_sample.astype(np.int8)

            # 3. Run Inference
            interpreter.set_tensor(input_details[0]['index'], quantized_input)
            interpreter.invoke()

            # 4. Get the result
            # We don't strictly need to dequantize the output because argmax(INT8)
            # yields the same index as argmax(Float) for classification probabilities.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data)
            y_pred.append(predicted_class)

        report = classification_report(y_test, y_pred, digits=5)
        print(report)

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Quantized TFLite Model Classification Report\n")
            f.write("============================================\n")
            f.write(report)
        print(f"Saved Quantized TFLite classification report to: {report_path}")

    except Exception as e:
        print(f"Failed to evaluate Quantized TFLite model: {e}")


def print_tflite_ops(tflite_model_content, ops_path):
    """Prints and saves the unique TFLite operations used by the Quantized model."""
    print("\nExtracting Quantized TFLite Operations...")
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        ops_details = interpreter._get_ops_details()

        ops = set(op["op_name"] for op in ops_details)
        ops_list = sorted(list(ops))

        print(f"Total number of unique ops: {len(ops_list)}")
        print("TensorFlow Lite Ops utilized by the model:")
        for op in ops_list:
            print(f" - {op}")

        os.makedirs(os.path.dirname(ops_path), exist_ok=True)
        with open(ops_path, "w") as f:
            f.write(f"Total number of unique ops: {len(ops_list)}\n")
            f.write("TensorFlow Lite Ops utilized by the model:\n")
            for op in ops_list:
                f.write(f" - {op}\n")
        print(f"Saved TFLite ops list to: {ops_path}")

    except Exception as e:
        print(f"Failed to extract TFLite ops: {e}")


def convert_tflite_to_c_header(tflite_path, header_path):
    """Converts the compiled Quantized TFLite model to a C header file using xxd."""
    print("\nConverting Quantized TFLite model to C header file...")
    try:
        subprocess.run(f"xxd -i {tflite_path} > {header_path}", shell=True, check=True)
        print(f"Successfully generated C header file: {header_path}")
    except subprocess.CalledProcessError as e:
        print(f"xxd command failed: {e}. Please ensure 'xxd' is installed.")
    except Exception as e:
        print(f"Failed to convert TFLite model to C header: {e}")


def main():
    # 1. Generate data
    X_train, X_test, y_train_onehot, y_test = generate_data()

    # 2. Build and train Keras model
    keras_model = build_and_train_model(X_train, y_train_onehot)

    # 3. Evaluate Keras model and save report
    evaluate_keras_model(keras_model, X_test, y_test, KERAS_REPORT_PATH)

    # 4. Convert to Fully Integer Quantized TFLite (INT8)
    tflite_quant_model = convert_to_quantized_tflite(keras_model, X_train, TFLITE_QUANT_MODEL_PATH)

    if tflite_quant_model:
        # 5. Evaluate Quantized TFLite model and save report
        evaluate_quantized_tflite_model(tflite_quant_model, X_test, y_test, TFLITE_QUANT_REPORT_PATH)

        # 6. Print utilized TFLite ops and save to file
        print_tflite_ops(tflite_quant_model, TFLITE_QUANT_OPS_PATH)

        # 7. Convert to C header using xxd
        convert_tflite_to_c_header(TFLITE_QUANT_MODEL_PATH, HEADER_FILE_PATH)


if __name__ == "__main__":
    main()
```

### 2.2 Quantized Deployment (PlatformIO)

In the quantized implementation,
the primary difference is the data
type of the input/output tensors (`int8_t` instead of `float`).

```cpp
#include <Arduino.h>
#include <esp32_tflm_wrapper.h>
#include "mlp_model_quant.h"

#define ARENA_SIZE 1800
#define TF_NUM_OPS 2
#define TF_NUM_INPUTS 20
#define TF_NUM_OUTPUTS 3

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> model;

void setup() {
    Serial.begin(115200);
    delay(2000);

    model.setNumInputs(TF_NUM_INPUTS);
    model.setNumOutputs(TF_NUM_OUTPUTS);

    model.resolver.AddFullyConnected();
    model.resolver.AddSoftmax();

    if (!model.begin(mlp_model_quant_tflite).isOk()) {
        Serial.println(model.exception.toString());
        while (true);
    }
}

// Input must be int8_t for fully quantized models
int8_t input_quant[TF_NUM_INPUTS];

void loop() {
    // Fill with quantized dummy data [-128, 127]
    for (int i = 0; i < TF_NUM_INPUTS; i++) {
        input_quant[i] = static_cast<int8_t>(random(-128, 127));
    }

    const auto start = micros();
    if (!model.predict(input_quant).isOk()) {
        Serial.println(model.exception.toString());
        return;
    }
    const auto end = micros();

    Serial.print("Quantized Class: ");
    Serial.print(model.classification);
    Serial.print(" | Latency: ");
    Serial.print(end - start);
    Serial.println(" us");

    delay(1000);
}
```

---

## Notes

### Tensor Arena Management

The `ARENA_SIZE` is a static allocation
of memory for the interpreter's scratchpad.
Researchers should aim for the
smallest possible value that successfully initializes the model.
Under-allocation results in initialization failure,
while over-allocation wastes valuable SRAM.

### Op Resolver Optimization

By utilizing `model.resolver.Add...()` instead
of the `AllOpsResolver`,
we significantly reduce the final binary size.
This is a critical step for deploying complex
models that might otherwise
exceed the ESP32's Flash limits.

### Accuracy vs. Latency Trade-off

Researchers must evaluate the classification
reports (saved during Python execution) to determine if the INT8
quantization degradation is acceptable for their specific application.
In many classification tasks, the index of the
maximum value (the class) remains stable despite the slight
precision loss in the underlying probabilities.

---

## 3. Phase II: Convolutional Neural Networks (CNN)

Convolutional Neural Networks (CNNs) are employed to extract
spatial-temporal features from sequential data,
treating time-series windows as single-channel images.

### 3.1 Static Context Analysis (Python)

We simulate a sequential dataset (e.g., 60 time steps with 3 features) and apply a 2D Convolutional architecture.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

TIMESTEPS = 60
FEATURES = 3
N_CLASSES = 5


def build_cnn_model():
    """
    Constructs a 2D CNN architecture for sequential feature extraction.
    The input is treated as a (Timesteps x Features x 1) tensor.
    """
    model = models.Sequential([
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                      input_shape=(TIMESTEPS, FEATURES, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Generate synthetic sequential data
X_train = np.random.rand(500, TIMESTEPS, FEATURES, 1).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, N_CLASSES, 500), N_CLASSES)

model = build_cnn_model()
model.fit(X_train, y_train, epochs=5, verbose=0)
```

### 3.2 Optimization: INT8 Quantization

Quantization is critical for CNNs as they involve significantly more parameters and intermediate feature maps.

```python
def representative_data_gen():
    for i in range(100):
        yield [X_train[i:i + 1]]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_disable_per_channel_quantization_for_dense_layers = True

tflite_quant_model = converter.convert()
```

### 3.3 CNN Implementation Considerations

CNNs require a substantially larger **Tensor Arena**.
The intermediate feature maps created during convolution
and pooling are stored here.
Based on our research analysis,
this model architecture results in
approximately **2.88 MFLOPs** and
requires the following operation registration:

```cpp
#include "cnn_model_quant.h"

// CNNs require significantly larger arenas than MLPs
#define ARENA_SIZE 32000 // Significantly larger than MLP
#define TF_NUM_OPS 5

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> model;

void setup() {
    model.setNumInputs(TIMESTEPS * FEATURES);
    model.setNumOutputs(N_CLASSES);
    
    // Register specific operations utilized by the CNN model:
    // ['CONV_2D', 'FULLY_CONNECTED', 'MAX_POOL_2D', 'RESHAPE', 'SOFTMAX']
    model.resolver.AddConv2D();
    model.resolver.AddFullyConnected();
    model.resolver.AddMaxPool2D();
    model.resolver.AddReshape();
    model.resolver.AddSoftmax();
    
    model.begin(cnn_model_quant_tflite);
}
```

---

## 4. Phase III: Recurrent Neural Networks (RNN)

Recurrent Neural Networks (RNNs) analyze dynamic context
by maintaining internal state across time steps.

### 4.1 Unrolling the RNN for Microcontrollers

TFLM has limited support for dynamic control flow loops.
To ensure broad compatibility and performance on the ESP32,
we **unroll** the RNN. This instructs the compiler
to expand the recurrent loop into a static graph of operations.
Our analysis shows this architecture results
in approximately **2.05 MFLOPs**.

```python
def build_rnn_model():
    model = tf.keras.models.Sequential([
        # unroll=True is mandatory for stable TFLM deployment
        tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(64),
                            input_shape=(TIMESTEPS, FEATURES),
                            unroll=True),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model
```

### 4.2 Quantization of Non-Linearities

Recurrent architectures frequently utilize `tanh` or `sigmoid` activations.
When performing INT8 quantization, ensure that the TFLM version
you are using includes the integer kernels for these non-linearities.
As identified in our research, unrolled RNNs
often require a broader set of utility operations
(like `STRIDED_SLICE`, `PACK`, and `TRANSPOSE`) to
manage the expanded state and data movement.

### 4.3 RNN Implementation

The RNN implementation requires registering a wider
variety of operations due to the unrolling
and state management identified during analysis.

```cpp
#include "rnn_model_quant.h"

#define ARENA_SIZE 15000
#define TF_NUM_OPS 10 // Based on research analysis

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> model;

void setup() {
    // Register operations utilized by the unrolled RNN:
    // ['ADD', 'FILL', 'FULLY_CONNECTED', 'PACK', 'SHAPE', 'SOFTMAX', 'STRIDED_SLICE', 'TANH', 'TRANSPOSE', 'UNPACK']
    model.resolver.AddAdd();
    model.resolver.AddFill();
    model.resolver.AddFullyConnected();
    model.resolver.AddPack();
    model.resolver.AddShape();
    model.resolver.AddSoftmax();
    model.resolver.AddStridedSlice();
    model.resolver.AddTanh();
    model.resolver.AddTranspose();
    model.resolver.AddUnpack();
    
    model.begin(rnn_model_quant_tflite);
}
```

---
