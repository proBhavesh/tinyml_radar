"""
TinyML Model Export and Quantization
Converts trained Keras model to TensorFlow Lite (int8) and C header

Target: ARM Cortex-M7 with TensorFlow Lite Micro
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_generator import generate_dataset, NUM_SAMPLES, NUM_CHIRPS

MODEL_PATH = 'models/radar_gesture_model.keras'
OUTPUT_DIR = 'models'
FIRMWARE_DIR = 'firmware'


def representative_dataset_gen():
    """
    Generator for calibration data during quantization
    Provides representative input samples for int8 calibration
    """
    # Generate calibration data
    X_train, _, _, _ = generate_dataset(samples_per_class=100)

    for i in range(min(200, len(X_train))):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]


def convert_to_tflite_float(model: keras.Model, output_path: str) -> int:
    """Convert model to TFLite (float32)"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size = len(tflite_model)
    print(f"Float32 TFLite model: {size:,} bytes ({size/1024:.1f} KB)")
    return size


def convert_to_tflite_int8(model: keras.Model, output_path: str) -> int:
    """
    Convert model to fully quantized TFLite (int8)
    This is the target format for Cortex-M7 deployment
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size = len(tflite_model)
    print(f"Int8 TFLite model: {size:,} bytes ({size/1024:.1f} KB)")
    return size


def convert_to_c_array(tflite_path: str, output_path: str, array_name: str = 'g_model'):
    """
    Convert TFLite model to C header file for embedding in firmware
    """
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    # Generate C header
    c_code = f"""/*
 * TinyML Radar Gesture Detection Model
 * Auto-generated from {os.path.basename(tflite_path)}
 *
 * Model size: {len(model_data):,} bytes ({len(model_data)/1024:.1f} KB)
 * Input:  64x32x1 int8 (radar frame)
 * Output: 4 int8 values (class probabilities)
 *
 * Classes:
 *   0 = no_presence
 *   1 = static_presence
 *   2 = wave_gesture
 *   3 = approach
 */

#ifndef RADAR_MODEL_DATA_H
#define RADAR_MODEL_DATA_H

#include <stdint.h>

#define RADAR_MODEL_SIZE {len(model_data)}

/* Model input/output dimensions */
#define MODEL_INPUT_SAMPLES  64
#define MODEL_INPUT_CHIRPS   32
#define MODEL_INPUT_SIZE     (MODEL_INPUT_SAMPLES * MODEL_INPUT_CHIRPS)
#define MODEL_OUTPUT_CLASSES 4

/* Class indices */
#define CLASS_NO_PRESENCE     0
#define CLASS_STATIC_PRESENCE 1
#define CLASS_WAVE_GESTURE    2
#define CLASS_APPROACH        3

/* Quantization parameters (from TFLite model) */
#define INPUT_SCALE    0.0078125f   /* 1/128 */
#define INPUT_ZERO_POINT  0
#define OUTPUT_SCALE   0.00390625f  /* 1/256 */
#define OUTPUT_ZERO_POINT -128

alignas(16) const uint8_t {array_name}[{len(model_data)}] = {{
"""

    # Add hex data in rows of 12 bytes
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        c_code += f"    {hex_values},\n"

    c_code += f"""}};\n
const unsigned int {array_name}_len = {len(model_data)};

#endif /* RADAR_MODEL_DATA_H */
"""

    with open(output_path, 'w') as f:
        f.write(c_code)

    print(f"C header generated: {output_path}")


def verify_tflite_model(tflite_path: str):
    """Verify the quantized model works correctly"""
    print("\nVerifying TFLite model...")

    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Get quantization parameters
    if 'quantization' in input_details[0]:
        scale, zero_point = input_details[0]['quantization']
        if scale != 0:
            print(f"Input quantization: scale={scale}, zero_point={zero_point}")

    # Test with random input
    X_train, X_test, y_train, y_test = generate_dataset(samples_per_class=50)

    correct = 0
    total = len(X_test)

    for i in range(total):
        # Prepare input
        if input_details[0]['dtype'] == np.int8:
            # Quantize input
            input_data = (X_test[i:i+1] / 0.0078125).astype(np.int8)
        else:
            input_data = X_test[i:i+1].astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output)

        if predicted == y_test[i]:
            correct += 1

    accuracy = correct / total
    print(f"TFLite Verification Accuracy: {accuracy*100:.1f}% ({correct}/{total})")

    return accuracy


def analyze_model_for_mcu(tflite_path: str):
    """Analyze model requirements for MCU deployment"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get tensor details
    tensor_details = interpreter.get_tensor_details()

    # Calculate memory requirements
    arena_size = 0
    max_tensor_size = 0

    for tensor in tensor_details:
        size = np.prod(tensor['shape']) * np.dtype(tensor['dtype']).itemsize
        arena_size += size
        max_tensor_size = max(max_tensor_size, size)

    # Read model file size
    with open(tflite_path, 'rb') as f:
        model_size = len(f.read())

    print("\n" + "=" * 60)
    print("MCU DEPLOYMENT ANALYSIS")
    print("=" * 60)
    print(f"Model Size (Flash):     {model_size:,} bytes ({model_size/1024:.1f} KB)")
    print(f"Tensor Arena (RAM):     {arena_size:,} bytes ({arena_size/1024:.1f} KB)")
    print(f"Largest Tensor:         {max_tensor_size:,} bytes")
    print(f"Number of Tensors:      {len(tensor_details)}")

    print("\n--- ATSAMS70Q21 Resource Usage ---")
    flash_total = 2 * 1024 * 1024  # 2MB
    ram_total = 384 * 1024          # 384KB

    flash_used = model_size
    # Estimate: arena + input buffer + output buffer + stack
    ram_used = arena_size + (64 * 32) + 16 + 8192

    print(f"Flash: {flash_used/1024:.1f} KB / {flash_total/1024:.0f} KB ({flash_used/flash_total*100:.2f}%)")
    print(f"RAM:   {ram_used/1024:.1f} KB / {ram_total/1024:.0f} KB ({ram_used/ram_total*100:.2f}%)")
    print(f"\nHeadroom:")
    print(f"  Flash available: {(flash_total-flash_used)/1024:.0f} KB")
    print(f"  RAM available:   {(ram_total-ram_used)/1024:.0f} KB")

    return {
        'model_size': model_size,
        'arena_size': arena_size,
        'flash_percent': flash_used / flash_total * 100,
        'ram_percent': ram_used / ram_total * 100
    }


def main():
    print("=" * 60)
    print("TINYML MODEL EXPORT AND QUANTIZATION")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIRMWARE_DIR, exist_ok=True)

    # Load trained model
    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Run train_model.py first.")
        print("Training model now...")
        from train_model import main as train_main
        model, _, _ = train_main()
    else:
        model = keras.models.load_model(MODEL_PATH)

    print(f"Model loaded: {model.name}")
    model.summary()

    # Convert to TFLite float32
    print("\n[1/4] Converting to TFLite (float32)...")
    float_path = os.path.join(OUTPUT_DIR, 'radar_model_float.tflite')
    convert_to_tflite_float(model, float_path)

    # Convert to TFLite int8
    print("\n[2/4] Converting to TFLite (int8 quantized)...")
    int8_path = os.path.join(OUTPUT_DIR, 'radar_model_int8.tflite')
    convert_to_tflite_int8(model, int8_path)

    # Generate C header
    print("\n[3/4] Generating C header for firmware...")
    c_header_path = os.path.join(FIRMWARE_DIR, 'radar_model_data.h')
    convert_to_c_array(int8_path, c_header_path)

    # Verify and analyze
    print("\n[4/4] Verifying and analyzing model...")
    verify_tflite_model(int8_path)
    mcu_analysis = analyze_model_for_mcu(int8_path)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"TFLite float32: {float_path}")
    print(f"TFLite int8:    {int8_path}")
    print(f"C header:       {c_header_path}")
    print(f"\nModel ready for Cortex-M7 deployment!")

    return mcu_analysis


if __name__ == '__main__':
    main()
