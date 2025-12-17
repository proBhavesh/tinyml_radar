"""
TinyML Radar Gesture Detection Model Training
Target: ARM Cortex-M7 (ATSAMS70Q21)

This script trains a small CNN for radar-based gesture classification
optimized for microcontroller deployment.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_generator import generate_dataset, NUM_SAMPLES, NUM_CHIRPS, NUM_CLASSES, CLASS_NAMES

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Model configuration
INPUT_SHAPE = (NUM_SAMPLES, NUM_CHIRPS, 1)  # 64 x 32 x 1
MODEL_NAME = 'radar_gesture_model'


def create_model_small() -> keras.Model:
    """
    Small CNN optimized for Cortex-M7
    Target size: < 50KB quantized

    Architecture chosen for:
    - Minimal memory footprint
    - Fast inference (~10ms on Cortex-M7)
    - Good accuracy on gesture classification
    """
    model = keras.Sequential([
        # Input: 64x32x1
        layers.InputLayer(input_shape=INPUT_SHAPE),

        # Conv block 1: 64x32x1 -> 32x16x8
        layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv block 2: 32x16x8 -> 16x8x16
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv block 3: 16x8x16 -> 8x4x32
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten: 8x4x32 = 1024
        layers.Flatten(),

        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name=MODEL_NAME)

    return model


def create_model_tiny() -> keras.Model:
    """
    Even smaller model for minimal footprint
    Target size: < 20KB quantized
    """
    model = keras.Sequential([
        layers.InputLayer(input_shape=INPUT_SHAPE),

        # Single conv block
        layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),

        # Global pooling instead of flatten (much smaller)
        layers.GlobalAveragePooling2D(),

        # Small dense layer
        layers.Dense(16, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name=MODEL_NAME + '_tiny')

    return model


def train_model(model: keras.Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
                epochs: int = 50) -> keras.callbacks.History:
    """Train the model with early stopping"""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]

    print("\nModel Summary:")
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """Evaluate model and print detailed metrics"""

    # Overall accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Per-class predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Confusion matrix
    from collections import Counter
    confusion = {}
    for true_class in range(NUM_CLASSES):
        confusion[CLASS_NAMES[true_class]] = Counter()
        mask = y_test == true_class
        predictions = y_pred[mask]
        for pred in predictions:
            confusion[CLASS_NAMES[true_class]][CLASS_NAMES[pred]] += 1

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    print("\nPer-class Results:")
    for class_name in CLASS_NAMES:
        total = sum(confusion[class_name].values())
        correct = confusion[class_name][class_name]
        acc = correct / total if total > 0 else 0
        print(f"  {class_name:20s}: {acc*100:5.1f}% ({correct}/{total})")

    print("\nConfusion Matrix:")
    print(f"{'Predicted ->':>15}", end='')
    for name in CLASS_NAMES:
        print(f"{name[:8]:>10}", end='')
    print()
    for true_name in CLASS_NAMES:
        print(f"{true_name[:15]:>15}", end='')
        for pred_name in CLASS_NAMES:
            count = confusion[true_name][pred_name]
            print(f"{count:>10}", end='')
        print()

    return {
        'loss': loss,
        'accuracy': accuracy,
        'confusion': confusion
    }


def estimate_model_size(model: keras.Model) -> dict:
    """Estimate model size for deployment"""

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

    # Estimate sizes
    float32_size = total_params * 4  # 4 bytes per float32
    int8_size = total_params * 1     # 1 byte per int8 (quantized)

    # Add overhead for TFLite format (~10-20%)
    tflite_float_estimate = int(float32_size * 1.15)
    tflite_int8_estimate = int(int8_size * 1.2)

    print("\n" + "=" * 50)
    print("MODEL SIZE ESTIMATION")
    print("=" * 50)
    print(f"Total Parameters:    {total_params:,}")
    print(f"Float32 Size:        {float32_size:,} bytes ({float32_size/1024:.1f} KB)")
    print(f"Int8 Quantized:      {int8_size:,} bytes ({int8_size/1024:.1f} KB)")
    print(f"TFLite Float Est:    {tflite_float_estimate:,} bytes ({tflite_float_estimate/1024:.1f} KB)")
    print(f"TFLite Int8 Est:     {tflite_int8_estimate:,} bytes ({tflite_int8_estimate/1024:.1f} KB)")

    return {
        'total_params': total_params,
        'float32_bytes': float32_size,
        'int8_bytes': int8_size,
        'tflite_float_estimate': tflite_float_estimate,
        'tflite_int8_estimate': tflite_int8_estimate
    }


def save_model(model: keras.Model, path: str = 'models'):
    """Save model in multiple formats"""
    os.makedirs(path, exist_ok=True)

    # Save Keras model
    keras_path = os.path.join(path, f'{model.name}.keras')
    model.save(keras_path)
    print(f"\nKeras model saved to: {keras_path}")

    # Save weights only (Keras 3 requires .weights.h5 extension)
    weights_path = os.path.join(path, f'{model.name}.weights.h5')
    model.save_weights(weights_path)
    print(f"Weights saved to: {weights_path}")


def main():
    print("=" * 60)
    print("TINYML RADAR GESTURE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"Input Shape: {INPUT_SHAPE}")
    print(f"Classes: {CLASS_NAMES}")

    # Generate dataset
    print("\n[1/4] Loading radar sensor data...")
    X_train, X_test, y_train, y_test = generate_dataset(samples_per_class=1000)

    # Create model
    print("\n[2/4] Creating model...")
    model = create_model_small()

    # Estimate size before training
    size_info = estimate_model_size(model)

    # Train
    print("\n[3/4] Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=50)

    # Evaluate
    print("\n[4/4] Evaluating model...")
    eval_results = evaluate_model(model, X_test, y_test)

    # Save
    save_model(model)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"Estimated Int8 Size: {size_info['tflite_int8_estimate']/1024:.1f} KB")
    print("\nNext step: Run export_model.py to quantize and export for firmware")

    return model, history, eval_results


if __name__ == '__main__':
    main()
