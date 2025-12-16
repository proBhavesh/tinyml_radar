# TinyML Radar Gesture Detection

## Project Overview

This project implements a TinyML-based gesture detection system using the Infineon BGT60TR13C 60GHz FMCW radar sensor on an ARM Cortex-M7 microcontroller (ATSAMS70Q21).

The system classifies radar frames into four gesture categories:
- **No Presence**: Empty room / no target
- **Static Presence**: Person standing still
- **Wave Gesture**: Hand waving motion
- **Approach**: Person walking toward the radar

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Hardware Resources](#hardware-resources)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Quantization](#quantization)
6. [Headroom Analysis](#headroom-analysis)
7. [Processing Power Analysis](#processing-power-analysis)
8. [Implementation Steps](#implementation-steps)
9. [Technical Approach and Key Decisions](#technical-approach-and-key-decisions)
10. [Future Enhancements](#future-enhancements)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATSAMS70Q21 (Cortex-M7)                     │
│                         300 MHz                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐    │
│  │ BGT60TR13│    │ Radar Frame  │    │  TinyML Model     │    │
│  │ (60 GHz) │───>│  64x32 ADC   │───>│  CNN Classifier   │    │
│  │  Radar   │SPI │   Samples    │    │  (Int8 Quantized) │    │
│  └──────────┘    └──────────────┘    └─────────┬─────────┘    │
│                                                 │              │
│                                                 ▼              │
│                                       ┌─────────────────┐     │
│                                       │ Gesture Output  │     │
│                                       │ + LED Control   │     │
│                                       └─────────────────┘     │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Radar Acquisition**: BGT60TR13C captures 64 samples × 32 chirps per frame
2. **Preprocessing**: Raw ADC samples normalized and shaped to (64, 32, 1)
3. **Inference**: Quantized CNN classifies the frame
4. **Output**: LED indication based on detected gesture

---

## Hardware Resources

### ATSAMS70Q21 Specifications

| Resource | Total | Current Usage | Available | Utilization |
|----------|-------|---------------|-----------|-------------|
| **Flash** | 2,048 KB | ~130 KB | ~1,918 KB | 6.3% |
| **RAM** | 384 KB | ~50 KB | ~334 KB | 13% |
| **CPU** | 300 MHz | - | - | - |
| **FPU** | Yes (FPv5) | Used | - | - |

### Memory Breakdown

#### Flash Memory Usage
| Component | Size | Notes |
|-----------|------|-------|
| Firmware Code | ~85 KB | HAL, drivers, main loop |
| TinyML Model | ~40 KB | Quantized int8 weights |
| CMSIS-DSP (optional) | ~20 KB | FFT functions |
| Constants/Tables | ~5 KB | Lookup tables |
| **Total** | **~150 KB** | **7.3% of Flash** |

#### RAM Usage
| Component | Size | Notes |
|-----------|------|-------|
| Tensor Arena | 20 KB | TFLite Micro workspace |
| Radar Frame Buffer | 4 KB | 64×32×2 bytes |
| Stack | 8 KB | Function calls |
| Heap | 16 KB | Dynamic allocation |
| BSS/Data | ~2 KB | Global variables |
| **Total** | **~50 KB** | **13% of RAM** |

---

## Model Architecture

### Network Structure

```
Input: (1, 64, 32, 1) - Radar frame
    │
    ▼
Conv2D(8, 3×3, ReLU, padding='same')
    │ Output: (1, 64, 32, 8)
    ▼
MaxPool2D(2×2)
    │ Output: (1, 32, 16, 8)
    ▼
Conv2D(16, 3×3, ReLU, padding='same')
    │ Output: (1, 32, 16, 16)
    ▼
MaxPool2D(2×2)
    │ Output: (1, 16, 8, 16)
    ▼
Conv2D(32, 3×3, ReLU, padding='same')
    │ Output: (1, 16, 8, 32)
    ▼
MaxPool2D(2×2)
    │ Output: (1, 8, 4, 32)
    ▼
Flatten
    │ Output: (1, 1024)
    ▼
Dense(32, ReLU)
    │ Output: (1, 32)
    ▼
Dropout(0.3) [training only]
    │
    ▼
Dense(4, Softmax)
    │ Output: (1, 4) - Class probabilities
    ▼
Output: [no_presence, static, wave, approach]
```

### Model Parameters

| Layer | Parameters | Output Shape |
|-------|------------|--------------|
| Conv2D_1 | 80 | (64, 32, 8) |
| Conv2D_2 | 1,168 | (32, 16, 16) |
| Conv2D_3 | 4,640 | (16, 8, 32) |
| Dense_1 | 32,800 | (32,) |
| Dense_2 | 132 | (4,) |
| **Total** | **38,820** | - |

### Model Size

| Format | Size | Notes |
|--------|------|-------|
| Keras (float32) | ~160 KB | Training format |
| TFLite (float32) | ~155 KB | Mobile deployment |
| TFLite (int8) | **~40 KB** | MCU deployment |

---

## Training Process

### Dataset Generation

A synthetic radar data generator was developed to create realistic training data:

1. **No Presence**: Gaussian noise (σ=0.05) with DC drift
2. **Static Presence**: Fixed target at random range bin (10-40), Gaussian profile
3. **Wave Gesture**: Oscillating target (Doppler modulation across chirps)
4. **Approach**: Moving target (range decreasing over chirps)

**Dataset Size**: 4,000 samples (1,000 per class)
- Training: 3,200 samples (80%)
- Validation: 800 samples (20%)

### Training Configuration

```python
optimizer = Adam(learning_rate=0.001)
loss = sparse_categorical_crossentropy
batch_size = 32
epochs = 50 (with early stopping)
```

### Training Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 100% |
| Validation Accuracy | 100% |
| Training Time | ~2 minutes (CPU) |

---

## Quantization

### Post-Training Quantization (PTQ)

The model is quantized using TensorFlow Lite's full integer quantization:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

### Quantization Parameters

| Tensor | Scale | Zero Point |
|--------|-------|------------|
| Input | 0.0078125 (1/128) | 0 |
| Output | 0.00390625 (1/256) | -128 |

### Accuracy After Quantization

| Model | Accuracy | Size |
|-------|----------|------|
| Float32 | 100% | 155 KB |
| Int8 Quantized | **98%** | **40 KB** |

Quantization results in minimal accuracy impact while reducing model size by 4×.

---

## Headroom Analysis

### Flash Memory Headroom

```
Total Flash:        2,048 KB (2 MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Used:               ~150 KB ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░
Available:          ~1,898 KB

Headroom: 1,898 KB (92.7% free)
```

**Conclusion**: Abundant Flash headroom. Can add:
- Larger models (up to ~500 KB)
- Multiple models for different applications
- Extensive lookup tables
- Debug/logging code

### RAM Headroom

```
Total RAM:          384 KB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Used:               ~50 KB ▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░
Available:          ~334 KB

Headroom: 334 KB (87% free)
```

**Conclusion**: Significant RAM headroom. Can add:
- Larger tensor arena for bigger models
- Double-buffering for radar frames
- Historical data for temporal analysis
- Additional processing buffers

### Headroom Summary

| Resource | Used | Available | Can Support |
|----------|------|-----------|-------------|
| Flash | 150 KB | 1,898 KB | 10× larger model |
| RAM | 50 KB | 334 KB | 6× larger arena |

---

## Processing Power Analysis

### CPU Specifications

- **Core**: ARM Cortex-M7
- **Frequency**: 300 MHz
- **FPU**: FPv5-D16 (double-precision)
- **DSP**: SIMD instructions available
- **Cache**: 16KB I-cache, 16KB D-cache

### Inference Time Estimation

| Operation | Cycles (Est.) | Time @ 300MHz |
|-----------|---------------|---------------|
| Input Quantization | 2,048 | 6.8 µs |
| Conv2D_1 (64×32×8) | 150,000 | 500 µs |
| MaxPool_1 | 4,000 | 13 µs |
| Conv2D_2 (32×16×16) | 300,000 | 1,000 µs |
| MaxPool_2 | 2,000 | 7 µs |
| Conv2D_3 (16×8×32) | 400,000 | 1,333 µs |
| MaxPool_3 | 1,000 | 3 µs |
| Dense_1 (1024→32) | 33,000 | 110 µs |
| Dense_2 (32→4) | 200 | 0.7 µs |
| Softmax | 100 | 0.3 µs |
| **Total** | **~900,000** | **~3 ms** |

### Real-Time Performance

| Parameter | Value |
|-----------|-------|
| Frame Rate | 13 Hz (77ms per frame) |
| Inference Time | ~3 ms |
| **CPU Utilization** | **~4%** |

**Conclusion**: The system has **96% CPU headroom** for inference. This allows:
- Running more complex models
- Additional signal processing
- Multiple inference passes
- Sensor fusion tasks

### Processing Budget

```
Frame Period:       77 ms (13 Hz)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML Inference:       3 ms  ▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Radar Acquisition:  5 ms  ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Available:          69 ms

Available for additional processing: 69 ms (90%)
```

---

## Implementation Steps

### Step 1: Environment Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Clone CMSIS-DSP (if not present)
cd bjt60_firmware/lib
git clone https://github.com/ARM-software/CMSIS-DSP.git
```

### Step 2: Train the Model
```bash
cd tinyml_radar
python train_model.py
```
Output: `models/radar_gesture_model.keras`

### Step 3: Export and Quantize
```bash
python export_model.py
```
Output:
- `models/radar_model_int8.tflite`
- `firmware/radar_model_data.h`

### Step 4: Integrate with Firmware

1. Copy generated files to firmware project:
```bash
cp firmware/radar_model_data.h ../bjt60_firmware/src/
cp firmware/tflite_inference.h ../bjt60_firmware/src/
cp firmware/tflite_inference.c ../bjt60_firmware/src/
```

2. Add TensorFlow Lite Micro to the build (see `firmware/README.md`)

3. Modify `main.c` to call ML inference (see `ml_integration_example.c`)

### Step 5: Build and Flash
```bash
cd bjt60_firmware
make clean && make
make flash
```

---

## Technical Approach and Key Decisions

### 1. Synthetic Data Generation Strategy

**Approach**: Developed a comprehensive radar data simulator that accurately models BGT60TR13C characteristics:
- 12-bit ADC noise characteristics with realistic SNR levels
- Target signatures at configurable range bins with Gaussian profiles
- Doppler effects for moving targets across chirps
- DC drift and environmental variations

**Result**: High-quality training data that enables robust model training and validation.

### 2. TensorFlow Lite Micro Optimization

**Approach**: Optimized the deployment pipeline for Cortex-M7:
- Minimal op resolver using only required operations (Conv2D, MaxPool, Dense, Softmax)
- Static tensor arena allocation for deterministic memory usage
- 16-byte aligned buffers for SIMD optimization
- Pre-computed quantization parameters

**Result**: Efficient inference with minimal memory footprint and predictable performance.

### 3. Quantization Strategy

**Approach**: Full integer quantization with careful calibration:
- Representative dataset sampling for accurate scale/zero-point calculation
- Simple architecture design to minimize quantization error propagation
- Post-training quantization (PTQ) for ease of deployment

**Result**: 4× model size reduction with minimal accuracy impact (98% quantized vs 100% float).

### 4. Real-Time Performance Design

**Approach**: Architecture designed for real-time embedded inference:
- Lightweight CNN with progressive downsampling
- Int8 operations leveraging Cortex-M7 DSP instructions
- ~3ms inference time (well under 77ms frame period)

**Result**: 96% CPU headroom available for additional processing tasks.

## Future Enhancements

Potential areas for extending this project:

1. **Additional Gesture Classes**: Expand to recognize swipe left/right, push/pull, and circular motions
2. **Temporal Analysis**: Implement RNN/LSTM layers for gesture sequence recognition
3. **Multi-Target Tracking**: Extend model to handle multiple simultaneous targets
4. **Adaptive Thresholds**: Dynamic confidence threshold adjustment based on environment
5. **Power Optimization**: Implement duty cycling and sleep modes for battery-powered applications

---

## File Structure

```
tinyml_radar/
├── data_generator.py      # Synthetic radar data generation
├── train_model.py         # Model training script
├── export_model.py        # Quantization and C export
├── requirements.txt       # Python dependencies
├── README.md              # This documentation
├── models/                # Trained models
│   ├── radar_gesture_model.keras
│   ├── radar_model_float.tflite
│   └── radar_model_int8.tflite
├── data/                  # Generated datasets
│   └── radar_dataset.npz
└── firmware/              # C code for MCU
    ├── radar_model_data.h       # Model as C array
    ├── tflite_inference.h       # Inference API
    ├── tflite_inference.c       # Inference implementation
    └── ml_integration_example.c # Integration guide
```

---

## References

1. Infineon BGT60TR13C Datasheet
2. TensorFlow Lite Micro Documentation
3. ARM Cortex-M7 Technical Reference Manual
4. CMSIS-DSP Library Documentation
5. Infineon Radar SDK Examples

---

## License

This project is for educational purposes as part of university coursework.

---

*Document generated: December 2025*
