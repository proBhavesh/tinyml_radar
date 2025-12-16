#!/usr/bin/env python3
"""
TinyML Radar Model Pipeline
Run this script to train, quantize, and export the model in one step.

Usage:
    python run_pipeline.py
"""

import os
import sys

def main():
    print("=" * 70)
    print("TINYML RADAR GESTURE DETECTION - FULL PIPELINE")
    print("=" * 70)

    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('firmware', exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1: TRAINING MODEL")
    print("=" * 70)

    from train_model import main as train_main
    model, history, eval_results = train_main()

    print("\n" + "=" * 70)
    print("STEP 2: EXPORTING AND QUANTIZING")
    print("=" * 70)

    from export_model import main as export_main
    mcu_analysis = export_main()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nGenerated Files:")
    print("  models/radar_gesture_model.keras  - Keras model")
    print("  models/radar_model_float.tflite   - TFLite float32")
    print("  models/radar_model_int8.tflite    - TFLite int8 (for MCU)")
    print("  firmware/radar_model_data.h       - C header for firmware")

    print("\nModel Performance:")
    print(f"  Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"  Model Size: {mcu_analysis['model_size']/1024:.1f} KB")
    print(f"  Flash Usage: {mcu_analysis['flash_percent']:.2f}%")
    print(f"  RAM Usage: {mcu_analysis['ram_percent']:.2f}%")

    print("\nNext Steps:")
    print("  1. Copy firmware/*.h and firmware/*.c to bjt60_firmware/src/")
    print("  2. Add TensorFlow Lite Micro to the build")
    print("  3. Integrate ML inference into main.c")
    print("  4. Build and flash firmware")

    return 0


if __name__ == '__main__':
    sys.exit(main())
