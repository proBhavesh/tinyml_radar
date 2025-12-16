/*
 * TensorFlow Lite Micro Inference for Radar Gesture Detection
 * Target: ARM Cortex-M7 (ATSAMS70Q21)
 *
 * This module provides ML inference for classifying radar frames
 * into gesture categories using a quantized neural network.
 */

#ifndef TFLITE_INFERENCE_H
#define TFLITE_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>

/* Model input dimensions */
#define ML_INPUT_SAMPLES    64
#define ML_INPUT_CHIRPS     32
#define ML_INPUT_SIZE       (ML_INPUT_SAMPLES * ML_INPUT_CHIRPS)

/* Model output classes */
#define ML_NUM_CLASSES      4

/* Class definitions */
typedef enum {
    ML_CLASS_NO_PRESENCE     = 0,
    ML_CLASS_STATIC_PRESENCE = 1,
    ML_CLASS_WAVE_GESTURE    = 2,
    ML_CLASS_APPROACH        = 3
} ml_class_t;

/* Inference result structure */
typedef struct {
    ml_class_t predicted_class;     /* Most likely class */
    float confidence;               /* Confidence score (0.0 - 1.0) */
    float class_scores[ML_NUM_CLASSES];  /* Probability for each class */
    uint32_t inference_time_us;     /* Inference time in microseconds */
    bool valid;                     /* True if inference succeeded */
} ml_result_t;

/* Error codes */
typedef enum {
    ML_OK = 0,
    ML_ERROR_NOT_INITIALIZED,
    ML_ERROR_ALLOCATION_FAILED,
    ML_ERROR_INVOKE_FAILED,
    ML_ERROR_INVALID_INPUT
} ml_error_t;

/*
 * Initialize the TFLite Micro interpreter
 * Must be called once before any inference
 *
 * Returns: ML_OK on success, error code otherwise
 */
ml_error_t ml_init(void);

/*
 * Run inference on a radar frame
 *
 * Parameters:
 *   input_frame: Pointer to radar frame data (64x32 int16 samples)
 *   result: Output structure to store inference results
 *
 * Returns: ML_OK on success, error code otherwise
 */
ml_error_t ml_inference(const int16_t *input_frame, ml_result_t *result);

/*
 * Run inference with normalized float input
 *
 * Parameters:
 *   input_float: Pointer to normalized radar frame (-1.0 to 1.0)
 *   result: Output structure to store inference results
 *
 * Returns: ML_OK on success, error code otherwise
 */
ml_error_t ml_inference_float(const float *input_float, ml_result_t *result);

/*
 * Get class name string
 *
 * Parameters:
 *   class_id: Class index (0-3)
 *
 * Returns: Pointer to class name string
 */
const char* ml_get_class_name(ml_class_t class_id);

/*
 * Get model information
 *
 * Returns model size in bytes
 */
uint32_t ml_get_model_size(void);

/*
 * Get tensor arena size (RAM usage)
 *
 * Returns arena size in bytes
 */
uint32_t ml_get_arena_size(void);

/*
 * Check if ML module is initialized
 */
bool ml_is_initialized(void);

#endif /* TFLITE_INFERENCE_H */
