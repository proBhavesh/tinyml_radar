/*
 * ML Integration Example
 * Shows how to integrate TinyML inference with the BJT60 radar firmware
 *
 * This file demonstrates the integration pattern - copy relevant parts
 * to src/main.c in the bjt60_firmware project.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* Include radar driver and ML headers */
#include "avian_radar.h"
#include "tflite_inference.h"

/* Optional: Keep presence detection as fallback */
#include "presence_detection.h"

/*
 * Initialize ML system
 * Call this during system startup after radar_init()
 */
bool ml_system_init(void)
{
    ml_error_t err = ml_init();

    if (err != ML_OK) {
        /* ML init failed - can fall back to presence detection */
        return false;
    }

    /* Print ML info (if UART available) */
    printf("ML Model initialized:\n");
    printf("  Model size:  %lu bytes\n", ml_get_model_size());
    printf("  Arena size:  %lu bytes\n", ml_get_arena_size());

    return true;
}

/*
 * Process radar frame with ML inference
 * Returns the detected gesture class
 */
ml_class_t process_frame_ml(const radar_frame_t *frame)
{
    ml_result_t result;

    /* Run ML inference on radar samples */
    ml_error_t err = ml_inference(frame->samples, &result);

    if (err != ML_OK || !result.valid) {
        return ML_CLASS_NO_PRESENCE;  /* Default on error */
    }

    /* Optional: Print results (remove in production) */
    printf("ML Result: %s (%.1f%%) in %lu us\n",
           ml_get_class_name(result.predicted_class),
           result.confidence * 100.0f,
           result.inference_time_us);

    return result.predicted_class;
}

/*
 * Update LED based on ML classification
 */
void update_led_from_ml(ml_class_t gesture)
{
    switch (gesture) {
        case ML_CLASS_NO_PRESENCE:
            /* All LEDs off */
            led_red_off();
            led_green_off();
            led_blue_off();
            break;

        case ML_CLASS_STATIC_PRESENCE:
            /* Green LED - person detected */
            led_red_off();
            led_green_on();
            led_blue_off();
            break;

        case ML_CLASS_WAVE_GESTURE:
            /* Blue LED - wave detected */
            led_red_off();
            led_green_off();
            led_blue_on();
            break;

        case ML_CLASS_APPROACH:
            /* Red LED - someone approaching */
            led_red_on();
            led_green_off();
            led_blue_off();
            break;
    }
}

/*
 * Example main loop with ML integration
 *
 * This replaces the main loop in bjt60_firmware/src/main.c
 */
void ml_main_loop(void)
{
    ml_class_t current_gesture = ML_CLASS_NO_PRESENCE;
    ml_class_t last_gesture = ML_CLASS_NO_PRESENCE;
    uint32_t gesture_count = 0;

    /* Start radar acquisition */
    radar_start();

    while (1) {
        /* Check if frame is ready */
        if (radar_frame_ready()) {
            /* Get frame data */
            const radar_frame_t *frame = radar_get_frame();

            if (frame && frame->valid) {
                /* Run ML inference */
                current_gesture = process_frame_ml(frame);

                /* Simple debouncing: require 3 consecutive same predictions */
                if (current_gesture == last_gesture) {
                    gesture_count++;
                    if (gesture_count >= 3) {
                        update_led_from_ml(current_gesture);
                    }
                } else {
                    gesture_count = 0;
                }

                last_gesture = current_gesture;
            }

            /* Restart frame acquisition */
            radar_start_frame();
        }

        /* Optional: Watchdog kick, sleep, etc. */
    }
}


/*
 * Hybrid mode: Use both ML and presence detection
 *
 * This approach uses:
 * - Presence detection for quick presence/absence check
 * - ML only when presence is detected (saves power)
 */
void hybrid_main_loop(void)
{
    presence_ctx_t presence_ctx;
    presence_init(&presence_ctx);

    radar_start();

    while (1) {
        if (radar_frame_ready()) {
            const radar_frame_t *frame = radar_get_frame();

            if (frame && frame->valid) {
                /* First: Quick presence check */
                bool is_present = presence_detect(&presence_ctx, frame);

                if (!is_present) {
                    /* No presence - all LEDs off, skip ML */
                    led_red_off();
                    led_green_off();
                    led_blue_off();
                } else {
                    /* Presence detected - run ML for gesture classification */
                    ml_class_t gesture = process_frame_ml(frame);
                    update_led_from_ml(gesture);
                }
            }

            radar_start_frame();
        }
    }
}


/*
 * Modified main() function
 * Copy this structure to bjt60_firmware/src/main.c
 */
#if 0  /* Disabled - for reference only */
int main(void)
{
    /* Disable watchdog */
    disable_watchdog();

    /* Initialize hardware */
    clock_init();
    gpio_init();
    spi_init();

    /* Initialize radar */
    if (!radar_init()) {
        /* Radar init failed - blink error pattern */
        while (1) {
            led_red_on();
            delay_ms(100);
            led_red_off();
            delay_ms(100);
        }
    }

    /* Initialize ML */
    if (!ml_system_init()) {
        /* ML init failed - fall back to presence detection only */
        presence_ctx_t ctx;
        presence_init(&ctx);
        radar_start();

        while (1) {
            if (radar_frame_ready()) {
                const radar_frame_t *frame = radar_get_frame();
                if (frame && frame->valid) {
                    if (presence_detect(&ctx, frame)) {
                        led_green_on();
                    } else {
                        led_green_off();
                    }
                }
                radar_start_frame();
            }
        }
    }

    /* ML initialized successfully - run ML main loop */
    ml_main_loop();

    return 0;
}
#endif
