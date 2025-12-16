"""
Synthetic Radar Data Generator for TinyML Training
Generates simulated BGT60TR13C radar frames for gesture classification

Target: 64 samples x 32 chirps (matching presence detection config)
Classes: 0=no_presence, 1=static_presence, 2=wave_gesture, 3=approach
"""

import numpy as np
from typing import Tuple, List

# Radar configuration (matching bjt60_firmware)
NUM_SAMPLES = 64      # Samples per chirp (range bins)
NUM_CHIRPS = 32       # Chirps per frame
FRAME_SHAPE = (NUM_SAMPLES, NUM_CHIRPS)

# Class labels
CLASS_NAMES = ['no_presence', 'static_presence', 'wave_gesture', 'approach']
NUM_CLASSES = len(CLASS_NAMES)


def add_noise(frame: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add Gaussian noise to frame"""
    signal_power = np.mean(frame ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), frame.shape)
    return frame + noise


def generate_no_presence(num_frames: int = 100) -> np.ndarray:
    """Generate frames with only background noise"""
    frames = []
    for _ in range(num_frames):
        # Pure noise with some DC offset variation
        frame = np.random.normal(0, 0.05, FRAME_SHAPE)
        frame += np.random.uniform(-0.02, 0.02)  # DC drift
        frames.append(frame)
    return np.array(frames)


def generate_static_presence(num_frames: int = 100) -> np.ndarray:
    """Generate frames with static target (person standing still)"""
    frames = []
    for _ in range(num_frames):
        frame = np.random.normal(0, 0.03, FRAME_SHAPE)

        # Static target at random range bin (10-40)
        target_bin = np.random.randint(10, 40)
        target_width = np.random.randint(2, 5)
        target_amplitude = np.random.uniform(0.3, 0.6)

        # Target appears in all chirps at same range
        for chirp in range(NUM_CHIRPS):
            for b in range(max(0, target_bin - target_width),
                          min(NUM_SAMPLES, target_bin + target_width)):
                distance = abs(b - target_bin)
                frame[b, chirp] += target_amplitude * np.exp(-distance * 0.5)

        frame = add_noise(frame, snr_db=15)
        frames.append(frame)
    return np.array(frames)


def generate_wave_gesture(num_frames: int = 100) -> np.ndarray:
    """Generate frames with waving motion (oscillating Doppler)"""
    frames = []
    for _ in range(num_frames):
        frame = np.random.normal(0, 0.03, FRAME_SHAPE)

        # Target at fixed range
        target_bin = np.random.randint(8, 25)
        target_amplitude = np.random.uniform(0.4, 0.7)

        # Wave frequency (oscillation across chirps)
        wave_freq = np.random.uniform(2, 5)  # Oscillations per frame
        phase = np.random.uniform(0, 2 * np.pi)

        for chirp in range(NUM_CHIRPS):
            # Doppler shift causes phase variation across chirps
            doppler_phase = np.sin(2 * np.pi * wave_freq * chirp / NUM_CHIRPS + phase)

            # Target with Doppler modulation
            bin_offset = int(doppler_phase * 2)  # Small range migration
            actual_bin = target_bin + bin_offset

            if 0 <= actual_bin < NUM_SAMPLES:
                frame[actual_bin, chirp] += target_amplitude * (1 + 0.3 * doppler_phase)
                # Spread to adjacent bins
                if actual_bin > 0:
                    frame[actual_bin - 1, chirp] += target_amplitude * 0.3
                if actual_bin < NUM_SAMPLES - 1:
                    frame[actual_bin + 1, chirp] += target_amplitude * 0.3

        frame = add_noise(frame, snr_db=12)
        frames.append(frame)
    return np.array(frames)


def generate_approach(num_frames: int = 100) -> np.ndarray:
    """Generate frames with approaching target (decreasing range)"""
    frames = []
    for _ in range(num_frames):
        frame = np.random.normal(0, 0.03, FRAME_SHAPE)

        # Target moving toward radar (range decreasing over chirps)
        start_bin = np.random.randint(30, 50)
        end_bin = np.random.randint(10, 25)
        target_amplitude = np.random.uniform(0.4, 0.8)

        for chirp in range(NUM_CHIRPS):
            # Linear interpolation of range
            progress = chirp / (NUM_CHIRPS - 1)
            current_bin = int(start_bin + (end_bin - start_bin) * progress)

            if 0 <= current_bin < NUM_SAMPLES:
                # Amplitude increases as target approaches
                amp = target_amplitude * (1 + 0.5 * progress)
                frame[current_bin, chirp] += amp
                # Spread
                if current_bin > 0:
                    frame[current_bin - 1, chirp] += amp * 0.4
                if current_bin < NUM_SAMPLES - 1:
                    frame[current_bin + 1, chirp] += amp * 0.4

        frame = add_noise(frame, snr_db=12)
        frames.append(frame)
    return np.array(frames)


def generate_dataset(samples_per_class: int = 500,
                     test_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    """
    Generate complete dataset for training

    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Generating {samples_per_class} samples per class...")

    # Generate data for each class
    no_presence = generate_no_presence(samples_per_class)
    static = generate_static_presence(samples_per_class)
    wave = generate_wave_gesture(samples_per_class)
    approach = generate_approach(samples_per_class)

    # Combine and create labels
    X = np.concatenate([no_presence, static, wave, approach], axis=0)
    y = np.concatenate([
        np.zeros(samples_per_class),
        np.ones(samples_per_class),
        np.full(samples_per_class, 2),
        np.full(samples_per_class, 3)
    ])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Normalize to [-1, 1] range
    X = np.clip(X, -1, 1)

    # Add channel dimension for CNN
    X = X.reshape(-1, NUM_SAMPLES, NUM_CHIRPS, 1)

    # Split train/test
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Dataset generated:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")
    print(f"  Shape:    {X_train.shape[1:]}")

    return X_train, X_test, y_train.astype(np.int32), y_test.astype(np.int32)


def save_dataset(path: str = 'data/radar_dataset.npz'):
    """Generate and save dataset to file"""
    X_train, X_test, y_train, y_test = generate_dataset()
    np.savez(path,
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)
    print(f"Dataset saved to {path}")


if __name__ == '__main__':
    save_dataset()
