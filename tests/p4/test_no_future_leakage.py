"""Tests verifying zero temporal leakage and training-only scaler isolation."""

import numpy as np
import pytest

from src.models.preprocessing import (
    TrafficStandardScaler,
    chronological_split,
    load_raw_metr_la,
    prepare_windowed_dataset,
)


def test_scaler_zero_leakage_and_isolation():
    """Verify scaler is fit exclusively on training data valid cells."""
    speed_mx, timestamps, sensor_ids = load_raw_metr_la()
    split = chronological_split(speed_mx, timestamps, sensor_ids)

    scaler = TrafficStandardScaler().fit(split.train_data)

    # Compute mean and std manually on valid train entries
    valid_train = split.train_data[split.train_data > 0]
    expected_mean = float(np.mean(valid_train))
    expected_std = float(np.std(valid_train))

    assert abs(scaler.mean - expected_mean) < 1e-4
    assert abs(scaler.std - expected_std) < 1e-4

    # Verify that validation and test distributions differ (proving zero leakage)
    valid_val = split.val_data[split.val_data > 0]
    valid_test = split.test_data[split.test_data > 0]
    assert abs(scaler.mean - float(np.mean(valid_val))) > 0.01
    assert abs(scaler.mean - float(np.mean(valid_test))) > 0.01


def test_no_future_target_in_input_windows():
    """Verify input window X[t] strictly precedes target window Y[t]."""
    speed_mx, timestamps, sensor_ids = load_raw_metr_la()
    split = chronological_split(speed_mx, timestamps, sensor_ids)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)

    # For any sample i, the last step of X is t+11, and the first step of Y is t+12
    for i in [0, 100, 1000, 5000]:
        x_last = windows.x_train[i, -1, :, 0]
        y_first = windows.y_train[i, 0, :, 0]

        # In chronological sequence, y_first is exactly 1 step after x_last
        expected_x_last = split.train_data[i + 11]
        expected_y_first = split.train_data[i + 12]

        np.testing.assert_allclose(x_last, expected_x_last, rtol=1e-5)
        np.testing.assert_allclose(y_first, expected_y_first, rtol=1e-5)
