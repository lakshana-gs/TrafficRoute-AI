"""Tests ensuring zero window overlap across chronological split boundaries."""

import numpy as np
import pytest

from src.models.preprocessing import (
    chronological_split,
    generate_sliding_windows,
    load_raw_metr_la,
    prepare_windowed_dataset,
)


def test_no_windows_cross_partition_boundaries():
    """
    Explicitly test that no sliding window crosses from train->val or val->test.
    
    Verifies that the last window of train is strictly inside train_data,
    first window of val is strictly inside val_data, and first window of test is in test_data.
    """
    speed_mx, timestamps, sensor_ids = load_raw_metr_la()
    split = chronological_split(speed_mx, timestamps, sensor_ids, train_ratio=0.70, val_ratio=0.15)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)

    T_train = len(split.train_data)
    T_val = len(split.val_data)
    T_test = len(split.test_data)

    # 1. Total window count matches exact mathematical formula (T_part - 24 + 1)
    assert windows.x_train.shape[0] == T_train - 24 + 1
    assert windows.x_val.shape[0] == T_val - 24 + 1
    assert windows.x_test.shape[0] == T_test - 24 + 1

    # 2. Check content of final train target matches end of train_data
    np.testing.assert_allclose(
        windows.y_train[-1, -1, :, 0],
        split.train_data[-1, :],
        rtol=1e-5,
    )

    # 3. Check content of first val input matches beginning of val_data
    np.testing.assert_allclose(
        windows.x_val[0, 0, :, 0],
        split.val_data[0, :],
        rtol=1e-5,
    )

    # 4. Check content of first test input matches beginning of test_data
    np.testing.assert_allclose(
        windows.x_test[0, 0, :, 0],
        split.test_data[0, :],
        rtol=1e-5,
    )
