"""Strict leakage verification tests for P3."""

import numpy as np
import pandas as pd
import pytest

from src.models.historical_mean import HistoricalTimeOfDayBaseline
from src.models.preprocessing import (
    TrafficStandardScaler,
    chronological_split,
    prepare_windowed_dataset,
)


def test_scaler_zero_leakage():
    """Verify that the standard scaler computes statistics strictly from training data."""
    train_data = np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float64)
    val_data = np.array([[500.0, 600.0]], dtype=np.float64)  # Extreme out-of-distribution values
    
    scaler = TrafficStandardScaler(null_val=0.0)
    scaler.fit(train_data)
    
    # Train mean: (50+60+70+80)/4 = 65.0
    assert scaler.mean == 65.0
    
    # Modifying val_data or test_data must have ZERO impact on fitted scaler
    val_transformed = scaler.transform(val_data)
    assert scaler.mean == 65.0  # Still 65.0, unaffected by val_data


def test_historical_mean_zero_leakage():
    """Verify that historical mean baseline uses training data only."""
    train_ts = pd.date_range("2026-01-01 00:00:00", periods=288, freq="5min")
    val_ts = pd.date_range("2026-01-02 00:00:00", periods=288, freq="5min")
    
    # Training data: uniform speed 40.0 km/h
    train_data = np.full((288, 2), 40.0, dtype=np.float32)
    # Validation data: uniform speed 100.0 km/h
    val_data = np.full((288, 2), 100.0, dtype=np.float32)
    
    model = HistoricalTimeOfDayBaseline(horizon_steps=12)
    model.fit(train_data, train_ts)
    
    # Prediction on validation timestamps must reflect the training mean (40.0), NOT 100.0
    val_pred = model.predict_for_timestamps(val_ts, history_steps=12)
    
    assert np.allclose(val_pred, 40.0)


def test_no_future_targets_in_input_windows():
    """Verify that input window X[:, :, :, :] contains strictly historical indices < target indices."""
    T = 200
    N = 2
    data = np.arange(T * N).reshape(T, N)
    timestamps = pd.date_range("2026-01-01", periods=T, freq="5min")
    sensor_ids = ["s1", "s2"]
    
    split = chronological_split(data, timestamps, sensor_ids, train_ratio=0.60, val_ratio=0.20)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)
    
    # In training windows:
    for i in range(len(windows.x_train)):
        max_history_val = np.max(windows.x_train[i])
        min_target_val = np.min(windows.y_train[i])
        assert max_history_val < min_target_val, "Future target leaked into input window!"
