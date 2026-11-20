"""Tests for METR-LA preprocessing, unit conversion, chronological splitting, and window generation."""

import numpy as np
import pandas as pd
import pytest

from src.models.preprocessing import (
    MPH_TO_KMH,
    chronological_split,
    generate_sliding_windows,
    load_raw_metr_la,
    load_sensor_graph,
    prepare_windowed_dataset,
    TrafficStandardScaler,
)


def test_unit_conversion_and_raw_loading():
    """Verify raw METR-LA loading and mph to km/h conversion."""
    speed_matrix_kmh, timestamps, sensor_ids = load_raw_metr_la(convert_to_kmh=True)
    
    assert speed_matrix_kmh.shape == (34272, 207)
    assert len(timestamps) == 34272
    assert len(sensor_ids) == 207
    assert speed_matrix_kmh.dtype == np.float64
    
    # Check max speed in km/h is roughly 70 mph * 1.609344 = 112.65 km/h
    assert np.max(speed_matrix_kmh) <= 113.0
    assert np.min(speed_matrix_kmh) >= 0.0


def test_sensor_graph_loading():
    """Verify sensor graph structure and dimensions."""
    sensor_ids, sensor_id_map, adj_mx = load_sensor_graph()
    
    assert len(sensor_ids) == 207
    assert len(sensor_id_map) == 207
    assert adj_mx.shape == (207, 207)
    assert np.allclose(np.diag(adj_mx), 1.0)
    assert np.all(adj_mx >= 0.0) and np.all(adj_mx <= 1.0)


def test_chronological_split_boundaries():
    """Verify that chronological splitting partitions data with zero temporal overlap."""
    T = 1000
    N = 5
    synthetic_data = np.arange(T * N).reshape(T, N)
    timestamps = pd.date_range("2026-01-01", periods=T, freq="5min")
    sensor_ids = [f"sensor_{i}" for i in range(N)]
    
    split = chronological_split(synthetic_data, timestamps, sensor_ids, train_ratio=0.70, val_ratio=0.15)
    
    # 70% of 1000 = 700, 15% = 150, 15% = 150
    assert len(split.train_data) == 700
    assert len(split.val_data) == 150
    assert len(split.test_data) == 150
    
    assert split.train_timestamps[-1] < split.val_timestamps[0]
    assert split.val_timestamps[-1] < split.test_timestamps[0]


def test_zero_leakage_window_generation():
    """Verify that sliding windows do not cross partition boundaries."""
    T_part = 100
    N = 10
    data = np.arange(T_part * N).reshape(T_part, N)
    
    X, Y = generate_sliding_windows(data, history_steps=12, horizon_steps=12)
    
    # Expected windows: 100 - 24 + 1 = 77
    assert X.shape == (77, 12, N, 1)
    assert Y.shape == (77, 12, N, 1)
    
    # Verify exact temporal continuity within window
    for w in range(len(X)):
        # Last history step of window w
        x_last = X[w, 11, 0, 0]
        # First target step of window w
        y_first = Y[w, 0, 0, 0]
        # In sequential data with step N=10, y_first must equal x_last + 10
        assert y_first == x_last + 10
