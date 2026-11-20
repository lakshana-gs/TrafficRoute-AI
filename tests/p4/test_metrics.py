"""Tests for P4 multi-horizon and sensor-level evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import evaluate_horizons, masked_mae, masked_mape, masked_rmse
from src.models.p4.evaluator import compute_missing_data_statistics, compute_sensor_level_metrics


def test_evaluate_horizons_structure_and_values():
    """Verify evaluate_horizons computes per-horizon metrics correctly."""
    B, H, N = 10, 12, 5
    y_true = np.ones((B, H, N)) * 80.0
    y_pred = np.ones((B, H, N)) * 70.0  # Error = 10.0 km/h

    res = evaluate_horizons(y_true, y_pred, step_minutes=5, null_val=0.0)

    assert abs(res["overall"]["mae"] - 10.0) < 1e-4
    assert abs(res["overall"]["rmse"] - 10.0) < 1e-4
    assert abs(res["overall"]["mape"] - 12.5) < 1e-4

    assert len(res["horizons"]) == 12
    for h in range(1, 13):
        h_min = f"{h * 5}_min"
        assert h_min in res["horizons"]
        assert abs(res["horizons"][h_min]["mae"] - 10.0) < 1e-4


def test_missing_data_and_sensor_metrics():
    """Verify sensor-level analysis and missing data statistics."""
    B, H, N = 10, 12, 3
    y_true = np.ones((B, H, N)) * 60.0
    y_true[0, 0, 0] = 0.0  # Missing point on sensor 0
    y_pred = np.ones((B, H, N)) * 55.0

    missing_stats = compute_missing_data_statistics(y_true, null_val=0.0)
    assert missing_stats["masked_observations"] == 1
    assert missing_stats["valid_observations"] == (B * H * N) - 1

    sensor_stats = compute_sensor_level_metrics(y_true, y_pred, sensor_ids=["s0", "s1", "s2"])
    assert sensor_stats["num_sensors"] == 3
    assert len(sensor_stats["best_performing_sensors"]) <= 3
