"""Tests for Masked MAE, Masked RMSE, Masked MAPE, and horizon evaluator."""

import numpy as np
import pytest

from src.evaluation.metrics import evaluate_horizons, masked_mae, masked_mape, masked_rmse


def test_masked_mae_and_rmse():
    """Verify that zero dropouts and NaNs are ignored in MAE/RMSE calculations."""
    y_true = np.array([50.0, 0.0, 60.0, 0.0, 70.0])  # 0.0 are sensor dropouts
    y_pred = np.array([52.0, 10.0, 58.0, 20.0, 75.0])
    
    # Valid entries: indices 0 (50 vs 52 -> err 2), 2 (60 vs 58 -> err 2), 4 (70 vs 75 -> err 5)
    # Expected MAE: (2 + 2 + 5) / 3 = 3.0
    mae = masked_mae(y_true, y_pred, null_val=0.0)
    assert np.isclose(mae, 3.0)
    
    # Expected RMSE: sqrt((4 + 4 + 25) / 3) = sqrt(33 / 3) = sqrt(11) = 3.31662479
    rmse = masked_rmse(y_true, y_pred, null_val=0.0)
    assert np.isclose(rmse, np.sqrt(11.0))


def test_masked_mape_thresholding():
    """Verify that MAPE ignores near-zero values to avoid division by zero."""
    y_true = np.array([50.0, 1.0, 100.0])  # 1.0 is below min_threshold 8.0
    y_pred = np.array([55.0, 10.0, 110.0])
    
    # Valid entries (>= 8.0): 50.0 (|50-55|/50 = 0.10) and 100.0 (|100-110|/100 = 0.10)
    # Expected MAPE: 10.0%
    mape = masked_mape(y_true, y_pred, min_threshold=8.0)
    assert np.isclose(mape, 10.0)


def test_evaluate_horizons_structure():
    """Verify evaluate_horizons returns correct dictionary schema across multi-step outputs."""
    B, H, N = 10, 12, 5
    y_true = np.full((B, H, N, 1), 60.0, dtype=np.float32)
    y_pred = np.full((B, H, N, 1), 63.0, dtype=np.float32)
    
    res = evaluate_horizons(y_true, y_pred, step_minutes=5)
    
    assert "overall" in res
    assert "horizons" in res
    assert len(res["horizons"]) == 12
    assert "5_min" in res["horizons"]
    assert "60_min" in res["horizons"]
    assert np.isclose(res["overall"]["mae"], 3.0)
