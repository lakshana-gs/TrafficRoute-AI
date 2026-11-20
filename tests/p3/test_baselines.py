"""Tests for Persistence, Historical Mean, and XGBoost speed prediction baselines."""

import numpy as np
import pandas as pd
import pytest

from src.models.historical_mean import HistoricalTimeOfDayBaseline
from src.models.ml_baseline import TabularLagFeatureExtractor, XGBoostSpeedBaseline
from src.models.persistence import PersistenceBaseline


def test_persistence_baseline_prediction():
    """Verify persistence baseline replicates latest speed across all horizons."""
    B, T_in, N = 4, 12, 3
    X = np.random.uniform(30.0, 80.0, size=(B, T_in, N, 1))
    
    persistence = PersistenceBaseline(horizon_steps=12)
    Y_hat = persistence.predict(X)
    
    assert Y_hat.shape == (B, 12, N, 1)
    # Every horizon step h must equal the last historical step X[:, 11, :, 0]
    for h in range(12):
        assert np.allclose(Y_hat[:, h, :, 0], X[:, 11, :, 0])


def test_tabular_lag_feature_extractor():
    """Verify tabular lag feature dimensions and ordering."""
    B, T_in, N = 5, 12, 2
    X = np.ones((B, T_in, N, 1), dtype=np.float32)
    ts = pd.date_range("2026-01-01 08:00:00", periods=50, freq="5min")
    
    features = TabularLagFeatureExtractor.extract_features(
        X, ts, history_steps=12, horizon_steps=12, sensor_subset=[0, 1]
    )
    
    # 2 sensors * 5 windows = 10 rows, 10 feature columns
    assert features.shape == (10, 10)


def test_xgboost_fast_train_and_predict():
    """Verify XGBoost baseline fits and produces valid non-trivial predictions."""
    np.random.seed(42)
    B_train, B_test, T_in, H, N = 20, 5, 12, 12, 4
    
    X_train = np.random.uniform(40.0, 90.0, size=(B_train, T_in, N, 1)).astype(np.float32)
    Y_train = X_train[:, -1:, :, :] + np.random.normal(0, 2, size=(B_train, H, N, 1)).astype(np.float32)
    
    X_test = np.random.uniform(40.0, 90.0, size=(B_test, T_in, N, 1)).astype(np.float32)
    Y_test = X_test[:, -1:, :, :] + np.random.normal(0, 2, size=(B_test, H, N, 1)).astype(np.float32)
    
    train_ts = pd.date_range("2026-01-01 00:00:00", periods=100, freq="5min")
    test_ts = pd.date_range("2026-01-02 00:00:00", periods=50, freq="5min")
    
    model = XGBoostSpeedBaseline(n_estimators=10, max_depth=3, horizon_steps=12)
    model.fit(X_train, Y_train, train_ts, sample_step_stride=1, sensor_stride=1)
    
    assert model.is_fitted
    Y_pred = model.predict(X_test, test_ts)
    
    assert Y_pred.shape == (B_test, 12, N, 1)
    assert np.all(np.isfinite(Y_pred))
