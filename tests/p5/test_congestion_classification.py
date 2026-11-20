"""Unit tests for P5 Congestion Classification, boundaries, and configurations."""

import math
import pytest

from src.traffic.congestion import (
    CongestionClassifier,
    CongestionLevel,
    CongestionStatus,
    CongestionThresholdConfig,
)


def test_default_congestion_categories():
    """Verify standard speed ratio classifications with default thresholds."""
    ff = 100.0  # free-flow 100 km/h

    # Low congestion (ratio >= 0.80)
    level, status, ratio, msg = CongestionClassifier.classify(85.0, ff)
    assert level == CongestionLevel.LOW
    assert status == CongestionStatus.NORMAL
    assert ratio == 0.85

    # Moderate congestion (0.60 <= ratio < 0.80)
    level, status, ratio, msg = CongestionClassifier.classify(70.0, ff)
    assert level == CongestionLevel.MODERATE
    assert status == CongestionStatus.NORMAL
    assert ratio == 0.70

    # High congestion (0.40 <= ratio < 0.60)
    level, status, ratio, msg = CongestionClassifier.classify(50.0, ff)
    assert level == CongestionLevel.HIGH
    assert status == CongestionStatus.NORMAL
    assert ratio == 0.50

    # Severe congestion (ratio < 0.40)
    level, status, ratio, msg = CongestionClassifier.classify(25.0, ff)
    assert level == CongestionLevel.SEVERE
    assert status == CongestionStatus.NORMAL
    assert ratio == 0.25


def test_exact_threshold_boundaries():
    """Test exact threshold boundary values and epsilon differences."""
    ff = 100.0

    # Boundary: ratio = 0.80 -> LOW
    level, status, ratio, _ = CongestionClassifier.classify(80.0, ff)
    assert level == CongestionLevel.LOW
    assert ratio == 0.80

    # Boundary: ratio = 0.7999 -> MODERATE
    level, status, ratio, _ = CongestionClassifier.classify(79.99, ff)
    assert level == CongestionLevel.MODERATE
    assert ratio == 0.7999

    # Boundary: ratio = 0.60 -> MODERATE
    level, status, ratio, _ = CongestionClassifier.classify(60.0, ff)
    assert level == CongestionLevel.MODERATE
    assert ratio == 0.60

    # Boundary: ratio = 0.5999 -> HIGH
    level, status, ratio, _ = CongestionClassifier.classify(59.99, ff)
    assert level == CongestionLevel.HIGH
    assert ratio == 0.5999

    # Boundary: ratio = 0.40 -> HIGH
    level, status, ratio, _ = CongestionClassifier.classify(40.0, ff)
    assert level == CongestionLevel.HIGH
    assert ratio == 0.40

    # Boundary: ratio = 0.3999 -> SEVERE
    level, status, ratio, _ = CongestionClassifier.classify(39.99, ff)
    assert level == CongestionLevel.SEVERE
    assert ratio == 0.3999


def test_speed_exceeding_free_flow():
    """Verify handling when predicted speed exceeds free-flow speed."""
    level, status, ratio, msg = CongestionClassifier.classify(120.0, 100.0)
    assert level == CongestionLevel.LOW
    assert status == CongestionStatus.FASTER_THAN_FREE_FLOW
    assert ratio == 1.20
    assert "exceeds free-flow" in msg


def test_exact_zero_predicted_speed():
    """Verify that zero predicted speed is classified as severe with explicit STOPPED status."""
    level, status, ratio, msg = CongestionClassifier.classify(0.0, 60.0)
    assert level == CongestionLevel.SEVERE
    assert status == CongestionStatus.STOPPED
    assert ratio == 0.0
    assert "stopped" in msg.lower()


def test_missing_and_nan_predicted_speed():
    """Verify that None and NaN speeds are classified as UNKNOWN without assuming severe congestion."""
    level, status, ratio, msg = CongestionClassifier.classify(None, 60.0)
    assert level == CongestionLevel.UNKNOWN
    assert status == CongestionStatus.MISSING_SPEED
    assert ratio is None

    level_nan, status_nan, ratio_nan, _ = CongestionClassifier.classify(float("nan"), 60.0)
    assert level_nan == CongestionLevel.UNKNOWN
    assert status_nan == CongestionStatus.MISSING_SPEED
    assert ratio_nan is None


def test_invalid_predicted_speed():
    """Verify that negative and infinite speeds are classified as INVALID_SPEED."""
    level, status, ratio, _ = CongestionClassifier.classify(-15.0, 60.0)
    assert level == CongestionLevel.UNKNOWN
    assert status == CongestionStatus.INVALID_SPEED
    assert ratio is None

    level_inf, status_inf, ratio_inf, _ = CongestionClassifier.classify(float("inf"), 60.0)
    assert level_inf == CongestionLevel.UNKNOWN
    assert status_inf == CongestionStatus.INVALID_SPEED
    assert ratio_inf is None


def test_invalid_free_flow_speed():
    """Verify that non-positive, NaN, and non-finite free-flow speeds are rejected safely."""
    for invalid_ff in [0.0, -50.0, float("nan"), float("inf"), float("-inf"), None]:
        level, status, ratio, _ = CongestionClassifier.classify(50.0, invalid_ff)
        assert level == CongestionLevel.UNKNOWN
        assert status == CongestionStatus.INVALID_FREE_FLOW_SPEED
        assert ratio is None


def test_custom_configurable_thresholds():
    """Verify custom threshold configuration changes boundaries correctly."""
    custom_cfg = CongestionThresholdConfig(
        low_min_ratio=0.85,
        moderate_min_ratio=0.70,
        high_min_ratio=0.50,
    )
    ff = 100.0

    # Under custom config, 80% is now moderate (since low_min is 0.85)
    level, status, ratio, _ = CongestionClassifier.classify(80.0, ff, config=custom_cfg)
    assert level == CongestionLevel.MODERATE

    # 86% is low
    level_low, _, _, _ = CongestionClassifier.classify(86.0, ff, config=custom_cfg)
    assert level_low == CongestionLevel.LOW


def test_invalid_threshold_configurations_raise_error():
    """Verify that invalid or inverted threshold configurations raise ValueError."""
    # Inverted: high >= moderate
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        CongestionThresholdConfig(low_min_ratio=0.80, moderate_min_ratio=0.50, high_min_ratio=0.60)

    # Inverted: moderate >= low
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        CongestionThresholdConfig(low_min_ratio=0.60, moderate_min_ratio=0.70, high_min_ratio=0.40)

    # Out of range: > 1.0
    with pytest.raises(ValueError, match="range"):
        CongestionThresholdConfig(low_min_ratio=1.20, moderate_min_ratio=0.60, high_min_ratio=0.40)

    # Out of range: <= 0.0
    with pytest.raises(ValueError, match="range"):
        CongestionThresholdConfig(low_min_ratio=0.80, moderate_min_ratio=0.60, high_min_ratio=-0.10)
