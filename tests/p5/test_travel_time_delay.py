"""Unit tests for P5 Travel Time and Delay Calculations."""

import math
import pytest

from src.traffic.congestion import (
    CongestionStatus,
    DelayCalculator,
)


def test_hand_computable_travel_times_and_delays():
    """Verify exact hand-computable travel time and delay values for canonical scenarios."""
    length_km = 10.0
    free_flow_kmh = 60.0  # 10 km at 60 km/h = 10.0 minutes baseline

    # Case 1: Speed equals free-flow speed (delay == 0.0)
    tt_pred, tt_ff, delay, status, _ = DelayCalculator.calculate(length_km, 60.0, free_flow_kmh)
    assert tt_ff == 10.0
    assert tt_pred == 10.0
    assert delay == 0.0
    assert status == CongestionStatus.NORMAL

    # Case 2: Speed is half of free-flow speed (30 km/h -> 20 min travel time -> delay = +10.0 min)
    tt_pred, tt_ff, delay, status, _ = DelayCalculator.calculate(length_km, 30.0, free_flow_kmh)
    assert tt_ff == 10.0
    assert tt_pred == 20.0
    assert delay == 10.0
    assert status == CongestionStatus.NORMAL

    # Case 3: Speed is one-fourth of free-flow speed (15 km/h -> 40 min travel time -> delay = +30.0 min)
    tt_pred, tt_ff, delay, status, _ = DelayCalculator.calculate(length_km, 15.0, free_flow_kmh)
    assert tt_ff == 10.0
    assert tt_pred == 40.0
    assert delay == 30.0
    assert status == CongestionStatus.NORMAL

    # Case 4: Moderate speed (45 km/h -> 60 * 10 / 45 = 13.333333 min -> delay = +3.333333 min)
    tt_pred, tt_ff, delay, status, _ = DelayCalculator.calculate(length_km, 45.0, free_flow_kmh)
    assert tt_ff == 10.0
    assert pytest.approx(tt_pred, 1e-5) == 13.333333
    assert pytest.approx(delay, 1e-5) == 3.333333
    assert status == CongestionStatus.NORMAL


def test_speed_exceeding_free_flow_preserves_signed_negative_delay():
    """Verify that predictions exceeding free-flow speed preserve signed negative delay (time savings)."""
    length_km = 10.0
    free_flow_kmh = 60.0  # free-flow time = 10.0 min

    # Speed = 80 km/h -> travel time = 60 * 10 / 80 = 7.5 min
    # Signed delay = 7.5 - 10.0 = -2.5 minutes (2.5 min time savings)
    tt_pred, tt_ff, delay, status, msg = DelayCalculator.calculate(length_km, 80.0, free_flow_kmh)
    assert tt_ff == 10.0
    assert tt_pred == 7.5
    assert delay == -2.5
    assert status == CongestionStatus.FASTER_THAN_FREE_FLOW
    assert "savings" in msg.lower()


def test_zero_predicted_speed_handling_no_division_by_zero():
    """Verify that zero predicted speed is handled safely without division by zero.
    
    Travel time cannot be computed as a finite number, so estimated travel time
    and delay must be None with status STOPPED.
    """
    length_km = 5.0
    free_flow_kmh = 50.0  # free-flow time = 6.0 min

    tt_pred, tt_ff, delay, status, msg = DelayCalculator.calculate(length_km, 0.0, free_flow_kmh)
    assert tt_ff == 6.0
    assert tt_pred is None
    assert delay is None
    assert status == CongestionStatus.STOPPED
    assert "undefined" in msg.lower() or "stopped" in msg.lower()


def test_missing_and_nan_speed_handling():
    """Verify that None and NaN speeds do not crash and leave travel time/delay as None."""
    length_km = 5.0
    free_flow_kmh = 50.0

    tt_pred, tt_ff, delay, status, _ = DelayCalculator.calculate(length_km, None, free_flow_kmh)
    assert tt_ff == 6.0
    assert tt_pred is None
    assert delay is None
    assert status == CongestionStatus.MISSING_SPEED

    tt_pred_nan, tt_ff_nan, delay_nan, status_nan, _ = DelayCalculator.calculate(length_km, float("nan"), free_flow_kmh)
    assert tt_ff_nan == 6.0
    assert tt_pred_nan is None
    assert delay_nan is None
    assert status_nan == CongestionStatus.MISSING_SPEED


def test_invalid_predicted_speed_handling():
    """Verify that negative and infinite predicted speeds are rejected safely."""
    length_km = 5.0
    free_flow_kmh = 50.0

    tt_pred, tt_ff, delay, status, _ = DelayCalculator.calculate(length_km, -10.0, free_flow_kmh)
    assert tt_pred is None
    assert delay is None
    assert status == CongestionStatus.INVALID_SPEED

    tt_pred_inf, _, delay_inf, status_inf, _ = DelayCalculator.calculate(length_km, float("inf"), free_flow_kmh)
    assert tt_pred_inf is None
    assert delay_inf is None
    assert status_inf == CongestionStatus.INVALID_SPEED


def test_invalid_segment_length_handling():
    """Verify that zero, negative, NaN, and infinite segment lengths are rejected."""
    for invalid_l in [0.0, -5.0, float("nan"), float("inf"), None]:
        tt_pred, tt_ff, delay, status, msg = DelayCalculator.calculate(invalid_l, 50.0, 50.0)
        assert tt_pred is None
        assert tt_ff is None
        assert delay is None
        assert status == CongestionStatus.INVALID_SEGMENT_LENGTH


def test_invalid_free_flow_speed_handling():
    """Verify that zero, negative, NaN, and infinite free-flow speeds are rejected."""
    for invalid_ff in [0.0, -60.0, float("nan"), float("inf"), None]:
        tt_pred, tt_ff, delay, status, msg = DelayCalculator.calculate(5.0, 50.0, invalid_ff)
        assert tt_pred is None
        assert tt_ff is None
        assert delay is None
        assert status == CongestionStatus.INVALID_FREE_FLOW_SPEED
