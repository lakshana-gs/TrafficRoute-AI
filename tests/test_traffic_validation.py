"""Unit tests for TrafficValidator and anomaly detection."""

from datetime import datetime, timezone
import pytest
import pandas as pd

from src.traffic.validation import TrafficValidator


@pytest.fixture
def sample_raw_traffic_df():
    return pd.DataFrame([
        # Valid normal record
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P1", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 45.0},
        # Negative speed
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P2", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": -10.0},
        # Extreme speed outlier (> 150 km/h)
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P3", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 220.0},
        # Zero speed (legitimate stopped)
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P4", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 0.0},
        # Out of bounds coordinates (Delhi)
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P5", "latitude": 28.61, "longitude": 77.20, "observed_speed_kmh": 50.0},
        # Duplicate of P1
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P1", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 45.0},
    ])


def test_traffic_validator_audit(sample_raw_traffic_df):
    validator = TrafficValidator()
    audit = validator.audit_dataset(sample_raw_traffic_df)

    assert audit.total_records == 6
    assert audit.negative_speeds == 1
    assert audit.extreme_high_speeds == 1
    assert audit.out_of_bounds_coordinates == 1
    assert audit.zero_speeds == 1
    assert audit.duplicate_records == 1
    assert audit.valid_records == 2  # P1 (first) and P4 (valid stop)
    assert audit.invalid_records == 4
