"""Unit tests for TrafficCleaner and anomaly tagging."""

import pytest
import pandas as pd

from src.traffic.cleaning import TrafficCleaner
from src.traffic.schemas import QualityFlag


@pytest.fixture
def sample_dirty_df():
    return pd.DataFrame([
        # 1. Normal valid
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P1", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 45.0},
        # 2. Legitimate stopped (0.0 km/h)
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P2", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 0.0},
        # 3. Negative speed anomaly
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P3", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": -15.0},
        # 4. Extreme outlier (> 150 km/h)
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P4", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 185.0},
        # 5. Out of bounds GPS
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P5", "latitude": 13.08, "longitude": 80.27, "observed_speed_kmh": 40.0},
        # 6. Duplicate of P1
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P1", "latitude": 11.02, "longitude": 77.01, "observed_speed_kmh": 45.0},
    ])


def test_cleaner_retains_valid_and_stopped_records(sample_dirty_df):
    cleaner = TrafficCleaner()
    clean_df, report = cleaner.clean_observations(sample_dirty_df)

    assert report.raw_count == 6
    assert report.total_retained == 2  # P1 and P2
    assert report.total_removed == 4
    assert report.stopped_count == 1
    assert report.negative_speed_count == 1
    assert report.extreme_speed_count == 1
    assert report.out_of_bounds_count == 1
    assert report.duplicate_count == 1

    # Verify quality flags on retained records
    p1_row = clean_df[clean_df["traffic_entity_id"] == "P1"].iloc[0]
    p2_row = clean_df[clean_df["traffic_entity_id"] == "P2"].iloc[0]

    assert p1_row["quality_flag"] == QualityFlag.VALID.value
    assert p2_row["quality_flag"] == QualityFlag.STOPPED.value
