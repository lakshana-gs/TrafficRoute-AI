"""Unit tests for TemporalNormalizer, calendar feature generation, and gap detection."""

import pytest
import pandas as pd

from src.traffic.temporal import TemporalNormalizer


@pytest.fixture
def sample_temporal_df():
    return pd.DataFrame([
        {"timestamp": "2026-09-01T08:05:00Z", "traffic_entity_id": "P1", "observed_speed_kmh": 40.0},
        {"timestamp": "2026-09-01T08:20:00Z", "traffic_entity_id": "P1", "observed_speed_kmh": 42.0},
        {"timestamp": "2026-09-01T08:35:00Z", "traffic_entity_id": "P1", "observed_speed_kmh": 38.0},
        {"timestamp": "2026-09-01T14:00:00Z", "traffic_entity_id": "P1", "observed_speed_kmh": 60.0},  # Gap > 5 hours
    ])


def test_temporal_normalization_and_bucketing(sample_temporal_df):
    normalizer = TemporalNormalizer(bucket_size_minutes=15, expected_interval_s=900.0)
    df_norm, report = normalizer.normalize_and_bucket(sample_temporal_df)

    assert len(df_norm) == 4
    assert "time_bucket" in df_norm.columns
    assert "hour_ist" in df_norm.columns
    assert "is_weekend_ist" in df_norm.columns

    # 08:05 UTC + 5:30 IST = 13:35 IST -> hour_ist = 13
    assert df_norm.iloc[0]["hour_ist"] == 13

    # Check 15-minute floor bucketing: 08:05 -> 08:00
    assert df_norm.iloc[0]["time_bucket"].minute == 0
    # 08:20 -> 08:15
    assert df_norm.iloc[1]["time_bucket"].minute == 15

    # Check gap detection
    assert report.total_gaps_detected >= 1
    assert report.longest_gap_hours >= 5.0
