"""Unit tests for TrafficFeatureEngineer, road-segment aggregation, and data leakage protection."""

import pytest
import pandas as pd

from src.traffic.features import TrafficFeatureEngineer, get_time_of_day_bucket


@pytest.fixture
def sample_matched_series_df():
    # 4 consecutive time buckets for segment "edge_A"
    return pd.DataFrame([
        # Bucket 1: 08:00 (2 observations)
        {"matched_edge_id": "edge_A", "time_bucket": "2026-09-01T08:00:00Z", "observed_speed_kmh": 40.0, "is_matched": True, "highway_type": "trunk", "road_name": "NH-544", "road_reference": "NH-544"},
        {"matched_edge_id": "edge_A", "time_bucket": "2026-09-01T08:00:00Z", "observed_speed_kmh": 50.0, "is_matched": True, "highway_type": "trunk", "road_name": "NH-544", "road_reference": "NH-544"},
        # Bucket 2: 08:15 (1 observation)
        {"matched_edge_id": "edge_A", "time_bucket": "2026-09-01T08:15:00Z", "observed_speed_kmh": 35.0, "is_matched": True, "highway_type": "trunk", "road_name": "NH-544", "road_reference": "NH-544"},
        # Bucket 3: 08:30 (1 observation)
        {"matched_edge_id": "edge_A", "time_bucket": "2026-09-01T08:30:00Z", "observed_speed_kmh": 30.0, "is_matched": True, "highway_type": "trunk", "road_name": "NH-544", "road_reference": "NH-544"},
        # Bucket 4: 08:45 (1 observation)
        {"matched_edge_id": "edge_A", "time_bucket": "2026-09-01T08:45:00Z", "observed_speed_kmh": 25.0, "is_matched": True, "highway_type": "trunk", "road_name": "NH-544", "road_reference": "NH-544"},
    ])


def test_segment_aggregation(sample_matched_series_df):
    engineer = TrafficFeatureEngineer()
    agg_df = engineer.aggregate_segment_time_buckets(sample_matched_series_df)

    assert len(agg_df) == 4
    # Bucket 1: (40 + 50) / 2 = 45.0 km/h
    b1 = agg_df.iloc[0]
    assert b1["segment_id"] == "edge_A"
    assert b1["mean_speed_kmh"] == 45.0
    assert b1["observation_count"] == 2
    assert b1["coverage_confidence"] == "MODERATE_CONFIDENCE"


def test_zero_data_leakage_in_features(sample_matched_series_df):
    engineer = TrafficFeatureEngineer()
    agg_df = engineer.aggregate_segment_time_buckets(sample_matched_series_df)
    feat_df = engineer.build_model_ready_features(agg_df)

    assert len(feat_df) == 4
    assert "speed_lag_1" in feat_df.columns
    assert "historical_speed_mean" in feat_df.columns

    # Check lag_1 for Bucket 2 (08:15): should equal Bucket 1 mean (45.0 km/h)
    assert feat_df.iloc[1]["speed_lag_1"] == 45.0

    # Check lag_1 for Bucket 3 (08:30): should equal Bucket 2 mean (35.0 km/h)
    assert feat_df.iloc[2]["speed_lag_1"] == 35.0

    # Check historical expanding mean for Bucket 3 (08:30):
    # Strictly mean of past buckets (45.0 and 35.0) -> (45 + 35) / 2 = 40.0
    # Must NOT include current bucket speed (30.0) or future speed (25.0)
    assert feat_df.iloc[2]["historical_speed_mean"] == pytest.approx(40.0, abs=0.1)

    # Check time of day bucket
    assert feat_df.iloc[0]["time_of_day_bucket"] in ("MORNING_PEAK", "MIDDAY_OFFPEAK", "EVENING_PEAK", "NIGHT", "SHOULDER_PERIOD")
