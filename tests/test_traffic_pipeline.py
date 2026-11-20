"""Integration tests for the complete TrafficDataPipeline."""

import pytest
import pandas as pd

from src.network.extraction import load_or_extract_pilot_network
from src.traffic.pipeline import TrafficDataPipeline


@pytest.fixture(scope="module")
def pilot_graph():
    return load_or_extract_pilot_network(force_reextract=False)


def test_traffic_pipeline_end_to_end(pilot_graph, tmp_path):
    pipeline = TrafficDataPipeline(
        road_graph=pilot_graph,
        bucket_size_minutes=15,
        max_snap_distance_m=75.0,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    out_dir = tmp_path / "traffic_pipeline_output"
    model_ready_df, summary = pipeline.run(output_dir=out_dir)

    # 1. Pipeline execution assertions
    assert len(model_ready_df) > 0
    assert summary.audit.total_records > 0
    assert summary.cleaning.total_retained > 0
    assert summary.spatial.matched_observations > 0
    assert summary.model_ready_rows == len(model_ready_df)

    # 2. File output assertions
    assert (out_dir / "processed" / "model_ready_traffic_dataset.parquet").exists()
    assert (out_dir / "processed" / "train_split.parquet").exists()
    assert (out_dir / "processed" / "val_split.parquet").exists()
    assert (out_dir / "processed" / "test_split.parquet").exists()
    assert (out_dir / "metadata" / "coimbatore_avinashi_processed_provenance.json").exists()

    # 3. Chronological split assertions (Zero temporal overlap)
    train_df = pd.read_parquet(out_dir / "processed" / "train_split.parquet")
    val_df = pd.read_parquet(out_dir / "processed" / "val_split.parquet")
    test_df = pd.read_parquet(out_dir / "processed" / "test_split.parquet")

    assert len(train_df) + len(val_df) + len(test_df) == len(model_ready_df)

    train_max_time = train_df["time_bucket"].max()
    val_min_time = val_df["time_bucket"].min()
    val_max_time = val_df["time_bucket"].max()
    test_min_time = test_df["time_bucket"].min()

    # Verify strict temporal sequencing
    assert train_max_time < val_min_time
    assert val_max_time < test_min_time

    # 4. Feature integrity assertions
    required_cols = [
        "segment_id",
        "time_bucket",
        "mean_speed_kmh",
        "median_speed_kmh",
        "speed_std_kmh",
        "observation_count",
        "hour_ist",
        "day_of_week_ist",
        "is_weekend_ist",
        "time_of_day_bucket",
        "speed_lag_1",
        "historical_speed_mean",
    ]
    for col in required_cols:
        assert col in model_ready_df.columns
        assert not model_ready_df[col].isna().any()
