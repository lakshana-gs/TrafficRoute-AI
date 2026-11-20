"""Unit tests for traffic data ingestion and provenance tracking."""

from datetime import datetime, timezone
import pytest
import pandas as pd

from src.traffic.ingestion import generate_empirical_pilot_corridor_telemetry, load_raw_traffic_csv
from src.traffic.provenance import TrafficDatasetProvenance
from src.traffic.schemas import RawTrafficObservation, VehicleClass


def test_raw_traffic_observation_schema():
    obs = RawTrafficObservation(
        timestamp=datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc),
        traffic_entity_id="PROBE_0001",
        latitude=11.0280,
        longitude=77.0210,
        observed_speed_kmh=42.5,
        heading=65.0,
        vehicle_class=VehicleClass.CAR,
        road_name="Hope College Junction",
        road_reference="Avinashi Road",
    )
    assert obs.traffic_entity_id == "PROBE_0001"
    assert obs.observed_speed_kmh == 42.5
    assert obs.vehicle_class == VehicleClass.CAR


def test_invalid_coordinates_raise_error():
    with pytest.raises(ValueError):
        RawTrafficObservation(
            timestamp=datetime.now(timezone.utc),
            traffic_entity_id="PROBE_INVALID",
            latitude=110.0,  # Invalid latitude > 90
            longitude=77.0,
            observed_speed_kmh=50.0,
        )


def test_ingestion_generates_valid_dataframe(tmp_path):
    out_file = tmp_path / "test_telemetry.csv"
    df = generate_empirical_pilot_corridor_telemetry(
        output_csv_path=out_file,
        days=1,
        probes_count=20,
        seed=123,
    )
    assert len(df) > 100
    assert "timestamp" in df.columns
    assert "observed_speed_kmh" in df.columns
    assert "traffic_entity_id" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns

    loaded_df = load_raw_traffic_csv(out_file)
    assert len(loaded_df) == len(df)
    assert pd.api.types.is_datetime64_any_dtype(loaded_df["timestamp"])
