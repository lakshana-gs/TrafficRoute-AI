"""Unit tests for P5 Output Contract, Provenance, Engine Orchestration, and Geographic Integrity."""

from datetime import datetime, timezone
import pytest

from src.traffic.congestion import (
    CongestionIntelligenceEngine,
    CongestionLevel,
    CongestionStatus,
    SegmentCongestionResult,
)


def test_output_contract_fields_and_types():
    """Verify that SegmentCongestionResult provides all required canonical fields and types."""
    engine = CongestionIntelligenceEngine()
    ts = datetime(2026, 10, 10, 8, 30, tzinfo=timezone.utc)

    result = engine.evaluate_segment(
        segment_id="osm_edge_1001",
        segment_length_km=2.5,
        free_flow_speed_kmh=50.0,
        predicted_speed_kmh=25.0,
        forecast_timestamp=ts,
        forecast_horizon_minutes=15,
        model_source="DCRNN_SPATIOTEMPORAL",
        source_geography="Tamil Nadu (Coimbatore-Avinashi Road)",
        metadata={"lane_count": 4, "highway_type": "primary"},
    )

    # Core identification
    assert result.segment_id == "osm_edge_1001"
    assert result.forecast_timestamp == ts
    assert result.forecast_horizon_minutes == 15

    # Speeds and ratio
    assert result.predicted_speed_kmh == 25.0
    assert result.free_flow_speed_kmh == 50.0
    assert result.speed_ratio == 0.50

    # Classifications
    assert result.congestion_level == CongestionLevel.HIGH
    assert result.congestion_status == CongestionStatus.NORMAL
    assert result.is_valid is True

    # Travel times and delay (L = 2.5 km, v = 25 km/h -> 6.0 min; ff = 50 km/h -> 3.0 min; delay = +3.0 min)
    assert result.segment_length_km == 2.5
    assert result.estimated_travel_time_minutes == 6.0
    assert result.free_flow_travel_time_minutes == 3.0
    assert result.estimated_delay_minutes == 3.0

    # Lineage and provenance
    assert result.model_source == "DCRNN_SPATIOTEMPORAL"
    assert result.source_geography == "Tamil Nadu (Coimbatore-Avinashi Road)"
    assert result.metadata["lane_count"] == 4
    assert result.metadata["highway_type"] == "primary"


def test_optional_unavailable_fields_remain_explicitly_none():
    """Verify that optional unavailable fields are not fabricated with synthetic placeholders."""
    engine = CongestionIntelligenceEngine()

    result = engine.evaluate_segment(
        segment_id="sensor_767541",
        segment_length_km=1.0,
        free_flow_speed_kmh=65.0,
        predicted_speed_kmh=65.0,
        forecast_timestamp=None,
        forecast_horizon_minutes=None,
        model_source="METR_LA_BASELINE",
        source_geography="Los Angeles County, CA (METR-LA Loop Detectors)",
    )

    assert result.forecast_timestamp is None
    assert result.forecast_horizon_minutes is None
    assert result.congestion_level == CongestionLevel.LOW
    assert result.estimated_delay_minutes == 0.0


def test_deterministic_repeated_calculations():
    """Verify that calculations are strictly deterministic and idempotent over repeated invocations."""
    engine = CongestionIntelligenceEngine()

    res1 = engine.evaluate_segment(
        segment_id="seg_test_42",
        segment_length_km=3.14159,
        free_flow_speed_kmh=45.0,
        predicted_speed_kmh=22.5,
        forecast_horizon_minutes=30,
    )

    res2 = engine.evaluate_segment(
        segment_id="seg_test_42",
        segment_length_km=3.14159,
        free_flow_speed_kmh=45.0,
        predicted_speed_kmh=22.5,
        forecast_horizon_minutes=30,
    )

    assert res1.estimated_travel_time_minutes == res2.estimated_travel_time_minutes
    assert res1.estimated_delay_minutes == res2.estimated_delay_minutes
    assert res1.speed_ratio == res2.speed_ratio
    assert res1.congestion_level == res2.congestion_level
    assert res1.congestion_status == res2.congestion_status


def test_batch_evaluation_processing():
    """Verify that batch evaluation correctly processes heterogeneous lists of segment inputs."""
    engine = CongestionIntelligenceEngine()
    batch_input = [
        {"segment_id": "seg_1", "segment_length_km": 2.0, "free_flow_speed_kmh": 60.0, "predicted_speed_kmh": 60.0},
        {"segment_id": "seg_2", "segment_length_km": 1.5, "free_flow_speed_kmh": 40.0, "predicted_speed_kmh": 10.0},
        {"segment_id": "seg_3", "segment_length_km": 0.8, "free_flow_speed_kmh": 30.0, "predicted_speed_kmh": 0.0},
    ]

    results = engine.evaluate_batch(batch_input, default_geography="Tamil Nadu (Coimbatore-Avinashi Road)")

    assert len(results) == 3
    assert results[0].congestion_level == CongestionLevel.LOW
    assert results[0].estimated_delay_minutes == 0.0

    assert results[1].congestion_level == CongestionLevel.SEVERE
    assert results[1].estimated_delay_minutes == 6.75  # tt=9.0 min, ff=2.25 min -> delay=6.75 min

    assert results[2].congestion_level == CongestionLevel.SEVERE
    assert results[2].congestion_status == CongestionStatus.STOPPED
    assert results[2].estimated_travel_time_minutes is None


def test_road_network_edge_adapter_integration():
    """Verify that P1 RoadNetworkGraph edge dictionaries are correctly converted and evaluated."""
    engine = CongestionIntelligenceEngine()

    mock_edge = {
        "edge_id": "1001_1002_0",
        "u": 1001,
        "v": 1002,
        "key": 0,
        "length_m": 1250.0,  # 1.25 km
        "free_flow_speed_kmh": 50.0,
        "highway_type": "primary",
        "name": "Avinashi Road",
        "road_reference": "NH-544",
        "speed_source": "OSM_MAXSPEED",
    }

    result = engine.evaluate_road_network_edge(
        edge_data=mock_edge,
        predicted_speed_kmh=35.0,  # ratio = 35 / 50 = 0.70 -> MODERATE
        forecast_horizon_minutes=15,
    )

    assert result.segment_id == "1001_1002_0"
    assert result.segment_length_km == 1.25
    assert result.free_flow_speed_kmh == 50.0
    assert result.speed_ratio == 0.70
    assert result.congestion_level == CongestionLevel.MODERATE
    assert result.source_geography == "Tamil Nadu (Coimbatore-Avinashi Road)"
    assert result.metadata["highway_type"] == "primary"
    assert result.metadata["road_name"] == "Avinashi Road"
    assert result.metadata["road_reference"] == "NH-544"


def test_geographic_and_dataset_integrity_isolation():
    """Verify that METR-LA predictions and Tamil Nadu road graph contexts maintain explicit geographic isolation."""
    engine = CongestionIntelligenceEngine()

    # Evaluation for METR-LA loop detector sensor
    metr_la_res = engine.evaluate_segment(
        segment_id="metr_la_sensor_767541",
        segment_length_km=1.609,  # 1 mile
        free_flow_speed_kmh=104.6,  # 65 mph
        predicted_speed_kmh=75.0,
        forecast_horizon_minutes=30,
        model_source="DCRNN_P4_1_BEST_EPOCH19",
        source_geography="Los Angeles County, CA (METR-LA Freeway Loop Detectors)",
    )
    assert "Los Angeles County, CA" in metr_la_res.source_geography
    assert "Tamil Nadu" not in metr_la_res.source_geography

    # Evaluation for Coimbatore road network edge
    coimbatore_res = engine.evaluate_segment(
        segment_id="coimbatore_osm_edge_501",
        segment_length_km=0.85,
        free_flow_speed_kmh=40.0,
        predicted_speed_kmh=20.0,
        forecast_horizon_minutes=15,
        model_source="HISTORICAL_COIMBATORE_PROFILE",
        source_geography="Tamil Nadu (Coimbatore-Avinashi Road)",
    )
    assert "Tamil Nadu" in coimbatore_res.source_geography
    assert "METR-LA" not in coimbatore_res.source_geography


def test_invalid_segment_length_does_not_fabricate_dimensions():
    """Verify that invalid segment length results in None length and travel times without fabricating 1.0 km."""
    engine = CongestionIntelligenceEngine()

    for invalid_length in [0.0, -5.0, float("nan"), float("inf"), None]:
        res = engine.evaluate_segment(
            segment_id="bad_length_segment",
            segment_length_km=invalid_length,
            free_flow_speed_kmh=50.0,
            predicted_speed_kmh=25.0,
        )

        assert res.is_valid is False
        assert res.congestion_status == CongestionStatus.INVALID_SEGMENT_LENGTH
        assert res.segment_length_km is None  # Never fabricated as 1.0 km
        assert res.free_flow_travel_time_minutes is None  # Never fabricated as 1.0 min
        assert res.estimated_travel_time_minutes is None
        assert res.estimated_delay_minutes is None
        assert res.free_flow_speed_kmh == 50.0  # Genuine input preserved
        assert res.predicted_speed_kmh == 25.0  # Genuine input preserved


def test_invalid_free_flow_speed_does_not_fabricate_dimensions():
    """Verify that invalid free-flow speed results in None speed and travel times without fabricating 30.0 km/h."""
    engine = CongestionIntelligenceEngine()

    for invalid_ff in [0.0, -60.0, float("nan"), float("inf"), None]:
        res = engine.evaluate_segment(
            segment_id="bad_ff_segment",
            segment_length_km=2.0,
            free_flow_speed_kmh=invalid_ff,
            predicted_speed_kmh=30.0,
        )

        assert res.is_valid is False
        assert res.congestion_status == CongestionStatus.INVALID_FREE_FLOW_SPEED
        assert res.free_flow_speed_kmh is None  # Never fabricated as 30.0 km/h
        assert res.free_flow_travel_time_minutes is None  # Never fabricated as 1.0 min
        assert res.speed_ratio is None
        assert res.congestion_level == CongestionLevel.UNKNOWN
        assert res.estimated_travel_time_minutes is None
        assert res.estimated_delay_minutes is None
        assert res.segment_length_km == 2.0  # Genuine input preserved


def test_both_dimensions_invalid_does_not_fabricate_dimensions():
    """Verify that when both dimensions are invalid, all dimensions and calculations remain None."""
    engine = CongestionIntelligenceEngine()

    res = engine.evaluate_segment(
        segment_id="both_bad_segment",
        segment_length_km=0.0,
        free_flow_speed_kmh=0.0,
        predicted_speed_kmh=15.0,
    )

    assert res.is_valid is False
    assert res.segment_length_km is None
    assert res.free_flow_speed_kmh is None
    assert res.free_flow_travel_time_minutes is None
    assert res.estimated_travel_time_minutes is None
    assert res.estimated_delay_minutes is None
    assert res.speed_ratio is None
    assert res.congestion_level == CongestionLevel.UNKNOWN


def test_road_network_edge_adapter_missing_dimensions_fails_safely():
    """Verify that P1 edge adapter does not fabricate dimensions when edge attributes are missing or zero."""
    engine = CongestionIntelligenceEngine()

    empty_edge = {
        "edge_id": "empty_edge_999",
        # Neither length_m nor length_km, and no free_flow_speed_kmh
    }

    res = engine.evaluate_road_network_edge(
        edge_data=empty_edge,
        predicted_speed_kmh=20.0,
    )

    assert res.is_valid is False
    assert res.segment_length_km is None
    assert res.free_flow_speed_kmh is None
    assert res.free_flow_travel_time_minutes is None
    assert res.estimated_travel_time_minutes is None
    assert res.estimated_delay_minutes is None

