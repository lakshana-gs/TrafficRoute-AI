"""Unit tests for P7.1: Forward Spatio-Temporal Time Propagation and Speed Providers."""

from datetime import datetime, timedelta, timezone
import math
import networkx as nx
import pytest

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, Route, RouteSegment, RoutingEngine
from src.network.snapping import SnapResult
from src.traffic.congestion import CongestionLevel, CongestionStatus
from src.traffic.predictive_evaluation import (
    BaseSpeedProvider,
    EvaluatedRoute,
    EvaluatedRouteSegment,
    ForecastHorizonStatus,
    HeuristicDiurnalSpeedProvider,
    IST_OFFSET,
    MockSpeedProvider,
    SpeedQueryResult,
    SpeedProvenanceType,
    StaticFreeFlowSpeedProvider,
    forward_propagate_route,
)


@pytest.fixture
def synthetic_linear_network():
    """
    Synthetic 3-segment linear corridor: 1 -> 2 -> 3 -> 4
    Edge 1->2: Highway trunk (length: 1000m, free_flow_speed: 80 km/h)
    Edge 2->3: Urban primary (length: 600m, free_flow_speed: 60 km/h)
    Edge 3->4: Residential (length: 400m, free_flow_speed: 30 km/h)
    Total distance = 2000m
    """
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.01, longitude=77.01)
    g.add_node(3, latitude=11.02, longitude=77.02)
    g.add_node(4, latitude=11.03, longitude=77.03)

    g.add_edge(1, 2, key=0, edge_id="1_2_0", highway="trunk", name="NH-544 Bypass", road_reference="NH-544", length_m=1000.0, free_flow_speed_kmh=80.0)
    g.add_edge(2, 3, key=0, edge_id="2_3_0", highway="primary", name="Avinashi Road", road_reference=None, length_m=600.0, free_flow_speed_kmh=60.0)
    g.add_edge(3, 4, key=0, edge_id="3_4_0", highway="residential", name="Local Street", road_reference=None, length_m=400.0, free_flow_speed_kmh=30.0)

    road_graph = RoadNetworkGraph(graph=g)
    engine = RoutingEngine(road_graph)
    route = engine.calculate_route_by_nodes(1, 4, metric=CostMetric.FREE_FLOW_TIME)
    assert route is not None
    assert len(route.edge_sequence) == 3
    return road_graph, route


@pytest.fixture
def single_node_route():
    """Synthetic single-node, zero-distance route (origin == destination)."""
    snap = SnapResult(
        input_latitude=11.00,
        input_longitude=77.00,
        snapped_latitude=11.00,
        snapped_longitude=77.00,
        nearest_node_id=1,
        distance_meters=0.0,
        is_valid=True,
    )
    return Route(
        route_id="single_node_test",
        name="Stationary Route",
        origin_snap=snap,
        destination_snap=snap,
        node_sequence=[1],
        edge_sequence=[],
        total_distance_m=0.0,
        total_time_s=0.0,
        cost_metric=CostMetric.FREE_FLOW_TIME,
    )


# =====================================================================
# 1. Monotonic Timestamp Advancement Across Multi-Segment Route
# =====================================================================

def test_strict_timestamp_monotonicity(synthetic_linear_network):
    """Verify arrival times advance strictly monotonically by traversal duration."""
    road_graph, route = synthetic_linear_network
    dep_time = datetime(2026, 10, 21, 8, 30, tzinfo=timezone.utc)

    # Mock provider: constant 50 km/h across all edges
    provider = MockSpeedProvider(default_speed_kmh=50.0)
    eval_route = forward_propagate_route(route, dep_time, provider, road_graph=road_graph)

    assert isinstance(eval_route, EvaluatedRoute)
    assert eval_route.departure_time == dep_time
    assert len(eval_route.evaluated_segments) == 3

    speed_mps = 50.0 * (1000.0 / 3600.0)  # ~13.8889 m/s

    curr_expected_time = dep_time
    for i, eval_seg in enumerate(eval_route.evaluated_segments):
        # Arrival timestamp must strictly equal accumulated time
        assert eval_seg.arrival_time == curr_expected_time

        # Traversal duration must match length / speed
        expected_duration_s = eval_seg.segment.length_m / speed_mps
        assert eval_seg.traversal_time_seconds == pytest.approx(expected_duration_s, abs=0.01)

        # Subsequent segment arrival time must be strictly greater
        next_time = eval_seg.arrival_time + timedelta(seconds=eval_seg.traversal_time_seconds)
        assert next_time > eval_seg.arrival_time

        curr_expected_time = next_time

    # Final route arrival time must match cumulative step completion
    assert eval_route.estimated_arrival_time == curr_expected_time
    assert eval_route.predicted_travel_time_seconds == pytest.approx(
        sum(s.traversal_time_seconds for s in eval_route.evaluated_segments), abs=0.01
    )


# =====================================================================
# 2. Timezone Validation and Handling
# =====================================================================

def test_timezone_validation_and_handling(synthetic_linear_network):
    """Verify naive departure timestamps are rejected; UTC and IST are preserved."""
    road_graph, route = synthetic_linear_network
    provider = StaticFreeFlowSpeedProvider()

    # 1. Timezone-naive datetime must raise ValueError
    naive_time = datetime(2026, 10, 21, 8, 30)
    with pytest.raises(ValueError, match="timezone-aware"):
        forward_propagate_route(route, naive_time, provider)

    # 2. Explicit UTC departure time
    utc_time = datetime(2026, 10, 21, 8, 30, tzinfo=timezone.utc)
    eval_utc = forward_propagate_route(route, utc_time, provider)
    assert eval_utc.departure_time.tzinfo == timezone.utc
    assert eval_utc.estimated_arrival_time.tzinfo == timezone.utc
    for seg in eval_utc.evaluated_segments:
        assert seg.arrival_time.tzinfo == timezone.utc

    # 3. Explicit Indian Standard Time (IST, UTC+05:30) departure time
    ist_time = datetime(2026, 10, 21, 14, 0, tzinfo=IST_OFFSET)
    eval_ist = forward_propagate_route(route, ist_time, provider)
    assert eval_ist.departure_time.tzinfo == IST_OFFSET
    assert eval_ist.estimated_arrival_time.tzinfo == IST_OFFSET
    for seg in eval_ist.evaluated_segments:
        assert seg.arrival_time.tzinfo == IST_OFFSET


# =====================================================================
# 3. Single-Node Route Invariant (SAME_LOCATION)
# =====================================================================

def test_single_node_same_location_route(single_node_route):
    """Verify a single-node route returns zero distance, time, and delay without errors."""
    dep_time = datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)
    provider = HeuristicDiurnalSpeedProvider()

    eval_route = forward_propagate_route(single_node_route, dep_time, provider)

    assert eval_route.total_distance_m == 0.0
    assert eval_route.total_distance_km == 0.0
    assert eval_route.free_flow_time_seconds == 0.0
    assert eval_route.predicted_travel_time_seconds == 0.0
    assert eval_route.predicted_travel_time_minutes == 0.0
    assert eval_route.total_delay_seconds == 0.0
    assert eval_route.total_delay_minutes == 0.0
    assert eval_route.average_speed_kmh == 0.0
    assert eval_route.estimated_arrival_time == dep_time
    assert len(eval_route.evaluated_segments) == 0
    assert eval_route.route_congestion_level == CongestionLevel.LOW


# =====================================================================
# 4. Heuristic Diurnal Speed Variations (Peak vs Off-Peak)
# =====================================================================

def test_heuristic_diurnal_speed_variations(synthetic_linear_network):
    """Verify HeuristicDiurnalSpeedProvider reflects diurnal variation across IST hours."""
    road_graph, route = synthetic_linear_network
    provider = HeuristicDiurnalSpeedProvider()

    # Morning Peak: 08:30 AM IST (03:00 UTC)
    morning_peak_ist = datetime(2026, 10, 21, 8, 30, tzinfo=IST_OFFSET)
    eval_morning = forward_propagate_route(route, morning_peak_ist, provider, road_graph=road_graph)

    # Midday Inter-Peak: 14:00 PM IST (08:30 UTC)
    midday_ist = datetime(2026, 10, 21, 14, 0, tzinfo=IST_OFFSET)
    eval_midday = forward_propagate_route(route, midday_ist, provider, road_graph=road_graph)

    # Midnight Off-Peak: 02:00 AM IST (20:30 previous day UTC)
    night_ist = datetime(2026, 10, 21, 2, 0, tzinfo=IST_OFFSET)
    eval_night = forward_propagate_route(route, night_ist, provider, road_graph=road_graph)

    # 1. Travel time ordering: Morning peak should be slower than midday, which is slower than night
    assert eval_morning.predicted_travel_time_seconds > eval_midday.predicted_travel_time_seconds
    assert eval_midday.predicted_travel_time_seconds > eval_night.predicted_travel_time_seconds

    # 2. Delay ordering: Morning peak delay > Midday delay > Night delay
    assert eval_morning.total_delay_seconds > eval_midday.total_delay_seconds
    assert eval_midday.total_delay_seconds > eval_night.total_delay_seconds

    # 3. Night should be near zero delay (free-flow)
    assert eval_night.total_delay_seconds == pytest.approx(0.0, abs=5.0)

    # 4. Verify explicit provenance classification (strictly not claiming empirical calibration)
    for seg in eval_morning.evaluated_segments:
        assert seg.speed_result.provenance_type == SpeedProvenanceType.ASSUMPTION_BASED_DIURNAL
        assert seg.speed_result.confidence_tier == "HEURISTIC_ESTIMATE"
        assert seg.speed_result.model_source == "HEURISTIC_DIURNAL"
        assert seg.speed_result.is_fallback is False


# =====================================================================
# 5. Fallback Provenance on Missing or Invalid Speeds
# =====================================================================

def test_fallback_provenance_on_missing_or_invalid_speeds(synthetic_linear_network):
    """Verify StaticFreeFlowSpeedProvider outputs explicit fallback provenance."""
    road_graph, route = synthetic_linear_network
    dep_time = datetime(2026, 10, 21, 10, 0, tzinfo=timezone.utc)
    provider = StaticFreeFlowSpeedProvider(default_speed_kmh=35.0)

    eval_route = forward_propagate_route(route, dep_time, provider, road_graph=road_graph)

    # Static free-flow produces zero signed delay
    assert eval_route.total_delay_seconds == pytest.approx(0.0, abs=0.01)
    assert eval_route.predicted_travel_time_seconds == pytest.approx(eval_route.free_flow_time_seconds, abs=0.01)

    # Provenance must be explicitly flagged as fallback
    for seg in eval_route.evaluated_segments:
        assert seg.speed_result.provenance_type == SpeedProvenanceType.FREE_FLOW_FALLBACK
        assert seg.speed_result.confidence_tier == "BASELINE_UNMONITORED"
        assert seg.speed_result.is_fallback is True
        assert seg.speed_result.forecast_horizon_status == ForecastHorizonStatus.NOT_APPLICABLE


# =====================================================================
# 6. Zero Speed (Stopped Traffic) Crawl Floor Guard
# =====================================================================

def test_zero_speed_stopped_traffic_crawl_floor(synthetic_linear_network):
    """Verify stopped traffic (0.0 km/h) applies crawl floor without zero division and sets STOPPED status."""
    road_graph, route = synthetic_linear_network
    dep_time = datetime(2026, 10, 21, 8, 30, tzinfo=timezone.utc)

    # Edge 2->3 experiences complete standstill (0.0 km/h)
    edge_speeds = {
        (1, 2, 0): 60.0,
        (2, 3, 0): 0.0,  # STOPPED
        (3, 4, 0): 30.0,
    }
    provider = MockSpeedProvider(edge_speeds=edge_speeds)

    # Run with standard crawl floor = 3.0 km/h
    eval_route = forward_propagate_route(route, dep_time, provider, road_graph=road_graph, crawl_speed_kmh=3.0)

    assert isinstance(eval_route, EvaluatedRoute)
    assert math.isfinite(eval_route.predicted_travel_time_seconds)

    # Segment 2->3 (index 1) checks
    stopped_seg = eval_route.evaluated_segments[1]
    assert stopped_seg.speed_result.speed_kmh == 0.0
    assert stopped_seg.congestion_result.congestion_status == CongestionStatus.STOPPED
    assert stopped_seg.congestion_result.congestion_level == CongestionLevel.SEVERE

    # Traversal time on 600m segment at 3.0 km/h (0.8333 m/s) should be ~720 seconds (12 mins)
    expected_crawl_time_s = 600.0 / (3.0 * 1000.0 / 3600.0)  # 720s
    assert stopped_seg.traversal_time_seconds == pytest.approx(expected_crawl_time_s, abs=1.0)

    # Substantial delay should be recorded
    assert stopped_seg.signed_delay_seconds > 600.0
    assert eval_route.total_delay_seconds > 600.0


# =====================================================================
# 7. Non-Mutation Guarantee
# =====================================================================

def test_non_mutation_of_route_and_graph(synthetic_linear_network):
    """Verify forward_propagate_route does not mutate the input Route or RoadNetworkGraph."""
    road_graph, route = synthetic_linear_network
    dep_time = datetime(2026, 10, 21, 8, 30, tzinfo=timezone.utc)

    # Capture pre-propagation state
    orig_total_dist = route.total_distance_m
    orig_total_time = route.total_time_s
    orig_seg_speeds = [seg.speed_kmh for seg in route.edge_sequence]
    orig_graph_edge_count = road_graph.graph.number_of_edges()

    provider = HeuristicDiurnalSpeedProvider()
    forward_propagate_route(route, dep_time, provider, road_graph=road_graph)

    # Invariants on input Route must be unaltered
    assert route.total_distance_m == orig_total_dist
    assert route.total_time_s == orig_total_time
    assert [seg.speed_kmh for seg in route.edge_sequence] == orig_seg_speeds
    assert road_graph.graph.number_of_edges() == orig_graph_edge_count
