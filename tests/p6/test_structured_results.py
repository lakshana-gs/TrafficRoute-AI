"""Unit tests for P6.3: Structured Route Results."""

from datetime import datetime, timezone
import json
import math
import networkx as nx
import pytest
from pydantic import ValidationError

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, Route, RouteSegment, RoutingEngine
from src.network.snapping import SnapResult


@pytest.fixture
def sample_snap():
    """Deterministic valid SnapResult."""
    return SnapResult(
        input_latitude=11.00,
        input_longitude=77.00,
        snapped_latitude=11.00,
        snapped_longitude=77.00,
        nearest_node_id=1,
        distance_meters=0.0,
        is_valid=True,
    )


@pytest.fixture
def sample_segment():
    """Deterministic valid RouteSegment."""
    return RouteSegment(
        u=1,
        v=2,
        key=0,
        edge_id="1_2_0",
        name="Avinashi Road",
        road_reference="NH-544",
        highway_type="trunk",
        length_m=1000.0,
        speed_kmh=60.0,
        traversal_time_s=60.0,
    )


@pytest.fixture
def triangle_network():
    """Deterministic valid 3-node road network graph."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_node(3, latitude=11.00, longitude=77.02)

    g.add_edge(1, 2, key=0, highway="trunk", name="Expressway", ref="NH-544", length_m=2000.0, free_flow_speed_kmh=80.0)
    g.add_edge(1, 3, key=0, highway="residential", name="Local 1", length_m=600.0, free_flow_speed_kmh=25.0)
    g.add_edge(3, 2, key=0, highway="residential", name="Local 2", length_m=600.0, free_flow_speed_kmh=25.0)

    return RoadNetworkGraph(graph=g)


# =====================================================================
# 1. RouteSegment Validation Tests
# =====================================================================

def test_route_segment_valid_construction(sample_segment):
    """Verify that a valid RouteSegment constructs successfully with expected fields."""
    assert sample_segment.u == 1
    assert sample_segment.v == 2
    assert sample_segment.length_m == 1000.0
    assert sample_segment.speed_kmh == 60.0
    assert sample_segment.traversal_time_s == 60.0


def test_route_segment_rejects_negative_or_zero_length():
    """Verify that length_m <= 0 is strictly rejected by RouteSegment."""
    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=0.0, speed_kmh=50.0, traversal_time_s=10.0,
        )

    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=-50.0, speed_kmh=50.0, traversal_time_s=10.0,
        )


def test_route_segment_rejects_negative_or_zero_speed():
    """Verify that speed_kmh <= 0 is strictly rejected by RouteSegment."""
    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=100.0, speed_kmh=0.0, traversal_time_s=10.0,
        )

    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=100.0, speed_kmh=-30.0, traversal_time_s=10.0,
        )


def test_route_segment_rejects_negative_traversal_time():
    """Verify that traversal_time_s < 0 is strictly rejected, while 0.0 is permitted."""
    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=100.0, speed_kmh=50.0, traversal_time_s=-1.0,
        )

    # 0.0 is non-negative and valid
    seg = RouteSegment(
        u=1, v=2, edge_id="1_2_0", highway_type="trunk",
        length_m=100.0, speed_kmh=50.0, traversal_time_s=0.0,
    )
    assert seg.traversal_time_s == 0.0


def test_route_segment_rejects_non_finite_and_boolean_dimensions():
    """Verify that NaN, infinite, and boolean dimensions raise ValidationError."""
    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=float("nan"), speed_kmh=50.0, traversal_time_s=10.0,
        )

    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=100.0, speed_kmh=float("inf"), traversal_time_s=10.0,
        )

    with pytest.raises(ValidationError):
        RouteSegment(
            u=1, v=2, edge_id="1_2_0", highway_type="trunk",
            length_m=True, speed_kmh=50.0, traversal_time_s=10.0,
        )


# =====================================================================
# 2. Route Validation & Invariant Tests
# =====================================================================

def test_route_valid_construction(sample_snap, sample_segment):
    """Verify that a valid Route constructs successfully with all required fields."""
    route = Route(
        route_id="r1",
        name="Main Route",
        origin_snap=sample_snap,
        destination_snap=sample_snap,
        node_sequence=[1, 2],
        edge_sequence=[sample_segment],
        total_distance_m=1000.0,
        total_time_s=60.0,
        cost_metric=CostMetric.FREE_FLOW_TIME,
    )
    assert route.route_id == "r1"
    assert route.total_distance_km == 1.0
    assert route.total_time_minutes == 1.0
    assert route.segment_count == 1
    assert route.average_speed_kmh == 60.0


def test_route_rejects_negative_or_non_finite_totals(sample_snap, sample_segment):
    """Verify that negative or non-finite total_distance_m / total_time_s are rejected."""
    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1, 2], edge_sequence=[sample_segment],
            total_distance_m=-1000.0, total_time_s=60.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )

    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1, 2], edge_sequence=[sample_segment],
            total_distance_m=1000.0, total_time_s=float("nan"),
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )


def test_route_rejects_empty_node_sequence(sample_snap):
    """Verify that an empty node_sequence is strictly rejected."""
    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[], edge_sequence=[],
            total_distance_m=0.0, total_time_s=0.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )


def test_route_rejects_sequence_length_mismatch(sample_snap, sample_segment):
    """Verify that len(edge_sequence) must equal len(node_sequence) - 1."""
    # 2 nodes but 0 edges
    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1, 2], edge_sequence=[],
            total_distance_m=0.0, total_time_s=0.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )

    # 2 nodes but 2 edges
    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1, 2], edge_sequence=[sample_segment, sample_segment],
            total_distance_m=2000.0, total_time_s=120.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )


def test_route_rejects_topological_disconnection(sample_snap, sample_segment):
    """Verify that segments must connect consecutive nodes in node_sequence."""
    # Segment connects 1 -> 2, but node sequence is 1 -> 3
    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1, 3], edge_sequence=[sample_segment],
            total_distance_m=1000.0, total_time_s=60.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )


def test_route_rejects_totals_deviating_from_segments(sample_snap, sample_segment):
    """Verify that route total distance and time must match sum of segments within tolerance."""
    # Segment has length 1000.0, but total is declared as 5000.0
    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1, 2], edge_sequence=[sample_segment],
            total_distance_m=5000.0, total_time_s=60.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )

    # Segment has time 60.0, but total is declared as 300.0
    with pytest.raises(ValidationError):
        Route(
            route_id="r1",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1, 2], edge_sequence=[sample_segment],
            total_distance_m=1000.0, total_time_s=300.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )


def test_single_node_zero_distance_route_preserved(sample_snap):
    """Verify that legitimate single-node routes with zero distance and time are accepted."""
    route = Route(
        route_id="r0",
        origin_snap=sample_snap,
        destination_snap=sample_snap,
        node_sequence=[1],
        edge_sequence=[],
        total_distance_m=0.0,
        total_time_s=0.0,
        cost_metric=CostMetric.FREE_FLOW_TIME,
    )
    assert route.segment_count == 0
    assert route.total_distance_km == 0.0
    assert route.total_time_minutes == 0.0
    assert route.average_speed_kmh == 0.0

    # Non-zero distance on single-node route must be rejected
    with pytest.raises(ValidationError):
        Route(
            route_id="r0",
            origin_snap=sample_snap, destination_snap=sample_snap,
            node_sequence=[1], edge_sequence=[],
            total_distance_m=100.0, total_time_s=0.0,
            cost_metric=CostMetric.FREE_FLOW_TIME,
        )


# =====================================================================
# 3. Summary & KPI Property Tests
# =====================================================================

def test_route_properties_and_speed_calculation(sample_snap):
    """Verify average_speed_kmh calculation across typical multi-segment route."""
    # 2 segments: 10,000m total in 600s (10 km in 10 minutes -> 60.0 km/h)
    seg1 = RouteSegment(
        u=1, v=2, edge_id="1_2_0", highway_type="trunk",
        length_m=6000.0, speed_kmh=60.0, traversal_time_s=360.0,
    )
    seg2 = RouteSegment(
        u=2, v=3, edge_id="2_3_0", highway_type="trunk",
        length_m=4000.0, speed_kmh=60.0, traversal_time_s=240.0,
    )
    route = Route(
        route_id="primary_route",
        name="Via NH-544",
        origin_snap=sample_snap,
        destination_snap=sample_snap,
        node_sequence=[1, 2, 3],
        edge_sequence=[seg1, seg2],
        total_distance_m=10000.0,
        total_time_s=600.0,
        cost_metric=CostMetric.FREE_FLOW_TIME,
    )

    assert route.segment_count == 2
    assert route.total_distance_km == 10.0
    assert route.total_time_minutes == 10.0
    # 10 km / (600 / 3600 h) = 60.0 km/h
    assert route.average_speed_kmh == 60.0


def test_route_summary_structure_and_json_serialization(sample_snap, sample_segment):
    """Verify route.summary() returns expected dictionary and is strictly JSON serializable."""
    dep_time = datetime(2026, 10, 10, 8, 30, tzinfo=timezone.utc)
    route = Route(
        route_id="route_101",
        name="Fastest Route",
        origin_snap=sample_snap,
        destination_snap=sample_snap,
        node_sequence=[1, 2],
        edge_sequence=[sample_segment],
        total_distance_m=1000.0,
        total_time_s=60.0,
        cost_metric=CostMetric.FREE_FLOW_TIME,
        departure_time=dep_time,
    )

    summary = route.summary()

    assert summary["route_id"] == "route_101"
    assert summary["name"] == "Fastest Route"
    assert summary["cost_metric"] == "free_flow_time"
    assert summary["total_distance_m"] == 1000.0
    assert summary["total_distance_km"] == 1.0
    assert summary["total_time_s"] == 60.0
    assert summary["total_time_minutes"] == 1.0
    assert summary["average_speed_kmh"] == 60.0
    assert summary["segment_count"] == 1
    assert summary["node_count"] == 2
    assert summary["departure_time"] == "2026-10-10T08:30:00+00:00"

    # Strictly JSON-serializable without custom default serializer
    serialized = json.dumps(summary)
    parsed = json.loads(serialized)
    assert parsed["route_id"] == "route_101"
    assert parsed["cost_metric"] == "free_flow_time"


def test_route_summary_with_null_departure_time(sample_snap, sample_segment):
    """Verify route.summary() handles departure_time=None cleanly."""
    route = Route(
        route_id="route_none_dep",
        origin_snap=sample_snap,
        destination_snap=sample_snap,
        node_sequence=[1, 2],
        edge_sequence=[sample_segment],
        total_distance_m=1000.0,
        total_time_s=60.0,
        cost_metric=CostMetric.DISTANCE,
        departure_time=None,
    )

    summary = route.summary()
    assert summary["departure_time"] is None
    assert summary["cost_metric"] == "distance"
    assert json.dumps(summary) is not None


# =====================================================================
# 4. Node Snap Validation & RoutingEngine Integration
# =====================================================================

def test_node_snap_validation_with_invalid_node_coordinates(triangle_network):
    """Verify that calculate_route_by_nodes returns None when node coordinates are invalid or non-finite."""
    engine = RoutingEngine(triangle_network)
    # Corrupt coordinates on node 1 after spatial indexer build
    triangle_network.graph.nodes[1]["latitude"] = float("nan")

    assert engine.calculate_route_by_nodes(1, 2) is None
    # Same node with corrupt coordinates must also return None
    assert engine.calculate_route_by_nodes(1, 1) is None


def test_node_snap_validation_out_of_bounds(triangle_network):
    """Verify that node with out-of-bounds coordinates fails snap and returns None."""
    engine = RoutingEngine(triangle_network)
    triangle_network.graph.nodes[2]["latitude"] = 999.0

    assert engine.calculate_route_by_nodes(1, 2) is None


def test_routing_engine_end_to_end_produces_structured_route(triangle_network):
    """Verify end-to-end routing produces fully valid structured Route matching all invariants."""
    engine = RoutingEngine(triangle_network)

    route = engine.calculate_route_by_nodes(1, 2, metric=CostMetric.FREE_FLOW_TIME)
    assert route is not None
    assert isinstance(route, Route)
    assert route.segment_count == 1
    assert route.node_sequence == [1, 2]
    assert len(route.edge_sequence) == 1
    assert route.total_distance_m == 2000.0
    # 2000m / (80 * 1000/3600) = 90.0s -> 80.0 km/h
    assert route.average_speed_kmh == 80.0

    summary = route.summary()
    assert summary["segment_count"] == 1
    assert summary["average_speed_kmh"] == 80.0
    assert json.dumps(summary) is not None
