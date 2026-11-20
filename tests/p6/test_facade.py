"""Unit tests for P6.5: Unified Route Planning Facade and RoutePlanResult."""

from datetime import datetime, timezone
import json
import math
import networkx as nx
import pytest

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, RoutePlanResult, RouteStatus, RoutingEngine


@pytest.fixture
def multi_corridor_graph():
    """
    Synthetic graph with 3 distinct corridors between Node 1 and Node 4:
    - Corridor 1 (Highway): 1 -> 2 -> 4 (length: 1000m + 1000m = 2000m, speed: 80 km/h)
    - Corridor 2 (Arterial): 1 -> 3 -> 4 (length: 1200m + 1200m = 2400m, speed: 60 km/h)
    - Corridor 3 (Scenic): 1 -> 5 -> 4 (length: 1500m + 1500m = 3000m, speed: 50 km/h)
    """
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_node(3, latitude=10.98, longitude=77.02)
    g.add_node(4, latitude=11.00, longitude=77.04)
    g.add_node(5, latitude=11.04, longitude=77.02)

    # Corridor 1
    g.add_edge(1, 2, key=0, highway="trunk", name="NH-544", road_reference="NH-544", length_m=1000.0, free_flow_speed_kmh=80.0)
    g.add_edge(2, 4, key=0, highway="trunk", name="NH-544", road_reference="NH-544", length_m=1000.0, free_flow_speed_kmh=80.0)

    # Corridor 2
    g.add_edge(1, 3, key=0, highway="primary", name="Avinashi Bypass", length_m=1200.0, free_flow_speed_kmh=60.0)
    g.add_edge(3, 4, key=0, highway="primary", name="Avinashi Bypass", length_m=1200.0, free_flow_speed_kmh=60.0)

    # Corridor 3
    g.add_edge(1, 5, key=0, highway="secondary", name="Ring Road", length_m=1500.0, free_flow_speed_kmh=50.0)
    g.add_edge(5, 4, key=0, highway="secondary", name="Ring Road", length_m=1500.0, free_flow_speed_kmh=50.0)

    return RoadNetworkGraph(graph=g)


# =====================================================================
# 1. Schema & Serialization Tests
# =====================================================================

def test_route_status_enum_values():
    """Verify RouteStatus members and string values."""
    assert RouteStatus.SUCCESS.value == "success"
    assert RouteStatus.SAME_LOCATION.value == "same_location"
    assert RouteStatus.SNAP_FAILED.value == "snap_failed"
    assert RouteStatus.NO_PATH_FOUND.value == "no_path_found"
    assert RouteStatus.INVALID_INPUT.value == "invalid_input"


def test_route_plan_result_serialization(multi_corridor_graph):
    """Verify RoutePlanResult serializes cleanly to JSON via Pydantic model_dump_json and model_dump."""
    engine = RoutingEngine(multi_corridor_graph)
    result = engine.plan_route_by_nodes(1, 4, num_alternatives=1)

    assert result.success is True
    assert result.status == RouteStatus.SUCCESS
    assert result.summary is not None
    assert result.primary_route is not None

    # Serialization test via model_dump(mode="json")
    dumped = result.model_dump(mode="json")
    assert dumped["status"] == "success"
    assert dumped["success"] is True
    assert isinstance(dumped["summary"], dict)

    # Strictly JSON serializable via standard json library
    json_str = json.dumps(dumped)
    parsed = json.loads(json_str)
    assert parsed["status"] == "success"

    # Pydantic native json string
    pydantic_json = result.model_dump_json()
    assert '"status":"success"' in pydantic_json


# =====================================================================
# 2. Unified Facade Success & Alternatives Tests
# =====================================================================

def test_plan_route_primary_only_mode(multi_corridor_graph):
    """Verify default num_alternatives=0 returns primary route with empty alternative_routes."""
    engine = RoutingEngine(multi_corridor_graph)

    # Coordinates
    coord_res = engine.plan_route(11.00, 77.00, 11.00, 77.04, num_alternatives=0)
    assert coord_res.success is True
    assert coord_res.status == RouteStatus.SUCCESS
    assert coord_res.primary_route is not None
    assert coord_res.primary_route.node_sequence == [1, 2, 4]
    assert coord_res.alternative_routes == []

    # Nodes
    node_res = engine.plan_route_by_nodes(1, 4, num_alternatives=0)
    assert node_res.success is True
    assert node_res.status == RouteStatus.SUCCESS
    assert node_res.primary_route is not None
    assert node_res.primary_route.node_sequence == [1, 2, 4]
    assert node_res.alternative_routes == []


def test_plan_route_with_alternatives_no_duplication(multi_corridor_graph):
    """Verify num_alternatives > 0 returns primary route and separate diverse alternatives without duplication."""
    engine = RoutingEngine(multi_corridor_graph)

    result = engine.plan_route_by_nodes(1, 4, num_alternatives=2)
    assert result.success is True
    assert result.status == RouteStatus.SUCCESS
    assert result.primary_route is not None
    assert result.primary_route.node_sequence == [1, 2, 4]

    assert len(result.alternative_routes) == 2
    # Ensure primary route is not duplicated in alternative_routes
    for alt in result.alternative_routes:
        assert alt.node_sequence != result.primary_route.node_sequence

    # Alternatives match corridors 2 and 3
    assert result.alternative_routes[0].node_sequence == [1, 3, 4]
    assert result.alternative_routes[1].node_sequence == [1, 5, 4]


def test_plan_route_with_distance_metric(multi_corridor_graph):
    """Verify plan_route supports CostMetric.DISTANCE cleanly."""
    engine = RoutingEngine(multi_corridor_graph)

    res = engine.plan_route_by_nodes(1, 4, metric=CostMetric.DISTANCE, num_alternatives=1)
    assert res.success is True
    assert res.status == RouteStatus.SUCCESS
    assert res.primary_route.cost_metric == CostMetric.DISTANCE


# =====================================================================
# 3. Diagnostic Semantics: INVALID_INPUT Tests
# =====================================================================

def test_plan_route_invalid_coordinates(multi_corridor_graph):
    """Verify non-finite, out-of-bounds, or boolean coordinates return INVALID_INPUT."""
    engine = RoutingEngine(multi_corridor_graph)

    # NaN
    res1 = engine.plan_route(float("nan"), 77.00, 11.00, 77.04)
    assert res1.success is False
    assert res1.status == RouteStatus.INVALID_INPUT
    assert res1.primary_route is None

    # Out of bounds WGS84
    res2 = engine.plan_route(11.00, 77.00, 999.0, 77.04)
    assert res2.success is False
    assert res2.status == RouteStatus.INVALID_INPUT

    # Boolean coordinates
    res3 = engine.plan_route(True, 77.00, 11.00, 77.04)
    assert res3.success is False
    assert res3.status == RouteStatus.INVALID_INPUT


def test_plan_route_invalid_node_ids_and_missing_nodes(multi_corridor_graph):
    """Verify non-integer node IDs, booleans, and non-existent nodes return INVALID_INPUT."""
    engine = RoutingEngine(multi_corridor_graph)

    # Boolean node IDs
    res1 = engine.plan_route_by_nodes(True, 4)
    assert res1.success is False
    assert res1.status == RouteStatus.INVALID_INPUT

    # Non-integer node IDs
    res2 = engine.plan_route_by_nodes("1", 4)
    assert res2.success is False
    assert res2.status == RouteStatus.INVALID_INPUT

    # Missing node IDs
    res3 = engine.plan_route_by_nodes(999, 4)
    assert res3.success is False
    assert res3.status == RouteStatus.INVALID_INPUT
    assert "999" in res3.status_message


def test_plan_route_invalid_num_alternatives(multi_corridor_graph):
    """Verify negative or non-integer num_alternatives returns INVALID_INPUT."""
    engine = RoutingEngine(multi_corridor_graph)

    res1 = engine.plan_route_by_nodes(1, 4, num_alternatives=-1)
    assert res1.success is False
    assert res1.status == RouteStatus.INVALID_INPUT

    res2 = engine.plan_route_by_nodes(1, 4, num_alternatives=True)
    assert res2.success is False
    assert res2.status == RouteStatus.INVALID_INPUT


def test_plan_route_invalid_metric(multi_corridor_graph):
    """Verify invalid cost metric returns INVALID_INPUT."""
    engine = RoutingEngine(multi_corridor_graph)

    res = engine.plan_route_by_nodes(1, 4, metric="invalid_metric")
    assert res.success is False
    assert res.status == RouteStatus.INVALID_INPUT


# =====================================================================
# 4. Diagnostic Semantics: SNAP_FAILED Tests
# =====================================================================

def test_plan_route_snap_failure(multi_corridor_graph):
    """Verify valid coordinates far off the network return SNAP_FAILED."""
    engine = RoutingEngine(multi_corridor_graph)

    # (20.0, 80.0) is hundreds of kilometers away from (11.0, 77.0), exceeding snap threshold
    res = engine.plan_route(20.0, 80.0, 11.00, 77.04)
    assert res.success is False
    assert res.status == RouteStatus.SNAP_FAILED
    assert res.primary_route is None


# =====================================================================
# 5. Diagnostic Semantics: SAME_LOCATION Tests
# =====================================================================

def test_plan_route_same_location(multi_corridor_graph):
    """Verify same origin and destination returns SAME_LOCATION with zero-distance route."""
    engine = RoutingEngine(multi_corridor_graph)

    # Node-based
    node_res = engine.plan_route_by_nodes(1, 1, num_alternatives=2)
    assert node_res.success is True
    assert node_res.status == RouteStatus.SAME_LOCATION
    assert node_res.primary_route is not None
    assert node_res.primary_route.node_sequence == [1]
    assert node_res.primary_route.total_distance_m == 0.0
    assert node_res.alternative_routes == []

    # Coordinate-based
    coord_res = engine.plan_route(11.00, 77.00, 11.00, 77.00)
    assert coord_res.success is True
    assert coord_res.status == RouteStatus.SAME_LOCATION
    assert coord_res.primary_route.total_distance_m == 0.0


# =====================================================================
# 6. Diagnostic Semantics: NO_PATH_FOUND Tests
# =====================================================================

def test_plan_route_no_path_found(multi_corridor_graph):
    """Verify disconnected graph endpoints return NO_PATH_FOUND."""
    multi_corridor_graph.graph.add_node(99, latitude=11.10, longitude=77.10)
    engine = RoutingEngine(multi_corridor_graph)

    res = engine.plan_route_by_nodes(1, 99)
    assert res.success is False
    assert res.status == RouteStatus.NO_PATH_FOUND
    assert res.primary_route is None
    assert res.alternative_routes == []


# =====================================================================
# 7. Backward Compatibility Tests
# =====================================================================

def test_backward_compatibility_calculate_route_methods(multi_corridor_graph):
    """Verify calculate_route and calculate_route_by_nodes signatures and Optional[Route] behavior are preserved."""
    engine = RoutingEngine(multi_corridor_graph)

    # calculate_route
    r1 = engine.calculate_route(11.00, 77.00, 11.00, 77.04)
    assert r1 is not None
    assert r1.total_distance_m == 2000.0

    # calculate_route_by_nodes
    r2 = engine.calculate_route_by_nodes(1, 4)
    assert r2 is not None
    assert r2.total_distance_m == 2000.0

    # Failure returns None directly
    assert engine.calculate_route_by_nodes(1, 9999) is None
    assert engine.calculate_route(float("nan"), 77.0, 11.0, 77.0) is None
