"""Unit tests for P6.2: Defensive input and edge-cost validation."""

import math
import networkx as nx
import pytest

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, RoutingEngine


@pytest.fixture
def triangle_network():
    """Deterministic valid 3-node network."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_node(3, latitude=11.00, longitude=77.02)

    g.add_edge(1, 2, key=0, highway="trunk", name="Expressway", length_m=2000.0, free_flow_speed_kmh=80.0)
    g.add_edge(1, 3, key=0, highway="residential", name="Local 1", length_m=600.0, free_flow_speed_kmh=25.0)
    g.add_edge(3, 2, key=0, highway="residential", name="Local 2", length_m=600.0, free_flow_speed_kmh=25.0)

    return RoadNetworkGraph(graph=g)


@pytest.fixture
def mixed_validity_parallel_graph():
    """Graph with parallel edges between 1 and 2: Key 0 is corrupt, Key 1 is valid."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)

    # Corrupt edge (negative length)
    g.add_edge(1, 2, key=0, highway="trunk", name="Corrupt Edge", length_m=-500.0, free_flow_speed_kmh=80.0)
    # Valid edge
    g.add_edge(1, 2, key=1, highway="primary", name="Valid Edge", length_m=800.0, free_flow_speed_kmh=40.0)

    return RoadNetworkGraph(graph=g)


@pytest.fixture
def corrupt_direct_with_valid_detour_graph():
    """Graph where direct edge is corrupt but a legal detour is valid."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_node(3, latitude=11.00, longitude=77.02)

    # Corrupt direct edge (zero length)
    g.add_edge(1, 2, key=0, length_m=0.0, free_flow_speed_kmh=50.0)
    # Valid detour
    g.add_edge(1, 3, key=0, length_m=700.0, free_flow_speed_kmh=40.0)
    g.add_edge(3, 2, key=0, length_m=700.0, free_flow_speed_kmh=40.0)

    return RoadNetworkGraph(graph=g)


@pytest.fixture
def all_invalid_graph():
    """Graph where all edges have corrupt/invalid dimensions."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)

    g.add_edge(1, 2, key=0, length_m=-1000.0, free_flow_speed_kmh=-30.0)

    return RoadNetworkGraph(graph=g)


# =====================================================================
# 1. Coordinate Validation Tests (calculate_route)
# =====================================================================

def test_invalid_coordinate_types_return_none(triangle_network):
    """Test that non-numeric coordinate types return None deterministically."""
    engine = RoutingEngine(triangle_network)

    assert engine.calculate_route("11.00", 77.00, 11.02, 77.02) is None
    assert engine.calculate_route(11.00, None, 11.02, 77.02) is None
    assert engine.calculate_route(11.00, 77.00, [], 77.02) is None
    assert engine.calculate_route(11.00, 77.00, 11.02, {}) is None
    assert engine.calculate_route(True, 77.00, 11.02, 77.02) is None


def test_nan_and_infinite_coordinates_return_none(triangle_network):
    """Test that NaN and infinite coordinates return None without throwing exceptions."""
    engine = RoutingEngine(triangle_network)

    assert engine.calculate_route(float("nan"), 77.00, 11.02, 77.02) is None
    assert engine.calculate_route(11.00, float("nan"), 11.02, 77.02) is None
    assert engine.calculate_route(11.00, 77.00, float("inf"), 77.02) is None
    assert engine.calculate_route(11.00, 77.00, 11.02, float("-inf")) is None
    # Identical NaN coordinates should also return None rather than a fabricated route
    assert engine.calculate_route(float("nan"), float("nan"), float("nan"), float("nan")) is None


def test_out_of_bounds_wgs84_coordinates_return_none(triangle_network):
    """Test that coordinates outside standard WGS84 ranges return None."""
    engine = RoutingEngine(triangle_network)

    assert engine.calculate_route(90.1, 77.00, 11.02, 77.02) is None
    assert engine.calculate_route(-90.1, 77.00, 11.02, 77.02) is None
    assert engine.calculate_route(11.00, 180.1, 11.02, 77.02) is None
    assert engine.calculate_route(11.00, -180.1, 11.02, 77.02) is None


# =====================================================================
# 2. Node ID Validation Tests (calculate_route_by_nodes)
# =====================================================================

def test_boolean_node_ids_rejected(triangle_network):
    """Test that boolean node IDs are strictly rejected and return None."""
    engine = RoutingEngine(triangle_network)

    # In Python, True == 1 and False == 0 in hash tables; these must be rejected
    assert engine.calculate_route_by_nodes(True, 2) is None
    assert engine.calculate_route_by_nodes(1, False) is None
    assert engine.calculate_route_by_nodes(True, True) is None


def test_invalid_node_id_types_rejected(triangle_network):
    """Test that non-integer node IDs return None."""
    engine = RoutingEngine(triangle_network)

    assert engine.calculate_route_by_nodes("1", 2) is None
    assert engine.calculate_route_by_nodes(1, "2") is None
    assert engine.calculate_route_by_nodes(None, 2) is None
    assert engine.calculate_route_by_nodes(1, 2.5) is None


# =====================================================================
# 3. Metric Validation Tests
# =====================================================================

def test_invalid_or_unsupported_metrics_return_none(triangle_network):
    """Test that invalid, unsupported, or malformed metrics return None."""
    engine = RoutingEngine(triangle_network)

    # Coordinate routing
    assert engine.calculate_route(11.00, 77.00, 11.02, 77.02, metric="unknown_metric") is None
    assert engine.calculate_route(11.00, 77.00, 11.02, 77.02, metric=None) is None
    assert engine.calculate_route(11.00, 77.00, 11.02, 77.02, metric=123) is None

    # Node-ID routing
    assert engine.calculate_route_by_nodes(1, 2, metric="flying_teleport") is None
    assert engine.calculate_route_by_nodes(1, 2, metric=None) is None
    assert engine.calculate_route_by_nodes(1, 2, metric=999) is None


def test_valid_string_metrics_accepted(triangle_network):
    """Test that valid string representations of CostMetric are safely accepted."""
    engine = RoutingEngine(triangle_network)

    route_dist = engine.calculate_route_by_nodes(1, 2, metric="distance")
    assert route_dist is not None
    assert route_dist.cost_metric == CostMetric.DISTANCE

    route_time = engine.calculate_route_by_nodes(1, 2, metric="free_flow_time")
    assert route_time is not None
    assert route_time.cost_metric == CostMetric.FREE_FLOW_TIME


# =====================================================================
# 4. Edge-Cost Attribute Rejection Tests (get_edge_weight)
# =====================================================================

def test_get_edge_weight_rejects_negative_or_zero_length(triangle_network):
    """Test that negative or zero edge length returns None without clamping."""
    engine = RoutingEngine(triangle_network)

    # Negative length
    assert engine.get_edge_weight(1, 2, {"length_m": -10.0, "free_flow_speed_kmh": 50.0}, CostMetric.DISTANCE) is None
    assert engine.get_edge_weight(1, 2, {"length_m": -10.0, "free_flow_speed_kmh": 50.0}, CostMetric.FREE_FLOW_TIME) is None

    # Zero length
    assert engine.get_edge_weight(1, 2, {"length_m": 0.0, "free_flow_speed_kmh": 50.0}, CostMetric.DISTANCE) is None
    assert engine.get_edge_weight(1, 2, {"length_m": 0.0, "free_flow_speed_kmh": 50.0}, CostMetric.FREE_FLOW_TIME) is None


def test_get_edge_weight_rejects_negative_or_zero_speed(triangle_network):
    """Test that negative or zero speed returns None under FREE_FLOW_TIME."""
    engine = RoutingEngine(triangle_network)

    # Negative speed
    assert engine.get_edge_weight(1, 2, {"length_m": 100.0, "free_flow_speed_kmh": -30.0}, CostMetric.FREE_FLOW_TIME) is None
    # Zero speed
    assert engine.get_edge_weight(1, 2, {"length_m": 100.0, "free_flow_speed_kmh": 0.0}, CostMetric.FREE_FLOW_TIME) is None

    # For DISTANCE, speed is irrelevant so valid length still returns distance
    assert engine.get_edge_weight(1, 2, {"length_m": 100.0, "free_flow_speed_kmh": 0.0}, CostMetric.DISTANCE) == 100.0


def test_get_edge_weight_rejects_nan_none_and_infinite_attributes(triangle_network):
    """Test that NaN, None, and infinite attributes return None without raising exceptions."""
    engine = RoutingEngine(triangle_network)

    # Missing / None
    assert engine.get_edge_weight(1, 2, {"length_m": None, "free_flow_speed_kmh": 50.0}) is None
    assert engine.get_edge_weight(1, 2, {"length_m": 100.0, "free_flow_speed_kmh": None}) is None
    assert engine.get_edge_weight(1, 2, {}) is None
    assert engine.get_edge_weight(1, 2, "not_a_dict") is None

    # NaN
    assert engine.get_edge_weight(1, 2, {"length_m": float("nan"), "free_flow_speed_kmh": 50.0}) is None
    assert engine.get_edge_weight(1, 2, {"length_m": 100.0, "free_flow_speed_kmh": float("nan")}) is None

    # Infinite
    assert engine.get_edge_weight(1, 2, {"length_m": float("inf"), "free_flow_speed_kmh": 50.0}) is None
    assert engine.get_edge_weight(1, 2, {"length_m": 100.0, "free_flow_speed_kmh": float("inf")}) is None


# =====================================================================
# 5. Parallel Edge and Detour Recovery Tests
# =====================================================================

def test_parallel_edges_selects_valid_edge(mixed_validity_parallel_graph):
    """Test that parallel edges select the valid edge and completely ignore the corrupt one."""
    engine = RoutingEngine(mixed_validity_parallel_graph)

    route = engine.calculate_route_by_nodes(1, 2, metric=CostMetric.DISTANCE)
    assert route is not None
    assert len(route.edge_sequence) == 1
    # Must select key 1 (valid 800m), NOT key 0 (corrupt -500m)
    assert route.edge_sequence[0].key == 1
    assert route.edge_sequence[0].name == "Valid Edge"
    assert route.total_distance_m == 800.0


def test_corrupt_direct_edge_takes_valid_detour(corrupt_direct_with_valid_detour_graph):
    """Test that a corrupt direct edge causes pathfinding to take a valid legal detour."""
    engine = RoutingEngine(corrupt_direct_with_valid_detour_graph)

    route = engine.calculate_route_by_nodes(1, 2)
    assert route is not None
    # Must route via 1 -> 3 -> 2 rather than traversing direct edge 1 -> 2
    assert route.node_sequence == [1, 3, 2]
    assert len(route.edge_sequence) == 2
    assert route.total_distance_m == 1400.0


def test_all_invalid_paths_return_none(all_invalid_graph):
    """Test that when all paths are invalid, router returns None without crashing."""
    engine = RoutingEngine(all_invalid_graph)

    assert engine.calculate_route_by_nodes(1, 2) is None
    assert engine.calculate_route(11.00, 77.00, 11.02, 77.02) is None


# =====================================================================
# 6. Standardized Graph Derivation Preservation
# =====================================================================

def test_standardized_hierarchy_fallback_preserved(triangle_network):
    """Preservation test: verify that derived hierarchy fallback speeds (30 km/h) remain valid."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.01, longitude=77.01)
    # Standard P1 hierarchy fallback edge
    g.add_edge(
        1, 2, key=0,
        highway="unclassified",
        length_m=1000.0,
        free_flow_speed_kmh=30.0,
        speed_source="DERIVED_ROAD_HIERARCHY",
    )
    rg = RoadNetworkGraph(graph=g)
    engine = RoutingEngine(rg)

    route = engine.calculate_route_by_nodes(1, 2, metric=CostMetric.FREE_FLOW_TIME)
    assert route is not None
    assert route.total_distance_m == 1000.0
    # 1000m / (30 * 1000/3600 m/s) = 120s
    assert route.total_time_s == pytest.approx(120.0, abs=0.5)


# =====================================================================
# 7. Additional Audit Gap Coverage Tests
# =====================================================================

def test_distance_routing_with_missing_speed_uses_fallback_safely():
    """Verify that DISTANCE routing on edge with None free_flow_speed_kmh safely uses 30 km/h fallback."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_edge(1, 2, key=0, length_m=1000.0, free_flow_speed_kmh=50.0)
    rg = RoadNetworkGraph(graph=g)
    # Simulate post-init missing speed attribute
    rg.graph[1][2][0]["free_flow_speed_kmh"] = None

    engine = RoutingEngine(rg)
    route = engine.calculate_route_by_nodes(1, 2, metric=CostMetric.DISTANCE)

    assert route is not None
    assert route.total_distance_m == 1000.0
    assert route.edge_sequence[0].speed_kmh == 30.0
    # 1000m / (30 * 1000 / 3600 m/s) = 120.0s
    assert route.total_time_s == pytest.approx(120.0, abs=0.5)


def test_digraph_routing_callback_handles_single_edge_dict():
    """Verify that NetworkX weight callback and segment assembly support DiGraph single-dict edge format."""
    rg = RoadNetworkGraph()
    g = nx.DiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_edge(1, 2, length_m=500.0, free_flow_speed_kmh=40.0)
    rg.graph = g

    engine = RoutingEngine(rg)
    route = engine.calculate_route_by_nodes(1, 2, metric=CostMetric.DISTANCE)

    assert route is not None
    assert route.node_sequence == [1, 2]
    assert route.total_distance_m == 500.0
    assert len(route.edge_sequence) == 1
    assert route.edge_sequence[0].key == 0
    assert route.edge_sequence[0].speed_kmh == 40.0


def test_corrupt_length_returns_none_without_exception():
    """Verify that corrupt length_m (None, non-numeric, or negative) returns None without raising exceptions."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_edge(1, 2, key=0, length_m=1000.0, free_flow_speed_kmh=40.0)
    rg = RoadNetworkGraph(graph=g)
    rg.graph[1][2][0]["length_m"] = "corrupt_length"

    engine = RoutingEngine(rg)
    assert engine.calculate_route_by_nodes(1, 2) is None
    assert engine.calculate_route(11.00, 77.00, 11.02, 77.02) is None

