"""Unit tests for P6.4: Alternative Routes & Multi-Path Generation Hardening."""

from datetime import datetime, timezone
import json
import math
import networkx as nx
import pytest

from src.network.alternatives import AlternativeRouteGenerator, compute_route_overlap
from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, Route, RoutingEngine


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


@pytest.fixture
def corrupt_edge_graph():
    """
    Graph with direct corrupt edge and valid detour:
    - Direct edge 1 -> 3 has non-finite length
    - Direct edge 1 -> 3 parallel key 1 has negative speed
    - Detour corridor 1 -> 2 -> 3 is completely valid
    """
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.01, longitude=77.01)
    g.add_node(3, latitude=11.02, longitude=77.02)

    # Corrupt direct edges
    g.add_edge(1, 3, key=0, highway="trunk", length_m=float("nan"), free_flow_speed_kmh=60.0)
    g.add_edge(1, 3, key=1, highway="trunk", length_m=-1000.0, free_flow_speed_kmh=60.0)

    # Valid detour
    g.add_edge(1, 2, key=0, highway="primary", name="Detour 1", length_m=600.0, free_flow_speed_kmh=50.0)
    g.add_edge(2, 3, key=0, highway="primary", name="Detour 2", length_m=600.0, free_flow_speed_kmh=50.0)

    return RoadNetworkGraph(graph=g)


# =====================================================================
# 1. Multi-Corridor & Node-ID Alternative Routing Tests
# =====================================================================

def test_generate_alternatives_by_nodes_multiple_corridors(multi_corridor_graph):
    """Verify generate_alternative_routes_by_nodes discovers distinct viable corridors."""
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine, max_shared_ratio=0.5)

    routes = alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=3)

    assert len(routes) == 3
    # Sorted by fastest travel time
    assert routes[0].node_sequence == [1, 2, 4]
    assert routes[1].node_sequence == [1, 3, 4]
    assert routes[2].node_sequence == [1, 5, 4]

    assert routes[0].route_id == "route_1"
    assert routes[1].route_id == "route_2"
    assert routes[2].route_id == "route_3"

    # Verify pairwise overlap is strictly <= max_shared_ratio (0.5)
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            overlap = compute_route_overlap(routes[i], routes[j])
            assert overlap <= 0.5


def test_alternatives_naming_and_highway_recognition(multi_corridor_graph):
    """Verify route naming derives descriptive labels from highway references and road names."""
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    routes = alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=3)
    assert len(routes) == 3
    assert "NH-544" in routes[0].name
    assert "Avinashi Bypass" in routes[1].name
    assert "Ring Road" in routes[2].name


# =====================================================================
# 2. Defensive Edge-Cost Penalization Tests
# =====================================================================

def test_alternatives_defensive_against_none_and_corrupt_edge_weights(corrupt_edge_graph):
    """Verify penalization safely ignores None and corrupt edge costs without TypeError."""
    engine = RoutingEngine(corrupt_edge_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    # Must find the valid detour without crashing on the corrupt parallel direct edges
    routes = alt_gen.generate_alternative_routes_by_nodes(1, 3, k_routes=2)
    assert len(routes) == 1
    assert routes[0].node_sequence == [1, 2, 3]
    assert routes[0].total_distance_m == 1200.0


def test_alternatives_assembly_skips_invalid_candidate(multi_corridor_graph):
    """Verify candidate with invalid segment dimensions during assembly is safely skipped."""
    # Corrupt edge attributes on Corridor 1 after pathfinding weight check
    multi_corridor_graph.graph[1][2][0]["length_m"] = -999.0

    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    # Corridor 1 should fail assembly; remaining valid corridors (2 and 3) must be returned
    routes = alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=2)
    assert len(routes) == 2
    assert routes[0].node_sequence == [1, 3, 4]
    assert routes[1].node_sequence == [1, 5, 4]


# =====================================================================
# 3. Constrained & Edge Case Topology Tests
# =====================================================================

def test_alternatives_only_one_viable_corridor_when_k_greater_than_one(multi_corridor_graph):
    """Verify that when only 1 viable corridor exists, generator returns 1 route without duplicating."""
    # Remove corridors 2 and 3 so only corridor 1 exists
    multi_corridor_graph.graph.remove_node(3)
    multi_corridor_graph.graph.remove_node(5)

    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine, max_shared_ratio=0.5)

    routes = alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=3)
    assert len(routes) == 1
    assert routes[0].node_sequence == [1, 2, 4]


def test_alternatives_one_way_restrictions_and_directionality(multi_corridor_graph):
    """Verify directed graphs strictly enforce one-way directionality in alternatives."""
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    # Forward direction has 3 routes
    forward_routes = alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=3)
    assert len(forward_routes) == 3

    # Reverse direction has no edges; must return empty list
    reverse_routes = alt_gen.generate_alternative_routes_by_nodes(4, 1, k_routes=3)
    assert reverse_routes == []


def test_alternatives_disconnected_nodes_returns_empty_list(multi_corridor_graph):
    """Verify unreachable target node in disconnected graph returns empty list."""
    multi_corridor_graph.graph.add_node(99, latitude=12.0, longitude=78.0)
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    assert alt_gen.generate_alternative_routes_by_nodes(1, 99) == []
    assert alt_gen.generate_alternative_routes_by_nodes(99, 1) == []


def test_alternatives_same_origin_and_destination(multi_corridor_graph):
    """Verify same origin and destination node returns single zero-distance route."""
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    # Node-based
    node_routes = alt_gen.generate_alternative_routes_by_nodes(1, 1, k_routes=3)
    assert len(node_routes) == 1
    assert node_routes[0].node_sequence == [1]
    assert node_routes[0].total_distance_m == 0.0
    assert node_routes[0].total_time_s == 0.0
    assert node_routes[0].segment_count == 0

    # Coordinate-based
    coord_routes = alt_gen.generate_alternative_routes(11.00, 77.00, 11.00, 77.00, k_routes=3)
    assert len(coord_routes) == 1
    assert coord_routes[0].total_distance_m == 0.0


# =====================================================================
# 4. Defensive Input Validation Tests
# =====================================================================

def test_alternatives_rejects_invalid_inputs(multi_corridor_graph):
    """Verify invalid k_routes, boolean node IDs, missing nodes, and invalid coordinates return []."""
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    # Invalid k_routes
    assert alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=0) == []
    assert alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=-2) == []
    assert alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=True) == []

    # Invalid node types
    assert alt_gen.generate_alternative_routes_by_nodes(True, 4) == []
    assert alt_gen.generate_alternative_routes_by_nodes(1, False) == []
    assert alt_gen.generate_alternative_routes_by_nodes("1", 4) == []

    # Missing nodes
    assert alt_gen.generate_alternative_routes_by_nodes(999, 4) == []
    assert alt_gen.generate_alternative_routes_by_nodes(1, 999) == []

    # Invalid coordinates
    assert alt_gen.generate_alternative_routes(float("nan"), 77.0, 11.0, 77.04) == []
    assert alt_gen.generate_alternative_routes(11.0, 77.0, 999.0, 77.04) == []
    assert alt_gen.generate_alternative_routes(11.0, 77.0, 11.0, 77.04, k_routes=-1) == []


# =====================================================================
# 5. P6.3 Route Invariants & Summary Verification
# =====================================================================

def test_alternatives_satisfy_p6_3_invariants_and_summary(multi_corridor_graph):
    """Verify all returned alternative routes satisfy P6.3 validation and JSON summary contract."""
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    dep_time = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc)
    routes = alt_gen.generate_alternative_routes_by_nodes(
        1, 4, k_routes=3, metric=CostMetric.FREE_FLOW_TIME, departure_time=dep_time
    )

    assert len(routes) == 3
    for r in routes:
        assert isinstance(r, Route)
        assert r.segment_count == len(r.edge_sequence)
        assert len(r.edge_sequence) == len(r.node_sequence) - 1
        assert r.total_distance_m > 0.0
        assert r.total_time_s > 0.0
        assert r.average_speed_kmh > 0.0

        # Topological connectivity
        for i, seg in enumerate(r.edge_sequence):
            assert seg.u == r.node_sequence[i]
            assert seg.v == r.node_sequence[i + 1]

        # JSON Summary serializability
        summary = r.summary()
        assert summary["departure_time"] == "2026-10-10T09:00:00+00:00"
        serialized = json.dumps(summary)
        assert serialized is not None


def test_alternatives_with_distance_metric(multi_corridor_graph):
    """Verify alternatives generation with CostMetric.DISTANCE works and sorts by total distance."""
    engine = RoutingEngine(multi_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine)

    routes = alt_gen.generate_alternative_routes_by_nodes(1, 4, k_routes=3, metric=CostMetric.DISTANCE)
    assert len(routes) == 3
    assert routes[0].total_distance_m <= routes[1].total_distance_m <= routes[2].total_distance_m
