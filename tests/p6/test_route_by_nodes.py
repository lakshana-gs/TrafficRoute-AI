"""Unit tests for P6.1: Direct node-ID routing in RoutingEngine."""

from datetime import datetime
import networkx as nx
import pytest

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, RoutingEngine


@pytest.fixture
def triangle_network():
    """
    Deterministic 3-node network:
    Node 1 -> Node 2: Direct trunk expressway (2000m @ 80 km/h -> 90s)
    Node 1 -> Node 3: Residential street (600m @ 25 km/h -> 86.4s)
    Node 3 -> Node 2: Residential street (600m @ 25 km/h -> 86.4s)
    Total 1 -> 3 -> 2: 1200m, 172.8s
    """
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_node(3, latitude=11.00, longitude=77.02)

    g.add_edge(1, 2, key=0, highway="trunk", name="Expressway", length_m=2000.0, free_flow_speed_kmh=80.0)
    g.add_edge(1, 3, key=0, highway="residential", name="Local 1", length_m=600.0, free_flow_speed_kmh=25.0)
    g.add_edge(3, 2, key=0, highway="residential", name="Local 2", length_m=600.0, free_flow_speed_kmh=25.0)

    return RoadNetworkGraph(graph=g)


@pytest.fixture
def oneway_detour_network():
    """
    Network demonstrating one-way directionality and legal detour:
    10 -> 20: Direct one-way arterial (1000m @ 50 km/h)
    20 -> 30: Detour leg 1 (800m @ 40 km/h)
    30 -> 10: Detour leg 2 (800m @ 40 km/h)
    No reverse edge 20 -> 10 directly.
    """
    g = nx.MultiDiGraph()
    g.add_node(10, latitude=11.10, longitude=77.10)
    g.add_node(20, latitude=11.12, longitude=77.12)
    g.add_node(30, latitude=11.14, longitude=77.10)

    g.add_edge(10, 20, key=0, highway="primary", name="Oneway Main", length_m=1000.0, free_flow_speed_kmh=50.0, oneway=True)
    g.add_edge(20, 30, key=0, highway="secondary", name="Detour East", length_m=800.0, free_flow_speed_kmh=40.0, oneway=True)
    g.add_edge(30, 10, key=0, highway="secondary", name="Detour West", length_m=800.0, free_flow_speed_kmh=40.0, oneway=True)

    return RoadNetworkGraph(graph=g)


@pytest.fixture
def parallel_edges_network():
    """
    Network with two parallel edges between nodes 100 and 200:
    Key 0: Short slow route (500m @ 25 km/h -> 72s)
    Key 1: Longer fast arterial (800m @ 80 km/h -> 36s)
    """
    g = nx.MultiDiGraph()
    g.add_node(100, latitude=11.20, longitude=77.20)
    g.add_node(200, latitude=11.21, longitude=77.21)

    g.add_edge(
        100,
        200,
        key=0,
        highway="residential",
        name="Short Slow Local",
        length_m=500.0,
        free_flow_speed_kmh=25.0,
        edge_id="e_100_200_0",
    )
    g.add_edge(
        100,
        200,
        key=1,
        highway="trunk",
        name="Long Fast Arterial",
        length_m=800.0,
        free_flow_speed_kmh=80.0,
        edge_id="e_100_200_1",
    )

    return RoadNetworkGraph(graph=g)


@pytest.fixture
def disconnected_network():
    """Network with two disconnected components."""
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.01, longitude=77.01)
    g.add_node(99, latitude=11.50, longitude=77.50)  # Isolated node

    g.add_edge(1, 2, key=0, highway="primary", name="Main Road", length_m=1000.0, free_flow_speed_kmh=50.0)

    return RoadNetworkGraph(graph=g)


def test_direct_routing_and_totals(triangle_network):
    """Test direct routing by node IDs and accurate metric aggregation."""
    engine = RoutingEngine(triangle_network)
    route = engine.calculate_route_by_nodes(
        origin_node_id=1,
        destination_node_id=2,
        metric=CostMetric.FREE_FLOW_TIME,
        route_name="Fastest Direct",
        route_id="route_fwd",
    )

    assert route is not None
    assert route.route_id == "route_fwd"
    assert route.name == "Fastest Direct"
    assert route.cost_metric == CostMetric.FREE_FLOW_TIME
    assert route.node_sequence == [1, 2]
    assert len(route.edge_sequence) == 1

    seg = route.edge_sequence[0]
    assert seg.u == 1
    assert seg.v == 2
    assert seg.name == "Expressway"
    assert seg.length_m == 2000.0
    assert seg.speed_kmh == 80.0
    assert seg.traversal_time_s == pytest.approx(90.0, abs=0.5)

    assert route.total_distance_m == 2000.0
    assert route.total_distance_km == 2.0
    assert route.total_time_s == pytest.approx(90.0, abs=0.5)
    assert route.total_time_minutes == 1.5

    # Sum of segment costs equals route totals
    assert sum(s.length_m for s in route.edge_sequence) == pytest.approx(route.total_distance_m)
    assert sum(s.traversal_time_s for s in route.edge_sequence) == pytest.approx(route.total_time_s)

    # Snaps reflect direct node reference with zero snap distance
    assert route.origin_snap.nearest_node_id == 1
    assert route.origin_snap.distance_meters == 0.0
    assert route.origin_snap.is_valid is True
    assert route.destination_snap.nearest_node_id == 2
    assert route.destination_snap.distance_meters == 0.0
    assert route.destination_snap.is_valid is True


def test_distance_vs_free_flow_time_different_paths(triangle_network):
    """Test that DISTANCE and FREE_FLOW_TIME metrics choose different paths."""
    engine = RoutingEngine(triangle_network)

    # Shortest distance chooses 1 -> 3 -> 2 (1200m < 2000m)
    route_dist = engine.calculate_route_by_nodes(1, 2, metric=CostMetric.DISTANCE)
    assert route_dist is not None
    assert route_dist.node_sequence == [1, 3, 2]
    assert route_dist.total_distance_m == 1200.0
    assert len(route_dist.edge_sequence) == 2
    assert route_dist.cost_metric == CostMetric.DISTANCE

    # Fastest free-flow time chooses 1 -> 2 (90s < 172.8s)
    route_time = engine.calculate_route_by_nodes(1, 2, metric=CostMetric.FREE_FLOW_TIME)
    assert route_time is not None
    assert route_time.node_sequence == [1, 2]
    assert route_time.total_distance_m == 2000.0
    assert route_time.total_time_s == pytest.approx(90.0, abs=0.5)
    assert len(route_time.edge_sequence) == 1
    assert route_time.cost_metric == CostMetric.FREE_FLOW_TIME


def test_one_way_directionality_restriction(oneway_detour_network):
    """Test that one-way edges prevent illegal reverse traversal."""
    # Graph with strictly 10 -> 20 only (no reverse path)
    g_strict = nx.MultiDiGraph()
    g_strict.add_node(10, latitude=11.10, longitude=77.10)
    g_strict.add_node(20, latitude=11.12, longitude=77.12)
    g_strict.add_edge(10, 20, key=0, length_m=1000.0, free_flow_speed_kmh=50.0)

    engine_strict = RoutingEngine(RoadNetworkGraph(graph=g_strict))

    # Forward route exists
    forward_route = engine_strict.calculate_route_by_nodes(10, 20)
    assert forward_route is not None
    assert forward_route.node_sequence == [10, 20]

    # Reverse route is strictly impossible -> returns None
    reverse_route = engine_strict.calculate_route_by_nodes(20, 10)
    assert reverse_route is None


def test_legal_directed_detour(oneway_detour_network):
    """Test that engine routes via legal directed detour when direct reverse is unavailable."""
    engine = RoutingEngine(oneway_detour_network)

    # 10 -> 20 goes direct
    direct_route = engine.calculate_route_by_nodes(10, 20)
    assert direct_route is not None
    assert direct_route.node_sequence == [10, 20]
    assert direct_route.total_distance_m == 1000.0

    # 20 -> 10 must take legal directed detour: 20 -> 30 -> 10
    detour_route = engine.calculate_route_by_nodes(20, 10)
    assert detour_route is not None
    assert detour_route.node_sequence == [20, 30, 10]
    assert len(detour_route.edge_sequence) == 2
    assert detour_route.total_distance_m == 1600.0


def test_parallel_edges_minimum_cost_selection(parallel_edges_network):
    """Test that parallel edges select minimum-cost edge with actual attributes."""
    engine = RoutingEngine(parallel_edges_network)

    # By DISTANCE: chooses Key 0 (500m < 800m)
    route_dist = engine.calculate_route_by_nodes(100, 200, metric=CostMetric.DISTANCE)
    assert route_dist is not None
    assert len(route_dist.edge_sequence) == 1
    seg_dist = route_dist.edge_sequence[0]
    assert seg_dist.key == 0
    assert seg_dist.name == "Short Slow Local"
    assert seg_dist.edge_id == "e_100_200_0"
    assert seg_dist.length_m == 500.0
    assert route_dist.total_distance_m == 500.0

    # By FREE_FLOW_TIME: chooses Key 1 (36s < 72s)
    route_time = engine.calculate_route_by_nodes(100, 200, metric=CostMetric.FREE_FLOW_TIME)
    assert route_time is not None
    assert len(route_time.edge_sequence) == 1
    seg_time = route_time.edge_sequence[0]
    assert seg_time.key == 1
    assert seg_time.name == "Long Fast Arterial"
    assert seg_time.edge_id == "e_100_200_1"
    assert seg_time.length_m == 800.0
    assert seg_time.speed_kmh == 80.0
    assert route_time.total_distance_m == 800.0
    assert route_time.total_time_s == pytest.approx(36.0, abs=0.5)


def test_disconnected_nodes_returns_none(disconnected_network):
    """Test that queries between disconnected components return None."""
    engine = RoutingEngine(disconnected_network)

    assert engine.calculate_route_by_nodes(1, 99) is None
    assert engine.calculate_route_by_nodes(99, 1) is None
    assert engine.calculate_route_by_nodes(2, 99) is None


def test_missing_node_ids_returns_none(triangle_network):
    """Test that missing origin or destination node IDs return None."""
    engine = RoutingEngine(triangle_network)

    # Missing origin
    assert engine.calculate_route_by_nodes(9999, 2) is None
    # Missing destination
    assert engine.calculate_route_by_nodes(1, 9999) is None
    # Both missing
    assert engine.calculate_route_by_nodes(9998, 9999) is None
    # Same non-existent node ID
    assert engine.calculate_route_by_nodes(9999, 9999) is None


def test_same_existing_node_zero_distance_and_time(triangle_network):
    """Test that routing between the same existing node returns zero distance and time."""
    engine = RoutingEngine(triangle_network)
    route = engine.calculate_route_by_nodes(1, 1)

    assert route is not None
    assert route.node_sequence == [1]
    assert route.edge_sequence == []
    assert route.total_distance_m == 0.0
    assert route.total_time_s == 0.0
    assert route.total_distance_km == 0.0
    assert route.total_time_minutes == 0.0
    assert route.origin_snap.nearest_node_id == 1
    assert route.destination_snap.nearest_node_id == 1
    assert route.origin_snap.distance_meters == 0.0
    assert route.destination_snap.distance_meters == 0.0
    assert route.origin_snap.is_valid is True
    assert route.destination_snap.is_valid is True
    assert route.cost_metric == CostMetric.FREE_FLOW_TIME
    assert route.name == "Fastest Route"
    assert route.route_id == "primary_route"


def test_deterministic_repeated_results(triangle_network):
    """Test that repeated route calculations yield identical results."""
    engine = RoutingEngine(triangle_network)

    results = [engine.calculate_route_by_nodes(1, 2) for _ in range(5)]
    first = results[0]

    for other in results[1:]:
        assert other is not None
        assert other.node_sequence == first.node_sequence
        assert other.total_distance_m == first.total_distance_m
        assert other.total_time_s == first.total_time_s
        assert len(other.edge_sequence) == len(first.edge_sequence)
        assert [s.edge_id for s in other.edge_sequence] == [s.edge_id for s in first.edge_sequence]


def test_custom_metadata_and_departure_time(triangle_network):
    """Test passing custom route_name, route_id, and departure_time."""
    engine = RoutingEngine(triangle_network)
    dep_time = datetime(2026, 10, 10, 15, 30, 0)
    route = engine.calculate_route_by_nodes(
        origin_node_id=1,
        destination_node_id=2,
        departure_time=dep_time,
        route_name="Pilot Express Route",
        route_id="custom_od_001",
    )

    assert route is not None
    assert route.route_id == "custom_od_001"
    assert route.name == "Pilot Express Route"
    assert route.departure_time == dep_time


def test_calculate_route_remains_unchanged(triangle_network):
    """Preservation test: verify that existing calculate_route() still functions identically."""
    engine = RoutingEngine(triangle_network)

    route = engine.calculate_route(
        origin_lat=11.00,
        origin_lon=77.00,
        destination_lat=11.02,
        destination_lon=77.02,
        metric=CostMetric.FREE_FLOW_TIME,
    )
    assert route is not None
    assert route.node_sequence == [1, 2]
    assert route.total_distance_m == 2000.0
    assert route.total_time_s == pytest.approx(90.0, abs=1.0)
