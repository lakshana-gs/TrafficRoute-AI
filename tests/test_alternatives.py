"""Unit tests for AlternativeRouteGenerator and diversity filtering."""

import pytest
import networkx as nx

from src.network.alternatives import AlternativeRouteGenerator, compute_route_overlap
from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, RoutingEngine


@pytest.fixture
def dual_corridor_graph():
    # Two distinct corridors from Node 1 (11.0, 77.0) to Node 4 (11.0, 77.04):
    # Corridor 1 (North): 1 -> 2 -> 4 (length: 2000m + 2000m = 4000m)
    # Corridor 2 (South): 1 -> 3 -> 4 (length: 2200m + 2200m = 4400m)
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_node(3, latitude=10.98, longitude=77.02)
    g.add_node(4, latitude=11.00, longitude=77.04)

    g.add_edge(1, 2, key=0, highway="primary", name="North Highway", length_m=2000.0, free_flow_speed_kmh=60.0)
    g.add_edge(2, 4, key=0, highway="primary", name="North Highway", length_m=2000.0, free_flow_speed_kmh=60.0)

    g.add_edge(1, 3, key=0, highway="secondary", name="South Bypass", length_m=2200.0, free_flow_speed_kmh=50.0)
    g.add_edge(3, 4, key=0, highway="secondary", name="South Bypass", length_m=2200.0, free_flow_speed_kmh=50.0)

    return RoadNetworkGraph(graph=g)


def test_alternative_routes_generation(dual_corridor_graph):
    engine = RoutingEngine(dual_corridor_graph)
    alt_gen = AlternativeRouteGenerator(engine, max_shared_ratio=0.5)

    routes = alt_gen.generate_alternative_routes(
        origin_lat=11.00, origin_lon=77.00,
        destination_lat=11.00, destination_lon=77.04,
        k_routes=2,
    )

    assert len(routes) == 2
    # Route 1: North corridor
    assert routes[0].node_sequence == [1, 2, 4]
    # Route 2: South bypass
    assert routes[1].node_sequence == [1, 3, 4]

    # Verify overlap is 0.0 (completely distinct)
    overlap = compute_route_overlap(routes[0], routes[1])
    assert overlap == 0.0
