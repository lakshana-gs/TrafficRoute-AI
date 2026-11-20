"""Unit tests for RoutingEngine and cost evaluation."""

import pytest
import networkx as nx
from shapely.geometry import LineString

from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, RoutingEngine


@pytest.fixture
def triangle_network():
    # Triangle: 1 -> 2 (Highway 80 km/h, 2000m) vs 1 -> 3 -> 2 (Residential 25 km/h, 1200m total)
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.00, longitude=77.00)
    g.add_node(2, latitude=11.02, longitude=77.02)
    g.add_node(3, latitude=11.00, longitude=77.02)

    # Edge 1 -> 2: High speed highway
    g.add_edge(1, 2, key=0, highway="trunk", name="Expressway", length_m=2000.0, free_flow_speed_kmh=80.0)
    # Edges 1 -> 3 -> 2: Short distance residential
    g.add_edge(1, 3, key=0, highway="residential", name="Local 1", length_m=600.0, free_flow_speed_kmh=25.0)
    g.add_edge(3, 2, key=0, highway="residential", name="Local 2", length_m=600.0, free_flow_speed_kmh=25.0)

    return RoadNetworkGraph(graph=g)


def test_shortest_distance_vs_fastest_time_routing(triangle_network):
    engine = RoutingEngine(triangle_network)

    # By Distance: 1 -> 3 -> 2 (1200m) should be chosen over 1 -> 2 (2000m)
    route_dist = engine.calculate_route(
        origin_lat=11.00, origin_lon=77.00,
        destination_lat=11.02, destination_lon=77.02,
        metric=CostMetric.DISTANCE
    )
    assert route_dist is not None
    assert route_dist.node_sequence == [1, 3, 2]
    assert route_dist.total_distance_m == 1200.0

    # By Free-Flow Time:
    # 1 -> 2: 2000m / (80 * 1000/3600) = 90s
    # 1 -> 3 -> 2: 1200m / (25 * 1000/3600) = 172.8s
    # 1 -> 2 is faster!
    route_time = engine.calculate_route(
        origin_lat=11.00, origin_lon=77.00,
        destination_lat=11.02, destination_lon=77.02,
        metric=CostMetric.FREE_FLOW_TIME
    )
    assert route_time is not None
    assert route_time.node_sequence == [1, 2]
    assert route_time.total_time_s == pytest.approx(90.0, abs=1.0)
    assert route_time.edge_sequence[0].name == "Expressway"


def test_same_origin_and_destination(triangle_network):
    engine = RoutingEngine(triangle_network)
    route = engine.calculate_route(11.00, 77.00, 11.00, 77.00)
    assert route is not None
    assert route.total_distance_m == 0.0
    assert len(route.node_sequence) == 1
