"""Unit tests for spatial indexing and location snapping."""

import pytest
import networkx as nx
from shapely.geometry import LineString

from src.network.graph import RoadNetworkGraph
from src.network.snapping import NetworkSpatialIndexer, SnapResult


@pytest.fixture
def sample_toy_graph():
    g = nx.MultiDiGraph()
    # Linear street: Node 1 (11.0, 77.0) -> Node 2 (11.0, 77.01)
    g.add_node(1, latitude=11.0, longitude=77.0)
    g.add_node(2, latitude=11.0, longitude=77.01)
    geom = LineString([(77.0, 11.0), (77.01, 11.0)])
    g.add_edge(1, 2, key=0, highway="primary", name="Main Street", geometry=geom, length_m=1090.0)
    return RoadNetworkGraph(graph=g)


def test_exact_node_snapping(sample_toy_graph):
    indexer = NetworkSpatialIndexer(sample_toy_graph)
    # Query exact coordinates of Node 1
    snap = indexer.snap_coordinate(11.0, 77.0)
    assert snap.is_valid is True
    assert snap.nearest_node_id == 1
    assert snap.distance_meters < 5.0
    assert snap.road_name == "Main Street"
    assert snap.highway_type == "primary"


def test_perpendicular_edge_snapping(sample_toy_graph):
    indexer = NetworkSpatialIndexer(sample_toy_graph)
    # Query point slightly north of midpoint (11.0005, 77.005)
    snap = indexer.snap_coordinate(11.0005, 77.005, snap_to_edge=True)
    assert snap.is_valid is True
    assert snap.snapped_latitude == pytest.approx(11.0, abs=1e-4)
    assert snap.snapped_longitude == pytest.approx(77.005, abs=1e-4)
    assert snap.distance_meters > 0.0
    assert snap.distance_meters < 100.0


def test_far_off_network_snapping(sample_toy_graph):
    indexer = NetworkSpatialIndexer(sample_toy_graph, max_snap_distance_m=500.0)
    # Coordinate 50 km away
    snap = indexer.snap_coordinate(11.5, 77.5)
    assert snap.is_valid is False
    assert "exceeds maximum threshold" in snap.warning
