"""Unit tests for RoadNetworkGraph schema, attributes, and serialization."""

import pytest
import networkx as nx
from shapely.geometry import LineString

from src.network.graph import RoadNetworkGraph, ROAD_CLASSIFICATION_HIERARCHY, parse_maxspeed_value, parse_lanes_value
from src.network.provenance import NetworkProvenance


def test_parse_maxspeed_value():
    assert parse_maxspeed_value("80") == 80.0
    assert parse_maxspeed_value("80 km/h") == 80.0
    assert parse_maxspeed_value(60) == 60.0
    assert parse_maxspeed_value(None) is None
    assert parse_maxspeed_value("signals") is None
    assert parse_maxspeed_value("unknown") is None
    assert parse_maxspeed_value(["80", "100"]) == 80.0


def test_parse_lanes_value():
    assert parse_lanes_value("2") == 2
    assert parse_lanes_value(4) == 4
    assert parse_lanes_value("2;3") == 2
    assert parse_lanes_value(None) is None
    assert parse_lanes_value("invalid") is None


def test_road_network_graph_schema_standardization():
    g = nx.MultiDiGraph()
    g.add_node(1, latitude=11.0, longitude=77.0)
    g.add_node(2, latitude=11.01, longitude=77.01)
    g.add_edge(
        1, 2, key=0,
        highway="trunk",
        name="Avinashi Road",
        ref="NH-544",
        maxspeed="80 km/h",
        lanes="4",
        oneway="yes"
    )

    prov = NetworkProvenance(
        dataset_name="TestNetwork",
        extraction_method="UnitTest",
        geographic_boundary={"north": 11.02, "south": 10.99, "east": 77.02, "west": 76.99},
    )

    road_graph = RoadNetworkGraph(graph=g, provenance=prov)
    assert road_graph.node_count == 2
    assert road_graph.edge_count == 1

    node1 = road_graph.get_node(1)
    assert node1["latitude"] == 11.0
    assert node1["longitude"] == 77.0

    edge = road_graph.get_edge(1, 2, 0)
    assert edge["highway_type"] == "trunk"
    assert edge["name"] == "Avinashi Road"
    assert edge["road_reference"] == "NH-544"
    assert edge["maxspeed"] == 80.0
    assert edge["free_flow_speed_kmh"] == 80.0
    assert edge["lanes"] == 4
    assert edge["oneway"] is True
    assert edge["length_m"] > 0.0
    assert isinstance(edge["geometry"], LineString)


def test_road_network_graph_binary_serialization(tmp_path):
    g = nx.MultiDiGraph()
    g.add_node(100, latitude=11.0016, longitude=76.9667)
    g.add_node(200, latitude=11.1932, longitude=77.2690)
    g.add_edge(100, 200, key=0, highway="trunk", name="NH-544", length_m=42000.0)

    prov = NetworkProvenance(
        dataset_name="SerializationTest",
        extraction_method="UnitTest",
        geographic_boundary={"north": 11.2, "south": 11.0, "east": 77.3, "west": 76.9},
    )
    rg = RoadNetworkGraph(graph=g, provenance=prov)

    bin_file = tmp_path / "test_graph.bin"
    rg.save_binary(bin_file)
    assert bin_file.exists()

    loaded_rg = RoadNetworkGraph.load_binary(bin_file)
    assert loaded_rg.node_count == 2
    assert loaded_rg.edge_count == 1
    assert loaded_rg.get_node(100)["latitude"] == 11.0016
    assert loaded_rg.get_edge(100, 200, 0)["highway_type"] == "trunk"
