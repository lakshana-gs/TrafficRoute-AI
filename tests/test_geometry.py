"""Unit tests for RouteGeometryBuilder and GeoJSON serialization."""

import pytest
import networkx as nx
from shapely.geometry import LineString

from src.network.geometry import RouteGeometryBuilder
from src.network.graph import RoadNetworkGraph
from src.network.routing import CostMetric, RoutingEngine


@pytest.fixture
def sample_route_graph():
    g = nx.MultiDiGraph()
    g.add_node(10, latitude=11.00, longitude=76.96)
    g.add_node(20, latitude=11.05, longitude=77.00)
    g.add_node(30, latitude=11.10, longitude=77.05)

    geom1 = LineString([(76.96, 11.00), (76.98, 11.025), (77.00, 11.05)])
    geom2 = LineString([(77.00, 11.05), (77.025, 11.075), (77.05, 11.10)])

    g.add_edge(10, 20, key=0, highway="trunk", name="NH-544", ref="NH-544", length_m=7000.0, free_flow_speed_kmh=80.0, geometry=geom1)
    g.add_edge(20, 30, key=0, highway="trunk", name="NH-544", ref="NH-544", length_m=7500.0, free_flow_speed_kmh=80.0, geometry=geom2)

    return RoadNetworkGraph(graph=g)


def test_route_geojson_structure(sample_route_graph):
    engine = RoutingEngine(sample_route_graph)
    route = engine.calculate_route(11.00, 76.96, 11.10, 77.05)
    assert route is not None

    builder = RouteGeometryBuilder(sample_route_graph)
    geojson_data = builder.build_route_geojson(route)

    assert geojson_data["type"] == "FeatureCollection"
    assert "bbox" in geojson_data
    assert len(geojson_data["features"]) == 3  # Route line, Origin Point, Dest Point

    line_feat = geojson_data["features"][0]
    assert line_feat["geometry"]["type"] == "LineString"
    assert len(line_feat["geometry"]["coordinates"]) >= 4
    assert line_feat["properties"]["name"] == "Fastest Route"
    assert line_feat["properties"]["total_distance_km"] == pytest.approx(14.5, abs=0.2)
    assert "NH-544" in line_feat["properties"]["distinct_road_names"]
    assert len(line_feat["properties"]["steps"]) >= 1

    orig_feat = geojson_data["features"][1]
    assert orig_feat["properties"]["type"] == "origin"

    dest_feat = geojson_data["features"][2]
    assert dest_feat["properties"]["type"] == "destination"
