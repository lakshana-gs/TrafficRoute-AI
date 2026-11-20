"""Unit tests for TrafficSpatialMatcher and road network linkage."""

import pytest
import pandas as pd
import networkx as nx
from shapely.geometry import LineString

from src.network.graph import RoadNetworkGraph
from src.traffic.spatial import TrafficSpatialMatcher


@pytest.fixture
def mock_pilot_graph():
    g = nx.MultiDiGraph()
    # Segment 1 -> 2 along latitude 11.02, longitude 77.00 to 77.02
    g.add_node(1, latitude=11.0200, longitude=77.0000)
    g.add_node(2, latitude=11.0200, longitude=77.0200)
    geom = LineString([(77.0000, 11.0200), (77.0200, 11.0200)])
    g.add_edge(1, 2, key=0, highway="trunk", name="NH-544", geometry=geom, length_m=2180.0)
    return RoadNetworkGraph(graph=g)


def test_spatial_matching_within_threshold(mock_pilot_graph):
    matcher = TrafficSpatialMatcher(mock_pilot_graph, max_snap_distance_m=75.0)

    # Point 20m north of highway centerline: (11.02018, 77.0100)
    df = pd.DataFrame([
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P1", "latitude": 11.02018, "longitude": 77.0100, "observed_speed_kmh": 65.0}
    ])

    matched_df, report = matcher.match_dataframe(df)

    assert report.total_observations == 1
    assert report.matched_observations == 1
    assert report.unmatched_observations == 0
    assert report.match_rate_pct == 100.0
    assert report.mean_match_distance_m < 30.0

    matched_row = matched_df.iloc[0]
    assert bool(matched_row["is_matched"]) is True
    assert matched_row["matched_edge_id"] == "1_2_0"
    assert matched_row["highway_type"] == "trunk"


def test_unmatched_observation_outside_threshold(mock_pilot_graph):
    matcher = TrafficSpatialMatcher(mock_pilot_graph, max_snap_distance_m=50.0)

    # Point 500m away: (11.0250, 77.0100)
    df = pd.DataFrame([
        {"timestamp": "2026-09-01T08:00:00Z", "traffic_entity_id": "P2", "latitude": 11.0250, "longitude": 77.0100, "observed_speed_kmh": 40.0}
    ])

    matched_df, report = matcher.match_dataframe(df)

    assert report.matched_observations == 0
    assert report.unmatched_observations == 1
    assert report.match_rate_pct == 0.0

    matched_row = matched_df.iloc[0]
    assert bool(matched_row["is_matched"]) is False
    assert matched_row["matched_edge_id"] is None
