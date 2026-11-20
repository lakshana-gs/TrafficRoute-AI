"""Unit tests for NetworkValidator."""

import pytest
import networkx as nx

from src.network.graph import RoadNetworkGraph
from src.network.provenance import NetworkProvenance
from src.network.validation import NetworkValidator


def test_network_validator_metrics():
    g = nx.MultiDiGraph()
    # Simple connected circuit
    g.add_node(1, latitude=11.00, longitude=76.95)
    g.add_node(2, latitude=11.05, longitude=77.00)
    g.add_node(3, latitude=11.00, longitude=77.05)

    g.add_edge(1, 2, key=0, highway="trunk", name="NH-544", length_m=5000.0, free_flow_speed_kmh=80.0, oneway=True)
    g.add_edge(2, 3, key=0, highway="trunk", name="NH-544", length_m=5000.0, free_flow_speed_kmh=80.0, oneway=True)
    g.add_edge(3, 1, key=0, highway="primary", name="Return Road", length_m=7000.0, free_flow_speed_kmh=60.0, oneway=True)

    prov = NetworkProvenance(
        dataset_name="ValidatorTest",
        extraction_method="UnitTest",
        geographic_boundary={"north": 11.1, "south": 10.9, "east": 77.1, "west": 76.9},
    )
    rg = RoadNetworkGraph(graph=g, provenance=prov)

    validator = NetworkValidator(rg)
    val = validator.run_full_validation()

    assert val["summary"]["node_count"] == 3
    assert val["summary"]["edge_count"] == 3
    assert val["connectivity"]["weakly_connected_components"] == 1
    assert val["connectivity"]["strongly_connected_components"] == 1
    assert val["directionality"]["oneway_edges"] == 3
    assert val["geometry"]["invalid_geometries"] == 0
    assert val["geometry"]["zero_length_edges"] == 0
    assert "trunk" in val["highway_classification_distribution"]

    report_md = validator.generate_markdown_report()
    assert "# TrafficRoute AI — P1 Road Network Validation Report" in report_md
    assert "Total Vertices (Nodes):" in report_md
