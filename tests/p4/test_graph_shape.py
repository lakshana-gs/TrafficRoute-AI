"""Tests for METR-LA sensor graph dimensions, finiteness, and properties."""

import numpy as np
import pytest

from src.models.p4.diffusion import validate_sensor_graph
from src.models.preprocessing import load_sensor_graph


def test_sensor_graph_properties_and_finiteness():
    """Verify exact 207-node graph properties, finite status, and asymmetry."""
    sensor_ids, _, adj_mx = load_sensor_graph("data/traffic/raw/metr_la/adj_mx.pkl")
    props = validate_sensor_graph(adj_mx, sensor_ids)

    assert props.sensor_count == 207
    assert props.adjacency_shape == (207, 207)
    assert props.is_finite is True
    assert props.has_nan is False
    assert props.has_inf is False
    assert props.min_weight >= 0.0
    assert props.max_weight <= 1.0
    assert props.diagonal_min == 1.0
    assert props.diagonal_max == 1.0
    assert props.is_symmetric is False  # Directed graph structure
    assert props.nonzero_edges == 1722
    assert props.total_entries == 42849
    assert 0.03 <= props.density <= 0.05
