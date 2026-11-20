"""Tests for METR-LA sensor graph loading and schema validation."""

from pathlib import Path
import numpy as np
import pytest

from src.models.preprocessing import load_sensor_graph


def test_graph_loading_and_types():
    """Verify that adj_mx.pkl loads sensor IDs, mapping dictionary, and numpy array."""
    pkl_path = Path("data/traffic/raw/metr_la/adj_mx.pkl")
    assert pkl_path.exists(), f"Graph file missing: {pkl_path}"

    sensor_ids, sensor_to_ind, adj_mx = load_sensor_graph(pkl_path)

    assert isinstance(sensor_ids, list)
    assert len(sensor_ids) == 207
    assert isinstance(sensor_to_ind, dict)
    assert len(sensor_to_ind) == 207
    assert isinstance(adj_mx, np.ndarray)
    assert adj_mx.shape == (207, 207)

    # Verify sensor index mapping bijection
    for idx, sid in enumerate(sensor_ids):
        assert sensor_to_ind[sid] == idx
