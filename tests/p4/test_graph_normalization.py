"""Tests for directed graph normalization and diffusion support matrices."""

from pathlib import Path
import numpy as np
import pytest

from src.models.p4.diffusion import (
    compute_diffusion_supports,
    normalize_directed_adjacency,
    save_normalized_supports,
)
from src.models.preprocessing import load_sensor_graph


def test_directed_graph_normalization_stochasticity():
    """Verify forward and backward random walk matrices are valid row-stochastic transitions."""
    _, _, adj_mx = load_sensor_graph("data/traffic/raw/metr_la/adj_mx.pkl")
    P_f, P_b = normalize_directed_adjacency(adj_mx)

    assert P_f.shape == (207, 207)
    assert P_b.shape == (207, 207)
    assert np.all(np.isfinite(P_f))
    assert np.all(np.isfinite(P_b))

    # All row sums should be approximately 1.0 (for connected components)
    row_sums_f = np.sum(P_f, axis=1)
    row_sums_b = np.sum(P_b, axis=1)
    np.testing.assert_allclose(row_sums_f, np.ones(207), atol=1e-5)
    np.testing.assert_allclose(row_sums_b, np.ones(207), atol=1e-5)


def test_diffusion_supports_order_k():
    """Verify diffusion polynomial supports construction for order K=2 (5 matrices)."""
    _, _, adj_mx = load_sensor_graph("data/traffic/raw/metr_la/adj_mx.pkl")
    supports = compute_diffusion_supports(adj_mx, k=2)

    assert len(supports) == 5
    for s in supports:
        assert s.shape == (207, 207)
        assert np.all(np.isfinite(s))
        assert np.min(s) >= 0.0

    # Support 0 is Identity
    np.testing.assert_allclose(supports[0], np.eye(207), atol=1e-6)


def test_save_and_reload_normalized_supports(tmp_path: Path):
    """Verify saving normalized supports to disk and verifying integrity."""
    _, _, adj_mx = load_sensor_graph("data/traffic/raw/metr_la/adj_mx.pkl")
    supports = compute_diffusion_supports(adj_mx, k=2)

    npz_path = tmp_path / "normalized_adj.npz"
    json_path = tmp_path / "normalized_adj_meta.json"
    save_normalized_supports(supports, output_npz_path=npz_path, metadata_json_path=json_path)

    assert npz_path.exists()
    assert json_path.exists()

    loaded = np.load(npz_path)
    assert len(loaded.files) == 5
    for i in range(5):
        np.testing.assert_allclose(loaded[f"support_{i}"], supports[i], atol=1e-6)
