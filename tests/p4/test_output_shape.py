"""Tests for exact multi-horizon output shape conformity."""

import pytest
import torch

from src.models.p4.dcrnn import DCRNNModel


def test_dcrnn_output_shape_and_batch_sizes():
    """Verify DCRNN output maintains exact shape (B, horizon, 207, 1) across varying batch sizes."""
    N = 207
    supports = [torch.eye(N) for _ in range(5)]
    model = DCRNNModel(in_features=1, out_features=1, hidden_dim=16, num_layers=1, num_supports=5, horizon=12)

    for b_size in [1, 7, 16]:
        x = torch.randn(b_size, 12, N, 1)
        out = model(x, supports)
        assert out.shape == (b_size, 12, N, 1)
        assert torch.all(torch.isfinite(out))
