"""Tests for DCRNN and Temporal-only forward propagation and gradient flow."""

import pytest
import torch

from src.models.p4.dcrnn import DCRNNModel, DCGRUCell, DiffusionGraphConv
from src.models.p4.temporal_only import TemporalOnlyGRU


def test_dcrnn_forward_and_backward():
    """Verify DCRNN processes input (B, 12, 207, 1) and propagates gradients."""
    B, T_in, N, C = 4, 12, 207, 1
    supports = [torch.eye(N) for _ in range(5)]
    model = DCRNNModel(in_features=1, out_features=1, hidden_dim=32, num_layers=2, num_supports=5, horizon=12)

    x = torch.randn(B, T_in, N, C, requires_grad=True)
    out = model(x, supports)

    assert out.shape == (B, 12, N, 1)
    loss = torch.sum(out)
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_temporal_only_forward_and_backward():
    """Verify Temporal-Only GRU processes input (B, 12, 207, 1) and propagates gradients."""
    B, T_in, N, C = 4, 12, 207, 1
    model = TemporalOnlyGRU(in_features=1, out_features=1, hidden_dim=32, num_layers=2, horizon=12)

    x = torch.randn(B, T_in, N, C, requires_grad=True)
    out = model(x)

    assert out.shape == (B, 12, N, 1)
    loss = torch.sum(out)
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
