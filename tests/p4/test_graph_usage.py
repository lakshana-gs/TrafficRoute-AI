"""Tests proving that spatial graph topology actively influences model predictions."""

import pytest
import torch

from src.models.p4.dcrnn import DCRNNModel
from src.models.p4.diffusion import compute_diffusion_supports
from src.models.preprocessing import load_sensor_graph


def test_altering_adjacency_matrix_changes_model_output():
    """
    CRITICAL PROOF OF SPATIAL GRAPH USAGE:
    
    Pass identical input speed tensor X through the DCRNN model with:
    1. Authentic METR-LA diffusion supports
    2. Permuted / altered diffusion supports
    
    Verify that the model outputs differ significantly, proving that the model
    actively computes over the sensor graph topology rather than ignoring it.
    """
    _, _, raw_adj = load_sensor_graph("data/traffic/raw/metr_la/adj_mx.pkl")
    N = raw_adj.shape[0]

    # 1. Authentic diffusion supports
    supports_real_np = compute_diffusion_supports(raw_adj, k=2)
    supports_real = [torch.from_numpy(s).float() for s in supports_real_np]

    # 2. Controlled altered / permuted supports (e.g. reverse graph or random permutation)
    perm = torch.randperm(N)
    adj_perm = raw_adj[perm.numpy(), :][:, perm.numpy()]
    supports_perm_np = compute_diffusion_supports(adj_perm, k=2)
    supports_perm = [torch.from_numpy(s).float() for s in supports_perm_np]

    # Initialize model with non-trivial weights
    torch.manual_seed(42)
    model = DCRNNModel(in_features=1, out_features=1, hidden_dim=32, num_layers=2, num_supports=5, horizon=12)
    model.eval()

    x_input = torch.randn(4, 12, N, 1)

    with torch.no_grad():
        out_real = model(x_input, supports_real)
        out_perm = model(x_input, supports_perm)

    # 3. Output tensors MUST differ
    diff = torch.abs(out_real - out_perm).mean().item()
    assert diff > 1e-4, f"Altered graph produced virtually identical output (diff={diff}). Graph is not used!"
    print(f"\nSpatial graph usage verified: Mean absolute output difference = {diff:.6f}")
