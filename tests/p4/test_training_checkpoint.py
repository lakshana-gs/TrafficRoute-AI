"""Tests for model checkpointing, metadata persistence, and reloading."""

from pathlib import Path
import json
import pytest
import torch

from src.models.p4.dcrnn import DCRNNModel


def test_checkpoint_save_and_reload(tmp_path: Path):
    """Verify state_dict can be saved, reloaded into a new model, and produce identical outputs."""
    N = 207
    supports = [torch.eye(N) for _ in range(5)]
    model_orig = DCRNNModel(in_features=1, out_features=1, hidden_dim=16, num_layers=1, num_supports=5, horizon=12)

    ckpt_path = tmp_path / "p4_test_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model_orig.state_dict(),
            "val_loss": 5.4321,
            "hidden_dim": 16,
        },
        ckpt_path,
    )

    assert ckpt_path.exists()

    # Load into fresh instance
    model_new = DCRNNModel(in_features=1, out_features=1, hidden_dim=16, num_layers=1, num_supports=5, horizon=12)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_new.load_state_dict(checkpoint["model_state_dict"])

    model_orig.eval()
    model_new.eval()

    x = torch.randn(2, 12, N, 1)
    with torch.no_grad():
        out_orig = model_orig(x, supports)
        out_new = model_new(x, supports)

    torch.testing.assert_close(out_orig, out_new)
