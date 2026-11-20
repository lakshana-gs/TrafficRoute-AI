"""End-to-end integration tests for P4 pipeline components and inference from checkpoint."""

from pathlib import Path
import pytest
import torch

from src.models.p4.dcrnn import DCRNNModel
from src.models.p4.diffusion import build_diffusion_supports_from_pkl
from src.models.p4.evaluator import P4Evaluator
from src.models.p4.dataset import PyTorchWindowDataset
from src.models.preprocessing import (
    TrafficStandardScaler,
    chronological_split,
    load_raw_metr_la,
    prepare_windowed_dataset,
)
from torch.utils.data import DataLoader


def test_end_to_end_test_set_inference():
    """Verify that a trained/initialized DCRNN model can run full inference on test partition."""
    speed_mx, timestamps, sensor_ids = load_raw_metr_la()
    split = chronological_split(speed_mx, timestamps, sensor_ids)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)
    scaler = TrafficStandardScaler.load_json("data/traffic/processed/metr_la/scaler.json")
    supports = build_diffusion_supports_from_pkl("data/traffic/raw/metr_la/adj_mx.pkl", k=2)

    # Mini test loader with 16 samples
    test_ds = PyTorchWindowDataset(windows.x_test[:16], windows.y_test[:16], scaler=scaler)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = DCRNNModel(in_features=1, out_features=1, hidden_dim=16, num_layers=1, num_supports=5, horizon=12)
    results, y_true, y_pred = P4Evaluator.evaluate_model(
        model=model,
        data_loader=test_loader,
        scaler=scaler,
        sensor_ids=sensor_ids,
        model_name="DCRNN-Mini",
        supports=supports,
    )

    assert results.overall_mae > 0.0
    assert results.overall_rmse > 0.0
    assert len(results.horizons) == 12
    assert y_pred.shape == (16, 12, 207, 1)
    assert y_true.shape == (16, 12, 207, 1)
