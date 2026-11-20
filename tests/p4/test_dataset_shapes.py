"""Tests for sliding window shapes and PyTorch dataset tensor shapes."""

import numpy as np
import pytest
import torch

from src.models.p4.dataset import PyTorchWindowDataset, create_traffic_dataloaders
from src.models.preprocessing import (
    TrafficStandardScaler,
    chronological_split,
    load_raw_metr_la,
    prepare_windowed_dataset,
)


def test_windowed_dataset_tensor_shapes():
    """Verify sliding window tensor shapes: (B, 12, 207, 1) across all partitions."""
    speed_mx, timestamps, sensor_ids = load_raw_metr_la()
    split = chronological_split(speed_mx, timestamps, sensor_ids)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)

    assert windows.x_train.shape == (23967, 12, 207, 1)
    assert windows.y_train.shape == (23967, 12, 207, 1)
    assert windows.x_val.shape == (5117, 12, 207, 1)
    assert windows.y_val.shape == (5117, 12, 207, 1)
    assert windows.x_test.shape == (5119, 12, 207, 1)
    assert windows.y_test.shape == (5119, 12, 207, 1)


def test_pytorch_dataloader_batch_shapes():
    """Verify PyTorch DataLoader yields batches of exact dimensions (B, 12, 207, 1)."""
    speed_mx, timestamps, sensor_ids = load_raw_metr_la()
    split = chronological_split(speed_mx, timestamps, sensor_ids)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)
    scaler = TrafficStandardScaler.load_json("data/traffic/processed/metr_la/scaler.json")

    train_loader, val_loader, test_loader = create_traffic_dataloaders(
        windows=windows,
        scaler=scaler,
        batch_size=32,
        shuffle_train=False,
    )

    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (32, 12, 207, 1)
    assert y_batch.shape == (32, 12, 207, 1)
    assert isinstance(x_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)
