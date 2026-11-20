"""Tests for masked loss computation, gradient masking, and dropout immunity."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import masked_mae, masked_mape, masked_rmse
from src.models.p4.dataset import MaskedMAELoss, masked_mae_loss_tensor


def test_numpy_masked_metrics_ignore_zeros_and_nans():
    """Verify numpy metric functions ignore zeros, negatives, and NaN values."""
    y_true = np.array([100.0, 0.0, -5.0, np.nan, 80.0])
    y_pred = np.array([90.0, 50.0, 50.0, 50.0, 70.0])

    # Valid points are index 0 (|100 - 90| = 10) and index 4 (|80 - 70| = 10)
    # Expected MAE = (10 + 10) / 2 = 10.0
    mae = masked_mae(y_true, y_pred, null_val=0.0)
    assert abs(mae - 10.0) < 1e-5

    rmse = masked_rmse(y_true, y_pred, null_val=0.0)
    assert abs(rmse - 10.0) < 1e-5

    # MAPE for 100.0 (10%) and 80.0 (12.5%) -> mean = 11.25%
    mape = masked_mape(y_true, y_pred, min_threshold=8.0)
    assert abs(mape - 11.25) < 1e-4


def test_pytorch_masked_mae_loss_zero_gradient_for_masked_targets():
    """Verify PyTorch MaskedMAELoss produces exactly zero gradient for dropped sensor observations."""
    y_pred_unscaled = torch.tensor([[[[90.0], [50.0]]]], requires_grad=True)
    # Sensor 0 is valid (100.0), Sensor 1 is dropped (0.0)
    y_true = torch.tensor([[[[100.0], [0.0]]]])

    loss_fn = MaskedMAELoss(null_val=0.0)
    loss = loss_fn(y_pred_unscaled, y_true)

    loss.backward()

    assert y_pred_unscaled.grad is not None
    # Gradient for valid sensor should be non-zero
    assert abs(float(y_pred_unscaled.grad[0, 0, 0, 0])) > 0.0
    # Gradient for dropped sensor MUST be exactly 0.0
    assert float(y_pred_unscaled.grad[0, 0, 1, 0]) == 0.0
