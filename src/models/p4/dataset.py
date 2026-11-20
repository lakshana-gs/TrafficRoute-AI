"""PyTorch Dataset, DataLoaders, and Masked Loss Functions for Traffic Forecasting."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.preprocessing import TrafficStandardScaler, WindowedDataset


def masked_mae_loss_tensor(
    y_pred_unscaled: torch.Tensor,
    y_true: torch.Tensor,
    null_val: float = 0.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Calculate Masked Mean Absolute Error (MAE) loss on PyTorch tensors.
    
    Zero / negative sensor dropouts (y_true <= null_val) are masked out so
    invalid sensor dropouts do not contribute to gradients or loss values.
    
    Args:
        y_pred_unscaled: Predicted speed in physical units (km/h), shape (B, T, N, 1)
        y_true: Ground truth speed in physical units (km/h), shape (B, T, N, 1)
        null_val: Threshold below which observations are treated as missing/invalid
        eps: Epsilon to prevent division by zero
        
    Returns:
        Scalar loss tensor with valid autograd backward graph.
    """
    mask = (y_true > null_val) & torch.isfinite(y_true) & torch.isfinite(y_pred_unscaled)
    mask = mask.float()
    
    # Normalize mask by valid fraction to ensure stable gradient scale
    mask_mean = torch.mean(mask)
    mask /= torch.clamp(mask_mean, min=eps)
    
    loss = torch.abs(y_pred_unscaled - y_true) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss(nn.Module):
    """Masked MAE Loss Module for PyTorch models."""
    def __init__(self, null_val: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.null_val = null_val
        self.eps = eps

    def forward(self, y_pred_unscaled: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return masked_mae_loss_tensor(
            y_pred_unscaled,
            y_true,
            null_val=self.null_val,
            eps=self.eps,
        )


class PyTorchWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding-window traffic speed arrays.
    
    Input X is standardized using the training-only scaler.
    Target Y is kept in physical units (km/h) for masked physical loss calculation.
    """
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        scaler: Optional[TrafficStandardScaler] = None,
    ):
        if len(x_data) != len(y_data):
            raise ValueError(f"X length ({len(x_data)}) != Y length ({len(y_data)}).")

        if scaler is not None:
            self.x = torch.from_numpy(scaler.transform(x_data)).float()
        else:
            self.x = torch.from_numpy(x_data).float()

        # Targets stored in physical units (km/h)
        self.y = torch.from_numpy(y_data).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_traffic_dataloaders(
    windows: WindowedDataset,
    scaler: TrafficStandardScaler,
    batch_size: int = 64,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for Train, Validation, and Test partitions.
    
    Zero-Leakage Assurance:
        - X is normalized strictly with the training-only scaler.
        - Validation and Test loaders do NOT shuffle.
    """
    train_dataset = PyTorchWindowDataset(windows.x_train, windows.y_train, scaler=scaler)
    val_dataset = PyTorchWindowDataset(windows.x_val, windows.y_val, scaler=scaler)
    test_dataset = PyTorchWindowDataset(windows.x_test, windows.y_test, scaler=scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
