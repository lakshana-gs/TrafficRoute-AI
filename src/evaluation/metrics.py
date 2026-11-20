"""Evaluation metrics with strict masking for traffic speed prediction."""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def masked_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_val: float = 0.0,
    eps: float = 1e-5,
) -> float:
    """
    Calculate Masked Mean Absolute Error (MAE).
    
    Ignores entries where y_true is <= null_val or NaN/inf.
    Formula: MAE = sum(|y_true - y_pred|) / M, for all valid entries.
    """
    mask = (y_true > null_val) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return 0.0
    loss = np.abs(y_true[mask] - y_pred[mask])
    return float(np.mean(loss))


def masked_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_val: float = 0.0,
) -> float:
    """
    Calculate Masked Root Mean Squared Error (RMSE).
    
    Ignores entries where y_true is <= null_val or NaN/inf.
    Formula: RMSE = sqrt(mean((y_true - y_pred)^2)), for all valid entries.
    """
    mask = (y_true > null_val) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return 0.0
    loss = (y_true[mask] - y_pred[mask]) ** 2
    return float(np.sqrt(np.mean(loss)))


def masked_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_threshold: float = 8.0,
) -> float:
    """
    Calculate Masked Mean Absolute Percentage Error (MAPE).
    
    To prevent division by zero or extreme skew from near-zero sensor dropouts,
    only speeds above min_threshold (default 8.0 km/h approx 5.0 mph) are evaluated.
    Formula: MAPE = (100 / M) * sum(|y_true - y_pred| / y_true) for y_true >= min_threshold.
    """
    mask = (y_true >= min_threshold) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return 0.0
    loss = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(loss) * 100.0)


def evaluate_horizons(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step_minutes: int = 5,
    null_val: float = 0.0,
    mape_threshold: float = 8.0,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate multi-step predictions across discrete horizons (e.g. 5, 15, 30, 45, 60 min).
    
    Args:
        y_true: Array of shape (B, H, N) or (B, H, N, 1)
        y_pred: Array of shape (B, H, N) or (B, H, N, 1)
        step_minutes: Minutes per horizon step (default: 5)
        null_val: Threshold below which values are treated as missing
        mape_threshold: Speed threshold for MAPE evaluation
        
    Returns:
        Dictionary with overall metrics and per-horizon breakdowns.
    """
    if y_true.ndim == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]
        
    H = y_true.shape[1]
    overall_mae = masked_mae(y_true, y_pred, null_val=null_val)
    overall_rmse = masked_rmse(y_true, y_pred, null_val=null_val)
    overall_mape = masked_mape(y_true, y_pred, min_threshold=mape_threshold)
    
    horizon_results: Dict[str, Dict[str, float]] = {}
    for h in range(H):
        h_min = (h + 1) * step_minutes
        h_true = y_true[:, h, :]
        h_pred = y_pred[:, h, :]
        
        h_mae = masked_mae(h_true, h_pred, null_val=null_val)
        h_rmse = masked_rmse(h_true, h_pred, null_val=null_val)
        h_mape = masked_mape(h_true, h_pred, min_threshold=mape_threshold)
        
        horizon_results[f"{h_min}_min"] = {
            "mae": round(h_mae, 4),
            "rmse": round(h_rmse, 4),
            "mape": round(h_mape, 4),
        }
        
    return {
        "overall": {
            "mae": round(overall_mae, 4),
            "rmse": round(overall_rmse, 4),
            "mape": round(overall_mape, 4),
        },
        "horizons": horizon_results,
    }
