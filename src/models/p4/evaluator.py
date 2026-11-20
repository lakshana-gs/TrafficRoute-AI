"""Comprehensive multi-step evaluator, sensor-level analysis, and baseline comparison for P4."""

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_horizons, masked_mae, masked_mape, masked_rmse
from src.models.preprocessing import TrafficStandardScaler


class SensorMetrics(NamedTuple):
    """Container for per-sensor accuracy statistics."""
    sensor_id: str
    sensor_index: int
    mae: float
    rmse: float
    mape: float
    valid_count: int
    masked_count: int


class P4EvaluationResults(NamedTuple):
    """Container for full P4 model evaluation."""
    model_name: str
    overall_mae: float
    overall_rmse: float
    overall_mape: float
    horizons: Dict[str, Dict[str, float]]
    sensor_summary: Dict[str, Any]
    missing_data_summary: Dict[str, Any]


def compute_missing_data_statistics(
    y_true: np.ndarray,
    null_val: float = 0.0,
) -> Dict[str, Any]:
    """
    Analyze missing/masked observation counts and percentages across targets.
    
    Args:
        y_true: Ground truth target array of shape (B, H, N) or (B, H, N, 1)
        null_val: Threshold below which observations are treated as missing
    """
    if y_true.ndim == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]

    B, H, N = y_true.shape
    total_observations = int(B * H * N)
    
    valid_mask = (y_true > null_val) & np.isfinite(y_true)
    valid_count = int(np.count_nonzero(valid_mask))
    masked_count = int(total_observations - valid_count)
    masked_percent = float((masked_count / total_observations) * 100.0)

    # Per-horizon missing percentage
    horizon_missing = {}
    for h in range(H):
        h_mask = valid_mask[:, h, :]
        h_valid = int(np.count_nonzero(h_mask))
        h_total = int(B * N)
        h_masked = h_total - h_valid
        horizon_missing[f"step_{(h+1)*5}_min"] = {
            "valid_count": h_valid,
            "masked_count": h_masked,
            "masked_percent": round((h_masked / h_total) * 100.0, 4),
        }

    # Per-sensor missing percentage
    sensor_valid = np.sum(valid_mask, axis=(0, 1))  # (N,)
    sensor_total = B * H
    sensor_masked_pcts = ((sensor_total - sensor_valid) / sensor_total) * 100.0

    return {
        "total_target_observations": total_observations,
        "valid_observations": valid_count,
        "masked_observations": masked_count,
        "masked_percentage": round(masked_percent, 4),
        "horizon_missing_breakdown": horizon_missing,
        "sensor_missing_stats": {
            "mean_masked_percent": round(float(np.mean(sensor_masked_pcts)), 4),
            "median_masked_percent": round(float(np.median(sensor_masked_pcts)), 4),
            "min_masked_percent": round(float(np.min(sensor_masked_pcts)), 4),
            "max_masked_percent": round(float(np.max(sensor_masked_pcts)), 4),
        },
    }


def compute_sensor_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_ids: List[str],
    null_val: float = 0.0,
    mape_threshold: float = 8.0,
) -> Dict[str, Any]:
    """
    Calculate granular performance across all 207 sensors.
    
    DISCLAIMER: Sensor IDs correspond to Los Angeles County loop detectors (METR-LA),
    NOT Tamil Nadu road segments.
    """
    if y_true.ndim == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    B, H, N = y_true.shape
    sensor_records: List[SensorMetrics] = []

    for n in range(N):
        s_true = y_true[:, :, n].flatten()
        s_pred = y_pred[:, :, n].flatten()
        
        valid_mask = (s_true > null_val) & np.isfinite(s_true)
        valid_cnt = int(np.count_nonzero(valid_mask))
        masked_cnt = int(len(s_true) - valid_cnt)

        s_mae = masked_mae(s_true, s_pred, null_val=null_val)
        s_rmse = masked_rmse(s_true, s_pred, null_val=null_val)
        s_mape = masked_mape(s_true, s_pred, min_threshold=mape_threshold)

        sensor_records.append(
            SensorMetrics(
                sensor_id=sensor_ids[n] if n < len(sensor_ids) else f"sensor_{n}",
                sensor_index=n,
                mae=round(s_mae, 4),
                rmse=round(s_rmse, 4),
                mape=round(s_mape, 4),
                valid_count=valid_cnt,
                masked_count=masked_cnt,
            )
        )

    mae_list = [r.mae for r in sensor_records]
    sorted_by_mae = sorted(sensor_records, key=lambda x: x.mae)

    best_5 = [
        {"sensor_id": r.sensor_id, "index": r.sensor_index, "mae": r.mae, "rmse": r.rmse}
        for r in sorted_by_mae[:5]
    ]
    worst_5 = [
        {"sensor_id": r.sensor_id, "index": r.sensor_index, "mae": r.mae, "rmse": r.rmse}
        for r in sorted_by_mae[-5:][::-1]
    ]

    return {
        "num_sensors": N,
        "mean_sensor_mae": round(float(np.mean(mae_list)), 4),
        "median_sensor_mae": round(float(np.median(mae_list)), 4),
        "min_sensor_mae": round(float(np.min(mae_list)), 4),
        "max_sensor_mae": round(float(np.max(mae_list)), 4),
        "std_sensor_mae": round(float(np.std(mae_list)), 4),
        "best_performing_sensors": best_5,
        "worst_performing_sensors": worst_5,
        "sensor_domain_note": (
            "METR-LA loop detectors in Los Angeles County. "
            "These IDs do NOT represent Tamil Nadu or Coimbatore road segments."
        ),
    }


class P4Evaluator:
    """Evaluation manager for generating model predictions, horizons, and comparisons."""

    @staticmethod
    def predict_loader(
        model: nn.Module,
        data_loader: DataLoader,
        scaler: TrafficStandardScaler,
        supports: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference across DataLoader, unscale predictions to physical km/h speeds.
        
        Returns:
            Tuple of (y_true_all, y_pred_all), both shape (B_total, 12, 207, 1) in km/h.
        """
        model.eval()
        if device is None:
            device = next(model.parameters()).device

        y_true_list = []
        y_pred_list = []

        if supports is not None:
            supports_dev = [s.to(device) for s in supports]
        else:
            supports_dev = None

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                
                if supports_dev is not None:
                    out_scaled = model(x_batch, supports_dev)
                else:
                    out_scaled = model(x_batch)

                out_unscaled = out_scaled * scaler.std + scaler.mean
                
                y_true_list.append(y_batch.cpu().numpy())
                y_pred_list.append(out_unscaled.cpu().numpy())

        y_true_all = np.concatenate(y_true_list, axis=0)
        y_pred_all = np.concatenate(y_pred_list, axis=0)
        return y_true_all, y_pred_all

    @classmethod
    def evaluate_model(
        cls,
        model: nn.Module,
        data_loader: DataLoader,
        scaler: TrafficStandardScaler,
        sensor_ids: List[str],
        model_name: str = "DCRNN",
        supports: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[P4EvaluationResults, np.ndarray, np.ndarray]:
        """
        Perform complete multi-horizon and sensor-level evaluation.
        """
        y_true, y_pred = cls.predict_loader(
            model=model,
            data_loader=data_loader,
            scaler=scaler,
            supports=supports,
            device=device,
        )

        horizon_eval = evaluate_horizons(y_true, y_pred, step_minutes=5, null_val=0.0)
        sensor_summary = compute_sensor_level_metrics(y_true, y_pred, sensor_ids=sensor_ids)
        missing_summary = compute_missing_data_statistics(y_true, null_val=0.0)

        results = P4EvaluationResults(
            model_name=model_name,
            overall_mae=float(horizon_eval["overall"]["mae"]),
            overall_rmse=float(horizon_eval["overall"]["rmse"]),
            overall_mape=float(horizon_eval["overall"]["mape"]),
            horizons=horizon_eval["horizons"],
            sensor_summary=sensor_summary,
            missing_data_summary=missing_summary,
        )

        return results, y_true, y_pred
