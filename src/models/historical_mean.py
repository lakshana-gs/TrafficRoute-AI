"""Historical Time-of-Day baseline for traffic speed prediction."""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from src.evaluation.metrics import evaluate_horizons


class HistoricalTimeOfDayBaseline:
    """
    Historical Time-of-Day Mean Baseline.
    
    Computes the average observed speed for each sensor and (day_of_week, 5-minute time slot)
    strictly from the training dataset.
    
    Zero-Leakage Guarantee:
    - Statistics are learned strictly from train_data (where speed > null_val).
    - Validation and test splits are never included in historical mean calculation.
    """
    def __init__(
        self,
        horizon_steps: int = 12,
        null_val: float = 0.0,
        slots_per_day: int = 288,
    ):
        self.horizon_steps = horizon_steps
        self.null_val = null_val
        self.slots_per_day = slots_per_day
        self.is_fitted: bool = False
        
        # Shape: (7 days, 288 slots, N sensors)
        self.profile_table: Optional[np.ndarray] = None
        # Fallback sensor mean: Shape: (N sensors,)
        self.sensor_mean: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    def fit(
        self,
        train_data: np.ndarray,
        train_timestamps: pd.DatetimeIndex,
    ) -> "HistoricalTimeOfDayBaseline":
        """
        Fit historical time-of-day profiles strictly on training data.
        
        Args:
            train_data: (T_train, N) array of training speeds
            train_timestamps: DatetimeIndex of length T_train
        """
        T_train, N = train_data.shape
        if len(train_timestamps) != T_train:
            raise ValueError(f"Timestamps length {len(train_timestamps)} != data rows {T_train}")

        # Compute sensor global means strictly on valid training cells
        valid_mask = (train_data > self.null_val) & np.isfinite(train_data)
        self.sensor_mean = np.zeros(N, dtype=np.float32)
        for s in range(N):
            s_valid = train_data[valid_mask[:, s], s]
            self.sensor_mean[s] = float(np.mean(s_valid)) if len(s_valid) > 0 else 0.0
            
        all_valid = train_data[valid_mask]
        self.global_mean = float(np.mean(all_valid)) if len(all_valid) > 0 else 0.0
        
        # Build 7 x 288 x N profile table
        self.profile_table = np.zeros((7, self.slots_per_day, N), dtype=np.float32)
        
        day_of_weeks = train_timestamps.dayofweek.values
        slot_indices = (train_timestamps.hour.values * 60 + train_timestamps.minute.values) // 5
        
        for d in range(7):
            d_mask = (day_of_weeks == d)
            for slot in range(self.slots_per_day):
                slot_mask = d_mask & (slot_indices == slot)
                if not np.any(slot_mask):
                    # Fallback to sensor overall mean if day-slot combination was not seen
                    self.profile_table[d, slot, :] = self.sensor_mean
                    continue
                
                sub_data = train_data[slot_mask, :]
                sub_valid = valid_mask[slot_mask, :]
                
                for s in range(N):
                    valid_vals = sub_data[sub_valid[:, s], s]
                    if len(valid_vals) > 0:
                        self.profile_table[d, slot, s] = float(np.mean(valid_vals))
                    else:
                        self.profile_table[d, slot, s] = self.sensor_mean[s]

        self.is_fitted = True
        return self

    def predict_for_timestamps(
        self,
        target_timestamps: pd.DatetimeIndex,
        history_steps: int = 12,
    ) -> np.ndarray:
        """
        Generate predictions for all windows in a dataset given its target timestamps.
        
        Args:
            target_timestamps: DatetimeIndex of the original partition (length T_part)
            history_steps: Number of past steps in window (default: 12)
            
        Returns:
            Y_hat array of shape (num_windows, horizon_steps, N, 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Baseline must be fit before calling predict.")
            
        T_part = len(target_timestamps)
        num_windows = T_part - history_steps - self.horizon_steps + 1
        N = self.profile_table.shape[2]
        
        Y_hat = np.zeros((num_windows, self.horizon_steps, N, 1), dtype=np.float32)
        
        day_of_weeks = target_timestamps.dayofweek.values
        slot_indices = (target_timestamps.hour.values * 60 + target_timestamps.minute.values) // 5
        
        for w in range(num_windows):
            for h in range(self.horizon_steps):
                t_idx = w + history_steps + h
                d = day_of_weeks[t_idx]
                slot = slot_indices[t_idx]
                Y_hat[w, h, :, 0] = self.profile_table[d, slot, :]
                
        return Y_hat

    def evaluate(
        self,
        target_timestamps: pd.DatetimeIndex,
        Y_true: np.ndarray,
        history_steps: int = 12,
        step_minutes: int = 5,
        null_val: float = 0.0,
        mape_threshold: float = 8.0,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Generate historical mean predictions and compute masked evaluation metrics."""
        Y_hat = self.predict_for_timestamps(target_timestamps, history_steps=history_steps)
        return evaluate_horizons(
            y_true=Y_true,
            y_pred=Y_hat,
            step_minutes=step_minutes,
            null_val=null_val,
            mape_threshold=mape_threshold,
        )
