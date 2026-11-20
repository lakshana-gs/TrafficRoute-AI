"""Simple Machine Learning baseline using XGBoost on tabular lag features."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
from src.evaluation.metrics import evaluate_horizons, masked_mae, masked_rmse, masked_mape


class TabularLagFeatureExtractor:
    """
    Extracts tabular auto-regressive lag and temporal features from sliding windows.
    
    Features extracted per sensor:
    - speed_t (latest observed speed at horizon start)
    - lag_1 (5 mins ago), lag_2 (10 mins ago), lag_3 (15 mins ago)
    - lag_6 (30 mins ago), lag_12 (60 mins ago)
    - window_mean (mean of 12 historical steps)
    - window_std (std of 12 historical steps)
    - hour (0-23)
    - day_of_week (0-6)
    - is_weekend (0 or 1)
    """
    @staticmethod
    def extract_features(
        X: np.ndarray,
        reference_timestamps: pd.DatetimeIndex,
        history_steps: int = 12,
        horizon_steps: int = 12,
        sensor_subset: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Extract feature matrix of shape (num_samples, num_features).
        
        Args:
            X: Input tensor of shape (B, 12, N, 1) or (B, 12, N)
            reference_timestamps: DatetimeIndex of length T_part
            history_steps: Past steps in window (12)
            horizon_steps: Future steps in window (12)
            sensor_subset: Optional list of sensor indices to extract
            
        Returns:
            Features array of shape (B * N_sub, num_features)
        """
        if X.ndim == 4:
            X = X[..., 0]
            
        B, T_in, N = X.shape
        sensors = sensor_subset if sensor_subset is not None else list(range(N))
        N_sub = len(sensors)
        
        num_windows = B
        # Reference timestamp for each window (the timestamp at the end of history / start of prediction)
        ref_ts = [reference_timestamps[w + history_steps - 1] for w in range(num_windows)]
        hours = np.array([ts.hour for ts in ref_ts], dtype=np.float32)
        dows = np.array([ts.dayofweek for ts in ref_ts], dtype=np.float32)
        weekends = np.array([1.0 if ts.dayofweek in (5, 6) else 0.0 for ts in ref_ts], dtype=np.float32)
        
        # Preallocate feature array: (B * N_sub, 10)
        feature_rows: List[np.ndarray] = []
        
        for s_idx in sensors:
            s_data = X[:, :, s_idx]  # Shape: (B, 12)
            
            speed_t = s_data[:, 11:12]
            lag_1 = s_data[:, 10:11]
            lag_2 = s_data[:, 9:10]
            lag_3 = s_data[:, 8:9]
            lag_6 = s_data[:, 5:6]
            lag_12 = s_data[:, 0:1]
            w_mean = np.mean(s_data, axis=1, keepdims=True)
            w_std = np.std(s_data, axis=1, keepdims=True)
            
            h_col = hours[:, None]
            dow_col = dows[:, None]
            
            s_features = np.hstack([
                speed_t,
                lag_1,
                lag_2,
                lag_3,
                lag_6,
                lag_12,
                w_mean,
                w_std,
                h_col,
                dow_col,
            ])
            feature_rows.append(s_features)
            
        return np.vstack(feature_rows)  # Shape: (N_sub * B, 10)


class XGBoostSpeedBaseline:
    """
    XGBoost Regression Baseline for Multi-Step Speed Prediction.
    
    Trains separate gradient-boosted trees for key forecast horizons (5, 15, 30, 45, 60 mins).
    
    Zero-Leakage Guarantee:
    - Trained strictly on X_train / Y_train.
    - Missing target cells (speed <= 0) are excluded from training loss.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        random_state: int = 42,
        horizon_steps: int = 12,
        null_val: float = 0.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.horizon_steps = horizon_steps
        self.null_val = null_val
        self.models: Dict[int, xgb.XGBRegressor] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        train_timestamps: pd.DatetimeIndex,
        sample_step_stride: int = 4,  # Subsample windows for fast, efficient training
        sensor_stride: int = 2,       # Representative sensor sampling for training
    ) -> "XGBoostSpeedBaseline":
        """
        Fit horizon-specific XGBoost regressors on training data.
        
        Args:
            X_train: (B_train, 12, N, 1) input array
            Y_train: (B_train, 12, N, 1) target array
            train_timestamps: DatetimeIndex of length T_train
            sample_step_stride: Stride over windows to speed up fitting
            sensor_stride: Stride over sensors for training set
        """
        if X_train.ndim == 4:
            X_train = X_train[..., 0]
        if Y_train.ndim == 4:
            Y_train = Y_train[..., 0]

        B, T_in, N = X_train.shape
        train_sensors = list(range(0, N, sensor_stride))
        sub_B_indices = list(range(0, B, sample_step_stride))
        
        X_sub = X_train[sub_B_indices]
        Y_sub = Y_train[sub_B_indices]
        sub_ts = train_timestamps
        
        print(f"Extracting features for XGBoost: {len(sub_B_indices)} windows across {len(train_sensors)} sensors...")
        feat_extractor = TabularLagFeatureExtractor()
        X_features = feat_extractor.extract_features(
            X_sub,
            sub_ts,
            history_steps=T_in,
            horizon_steps=self.horizon_steps,
            sensor_subset=train_sensors,
        )
        
        # Horizons to train: all 12 horizons (or key horizons interpolated)
        for h in range(self.horizon_steps):
            # Target for horizon h across all sampled sensors
            y_target_list: List[np.ndarray] = []
            for s_idx in train_sensors:
                y_s = Y_sub[:, h, s_idx]
                y_target_list.append(y_s)
            y_targets = np.concatenate(y_target_list)
            
            # Mask out missing target cells from training
            valid_train_mask = (y_targets > self.null_val) & np.isfinite(y_targets)
            X_valid = X_features[valid_train_mask]
            y_valid = y_targets[valid_train_mask]
            
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method="hist",
            )
            model.fit(X_valid, y_valid)
            self.models[h] = model
            print(f"  Fitted XGBoost horizon {h+1}/12 ({(h+1)*5} min ahead) on {len(y_valid):,} samples.")

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        reference_timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Generate multi-step predictions for all sensors in X.
        
        Args:
            X: (B, 12, N, 1) or (B, 12, N)
            reference_timestamps: DatetimeIndex of the partition
            
        Returns:
            Y_hat of shape (B, 12, N, 1)
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoost baseline must be fit before calling predict.")
            
        if X.ndim == 4:
            X = X[..., 0]
        B, T_in, N = X.shape
        
        feat_extractor = TabularLagFeatureExtractor()
        Y_hat = np.zeros((B, self.horizon_steps, N, 1), dtype=np.float32)
        
        for s_idx in range(N):
            s_features = feat_extractor.extract_features(
                X,
                reference_timestamps,
                history_steps=T_in,
                horizon_steps=self.horizon_steps,
                sensor_subset=[s_idx],
            )
            
            for h in range(self.horizon_steps):
                model = self.models[h]
                pred_h = model.predict(s_features)
                Y_hat[:, h, s_idx, 0] = pred_h
                
        return Y_hat

    def evaluate(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        reference_timestamps: pd.DatetimeIndex,
        step_minutes: int = 5,
        null_val: float = 0.0,
        mape_threshold: float = 8.0,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Generate XGBoost predictions and compute masked evaluation metrics."""
        Y_hat = self.predict(X, reference_timestamps)
        return evaluate_horizons(
            y_true=Y_true,
            y_pred=Y_hat,
            step_minutes=step_minutes,
            null_val=null_val,
            mape_threshold=mape_threshold,
        )
