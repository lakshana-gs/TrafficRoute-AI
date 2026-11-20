"""Preprocessing, chronological splitting, and window generation for METR-LA speed prediction."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import h5py
import numpy as np
import pandas as pd


MPH_TO_KMH: float = 1.609344


class ChronologicalSplit(NamedTuple):
    """Container for chronologically partitioned traffic tensors."""
    train_data: np.ndarray  # (T_train, N)
    val_data: np.ndarray    # (T_val, N)
    test_data: np.ndarray   # (T_test, N)
    train_timestamps: pd.DatetimeIndex
    val_timestamps: pd.DatetimeIndex
    test_timestamps: pd.DatetimeIndex
    sensor_ids: List[str]


class WindowedDataset(NamedTuple):
    """Container for sliding-window traffic speed arrays."""
    x_train: np.ndarray  # (B_train, T_in, N, 1)
    y_train: np.ndarray  # (B_train, T_out, N, 1)
    x_val: np.ndarray    # (B_val, T_in, N, 1)
    y_val: np.ndarray    # (B_val, T_out, N, 1)
    x_test: np.ndarray   # (B_test, T_in, N, 1)
    y_test: np.ndarray   # (B_test, T_out, N, 1)


class TrafficStandardScaler:
    """
    Standard Z-Score Scaler fit strictly on training data valid cells.
    
    Formula:
        z = (x - mean) / std, for x > null_val
    """
    def __init__(self, null_val: float = 0.0):
        self.null_val = null_val
        self.mean: float = 0.0
        self.std: float = 1.0
        self.is_fitted: bool = False

    def fit(self, train_data: np.ndarray) -> "TrafficStandardScaler":
        """Fit scaler strictly on positive/valid training cells."""
        valid_mask = (train_data > self.null_val) & np.isfinite(train_data)
        if not np.any(valid_mask):
            raise ValueError("No valid training cells found to fit scaler.")
        
        valid_values = train_data[valid_mask]
        self.mean = float(np.mean(valid_values))
        self.std = float(np.std(valid_values))
        if self.std < 1e-6:
            self.std = 1.0
        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply standardization; zero/missing cells remain zero."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fit before transform.")
        transformed = np.zeros_like(data, dtype=np.float32)
        valid_mask = (data > self.null_val) & np.isfinite(data)
        transformed[valid_mask] = (data[valid_mask] - self.mean) / self.std
        return transformed

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Reverse standardization back to original physical speed units (km/h)."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fit before inverse_transform.")
        return scaled_data * self.std + self.mean

    def save_json(self, output_path: Path | str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Save scaler parameters and metadata to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": "StandardScaler",
            "feature": "traffic_speed",
            "unit": "km/h",
            "training_mean": round(self.mean, 6),
            "training_std": round(self.std, 6),
            "null_val": self.null_val,
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "METR-LA (Train Split Only, Zero Leakage)",
            "extra_metadata": extra_meta or {},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, input_path: Path | str) -> "TrafficStandardScaler":
        """Load fitted scaler from JSON."""
        with open(input_path, "r") as f:
            meta = json.load(f)
        scaler = cls(null_val=meta.get("null_val", 0.0))
        scaler.mean = float(meta["training_mean"])
        scaler.std = float(meta["training_std"])
        scaler.is_fitted = True
        return scaler


def load_raw_metr_la(
    h5_path: Path | str = "data/traffic/raw/metr_la/metr-la.h5",
    convert_to_kmh: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Load raw speed matrix and timestamps from METR-LA HDF5 file.
    
    Args:
        h5_path: Path to metr-la.h5
        convert_to_kmh: If True, multiply speeds by 1.609344 (mph -> km/h).
        
    Returns:
        Tuple of (speed_matrix of shape (34272, 207), DatetimeIndex, list of sensor_ids).
    """
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw METR-LA H5 file not found at: {path}")

    with h5py.File(path, "r") as f:
        sensor_ids_raw = f["df/axis0"][:]
        sensor_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sensor_ids_raw]
        timestamps_raw = f["df/axis1"][:]
        speed_matrix = f["df/block0_values"][:].astype(np.float64)

    timestamps = pd.to_datetime(timestamps_raw)
    
    if convert_to_kmh:
        speed_matrix = speed_matrix * MPH_TO_KMH

    return speed_matrix, timestamps, sensor_ids


def load_sensor_graph(
    pkl_path: Path | str = "data/traffic/raw/metr_la/adj_mx.pkl",
) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """
    Load sensor IDs, mapping, and adjacency matrix from adj_mx.pkl.
    
    Returns:
        Tuple of (sensor_ids, sensor_id_to_ind, adj_mx of shape (207, 207)).
    """
    path = Path(pkl_path)
    if not path.exists():
        raise FileNotFoundError(f"Adjacency matrix file not found at: {path}")

    with open(path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    return list(sensor_ids), dict(sensor_id_to_ind), np.array(adj_mx, dtype=np.float32)


def chronological_split(
    speed_matrix: np.ndarray,
    timestamps: pd.DatetimeIndex,
    sensor_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> ChronologicalSplit:
    """
    Split traffic speed time series chronologically with zero shuffling.
    
    Args:
        speed_matrix: (T, N) speed array
        timestamps: DatetimeIndex of length T
        sensor_ids: List of N sensor IDs
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        
    Returns:
        ChronologicalSplit object containing train, val, and test partitions.
    """
    T, N = speed_matrix.shape
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    
    train_data = speed_matrix[:n_train]
    val_data = speed_matrix[n_train:n_train + n_val]
    test_data = speed_matrix[n_train + n_val:]
    
    train_ts = timestamps[:n_train]
    val_ts = timestamps[n_train:n_train + n_val]
    test_ts = timestamps[n_train + n_val:]
    
    return ChronologicalSplit(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_timestamps=train_ts,
        val_timestamps=val_ts,
        test_timestamps=test_ts,
        sensor_ids=sensor_ids,
    )


def generate_sliding_windows(
    data: np.ndarray,
    history_steps: int = 12,
    horizon_steps: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding windows for a single contiguous temporal partition.
    
    Ensures that for a partition of length T_part, windows are strictly contained
    within [0, T_part) so that no sample crosses partition boundaries.
    
    Args:
        data: (T_part, N) array
        history_steps: Number of past time steps (default: 12 = 60 mins)
        horizon_steps: Number of future time steps to predict (default: 12 = 60 mins)
        
    Returns:
        X of shape (num_windows, history_steps, N, 1),
        Y of shape (num_windows, horizon_steps, N, 1).
    """
    T_part, N = data.shape
    total_window_len = history_steps + horizon_steps
    if T_part < total_window_len:
        raise ValueError(f"Partition length {T_part} is too short for window {total_window_len}.")

    num_windows = T_part - total_window_len + 1
    X = np.zeros((num_windows, history_steps, N, 1), dtype=np.float32)
    Y = np.zeros((num_windows, horizon_steps, N, 1), dtype=np.float32)

    for i in range(num_windows):
        X[i, :, :, 0] = data[i : i + history_steps]
        Y[i, :, :, 0] = data[i + history_steps : i + total_window_len]

    return X, Y


def prepare_windowed_dataset(
    split: ChronologicalSplit,
    history_steps: int = 12,
    horizon_steps: int = 12,
) -> WindowedDataset:
    """
    Generate non-overlapping windowed datasets for train, val, and test splits.
    
    Zero-Leakage Guarantee:
    - Train windows are built strictly from split.train_data
    - Val windows are built strictly from split.val_data
    - Test windows are built strictly from split.test_data
    """
    x_train, y_train = generate_sliding_windows(split.train_data, history_steps, horizon_steps)
    x_val, y_val = generate_sliding_windows(split.val_data, history_steps, horizon_steps)
    x_test, y_test = generate_sliding_windows(split.test_data, history_steps, horizon_steps)

    return WindowedDataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
