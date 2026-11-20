"""Persistence baseline for traffic speed prediction."""

from typing import Dict, Union
import numpy as np
from src.evaluation.metrics import evaluate_horizons


class PersistenceBaseline:
    """
    Persistence baseline: carries the latest observed speed forward across all future horizons.
    
    Formula:
        y_hat(t + h) = x(t), for all h in [1, H]
    """
    def __init__(self, horizon_steps: int = 12):
        self.horizon_steps = horizon_steps

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict future speeds by repeating the latest known speed (last time step of history).
        
        Args:
            X: Input tensor of shape (B, T_in, N, 1) or (B, T_in, N)
            
        Returns:
            Y_hat of shape (B, H, N, 1)
        """
        if X.ndim == 4:
            latest_speed = X[:, -1, :, :]  # Shape: (B, N, 1)
            # Tile across the horizon dimension
            Y_hat = np.tile(np.expand_dims(latest_speed, axis=1), (1, self.horizon_steps, 1, 1))
        elif X.ndim == 3:
            latest_speed = X[:, -1, :]  # Shape: (B, N)
            Y_hat = np.tile(np.expand_dims(latest_speed, axis=1), (1, self.horizon_steps, 1))
        else:
            raise ValueError(f"Unsupported X dimensions: {X.shape}")

        return Y_hat

    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        step_minutes: int = 5,
        null_val: float = 0.0,
        mape_threshold: float = 8.0,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Generate persistence predictions and compute masked metrics."""
        Y_hat = self.predict(X)
        return evaluate_horizons(
            y_true=Y,
            y_pred=Y_hat,
            step_minutes=step_minutes,
            null_val=null_val,
            mape_threshold=mape_threshold,
        )
