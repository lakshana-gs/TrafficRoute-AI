"""Deterministic trainer, early stopping, and checkpointing for P4 traffic forecasting."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.p4.dataset import masked_mae_loss_tensor
from src.models.preprocessing import TrafficStandardScaler


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for P4 model training."""
    model_type: str = "DCRNN"  # "DCRNN" or "TemporalOnly"
    hidden_dim: int = 64
    num_layers: int = 2
    diffusion_order_k: int = 2
    horizon: int = 12
    history: int = 12
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 8
    random_seed: int = 42
    clip_grad_norm: float = 5.0
    teacher_forcing_initial: float = 0.5
    teacher_forcing_decay: float = 0.95
    device: str = "cpu"


@dataclass
class EpochRecord:
    """Record of a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    elapsed_seconds: float
    improved: bool


@dataclass
class TrainingHistory:
    """Container for complete training trajectory."""
    model_name: str
    total_epochs: int
    best_epoch: int
    best_val_loss: float
    total_duration_seconds: float
    epoch_records: List[Dict[str, Any]] = field(default_factory=list)

    def save_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        temp_path.replace(path)

    @classmethod
    def load_json(cls, input_path: str | Path) -> "TrainingHistory":
        with open(input_path, "r") as f:
            data = json.load(f)
        return cls(
            model_name=data.get("model_name", "DCRNN"),
            total_epochs=data.get("total_epochs", 0),
            best_epoch=data.get("best_epoch", 0),
            best_val_loss=data.get("best_val_loss", float("inf")),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            epoch_records=data.get("epoch_records", []),
        )


class ModelTrainer:
    """
    Standardized trainer for DCRNN and Temporal-only models.
    
    Features:
        - Deterministic reproducibility
        - Training strictly on TRAIN split
        - Early stopping / model selection strictly on VALIDATION split
        - Masked physical MAE loss (zeros/dropouts excluded from gradients)
        - Gradient clipping
        - Automatic best checkpoint serialization
    """
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        scaler: TrafficStandardScaler,
        supports: Optional[List[torch.Tensor]] = None,
    ):
        self.config = config
        self.scaler = scaler
        self.device = torch.device(config.device)
        
        # Set seeds for deterministic training
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available() and "cuda" in config.device:
            torch.cuda.manual_seed_all(config.random_seed)
        np.random.seed(config.random_seed)

        self.model = model.to(self.device)
        if supports is not None:
            self.supports = [s.to(self.device) for s in supports]
        else:
            self.supports = None

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        tf_ratio: float,
    ) -> float:
        """Run one training epoch over train_loader."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        std = self.scaler.std
        mean = self.scaler.mean

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            if self.supports is not None:
                # Scaled targets for teacher forcing if enabled
                y_target_scaled = (y_batch - mean) / std if tf_ratio > 0.0 else None
                out_scaled = self.model(
                    x_batch,
                    self.supports,
                    y_target=y_target_scaled,
                    teacher_forcing_ratio=tf_ratio,
                )
            else:
                y_target_scaled = (y_batch - mean) / std if tf_ratio > 0.0 else None
                out_scaled = self.model(
                    x_batch,
                    y_target=y_target_scaled,
                    teacher_forcing_ratio=tf_ratio,
                )

            # Unscale predictions to physical km/h speeds for masked MAE loss
            out_unscaled = out_scaled * std + mean
            loss = masked_mae_loss_tensor(out_unscaled, y_batch, null_val=0.0)

            loss.backward()
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                
            self.optimizer.step()

            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                raise FloatingPointError(f"Non-finite train loss encountered: {loss_val} at batch {total_batches}")

            total_loss += loss_val
            total_batches += 1

        return float(total_loss / max(1, total_batches))

    def evaluate_loss(self, val_loader: DataLoader) -> float:
        """Evaluate validation loss in physical km/h units."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        std = self.scaler.std
        mean = self.scaler.mean

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if self.supports is not None:
                    out_scaled = self.model(x_batch, self.supports)
                else:
                    out_scaled = self.model(x_batch)

                out_unscaled = out_scaled * std + mean
                loss = masked_mae_loss_tensor(out_unscaled, y_batch, null_val=0.0)

                loss_val = float(loss.item())
                if not math.isfinite(loss_val):
                    raise FloatingPointError(f"Non-finite validation loss encountered: {loss_val} at batch {total_batches}")

                total_loss += loss_val
                total_batches += 1

        return float(total_loss / max(1, total_batches))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: str | Path,
        verbose: bool = True,
        resume_checkpoint: Optional[str | Path] = None,
        existing_history: Optional[TrainingHistory] = None,
        history_save_path: Optional[str | Path] = None,
        latest_checkpoint_path: Optional[str | Path] = None,
    ) -> TrainingHistory:
        """
        Execute full training loop with validation early stopping, dual checkpointing
        (best and latest), and support for atomic writes and resumption from interruption.
        """
        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)

        if latest_checkpoint_path is not None:
            latest_ckpt = Path(latest_checkpoint_path)
        else:
            latest_ckpt = ckpt.parent / f"{ckpt.stem.replace('_best', '')}_latest{ckpt.suffix}"
        latest_ckpt.parent.mkdir(parents=True, exist_ok=True)

        history = TrainingHistory(
            model_name=self.config.model_type,
            total_epochs=0,
            best_epoch=0,
            best_val_loss=float("inf"),
            total_duration_seconds=0.0,
        )

        start_epoch = 1
        patience_counter = 0
        best_val_loss = float("inf")
        tf_ratio = self.config.teacher_forcing_initial
        accumulated_duration = 0.0

        # Resume state if requested and checkpoint exists
        if resume_checkpoint is not None and Path(resume_checkpoint).exists():
            ckpt_obj = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt_obj["model_state_dict"])
            if "optimizer_state_dict" in ckpt_obj:
                self.optimizer.load_state_dict(ckpt_obj["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_obj:
                try:
                    self.scheduler.load_state_dict(ckpt_obj["scheduler_state_dict"])
                except Exception:
                    pass
            ckpt_epoch = int(ckpt_obj.get("epoch", 0))
            start_epoch = ckpt_epoch + 1
            best_val_loss = float(ckpt_obj.get("best_val_loss", ckpt_obj.get("val_loss", float("inf"))))
            history.best_epoch = int(ckpt_obj.get("best_epoch", ckpt_epoch))
            history.best_val_loss = round(best_val_loss, 4)
            history.total_epochs = ckpt_epoch
            patience_counter = int(ckpt_obj.get("patience_counter", 0))

            # Compute decayed teacher forcing for resumed epoch
            tf_ratio = self.config.teacher_forcing_initial * (self.config.teacher_forcing_decay ** ckpt_epoch)

            if existing_history is not None:
                history.epoch_records = list(existing_history.epoch_records)
                accumulated_duration = existing_history.total_duration_seconds

            if verbose:
                print(
                    f"Resuming training from epoch {start_epoch:02d}/{self.config.max_epochs:02d} "
                    f"(previous best val loss: {best_val_loss:.4f} at epoch {history.best_epoch})",
                    flush=True,
                )

        start_time_all = time.time()

        for epoch in range(start_epoch, self.config.max_epochs + 1):
            t_epoch_start = time.time()
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            train_loss = self.train_epoch(train_loader, tf_ratio=tf_ratio)
            val_loss = self.evaluate_loss(val_loader)
            self.scheduler.step(val_loss)

            # Decay teacher forcing ratio
            tf_ratio *= self.config.teacher_forcing_decay

            elapsed = time.time() - t_epoch_start
            improved = False

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                patience_counter = 0
                history.best_epoch = epoch
                history.best_val_loss = round(best_val_loss, 4)
                
                # Atomic save best checkpoint (write to temp then rename)
                temp_ckpt = ckpt.with_suffix(".tmp")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": asdict(self.config),
                        "scaler_mean": self.scaler.mean,
                        "scaler_std": self.scaler.std,
                    },
                    temp_ckpt,
                )
                temp_ckpt.replace(ckpt)
            else:
                patience_counter += 1

            # Atomic save latest resumable checkpoint (every epoch)
            temp_latest = latest_ckpt.with_suffix(".tmp")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "val_loss": val_loss,
                    "patience_counter": patience_counter,
                    "best_val_loss": best_val_loss,
                    "best_epoch": history.best_epoch,
                    "config": asdict(self.config),
                    "scaler_mean": self.scaler.mean,
                    "scaler_std": self.scaler.std,
                },
                temp_latest,
            )
            temp_latest.replace(latest_ckpt)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "learning_rate": current_lr,
                "elapsed_seconds": round(elapsed, 2),
                "improved": improved,
            }
            history.epoch_records.append(record)
            history.total_epochs = epoch
            history.total_duration_seconds = round(accumulated_duration + (time.time() - start_time_all), 2)

            # Save progress history after every epoch atomically
            if history_save_path is not None:
                history.save_json(history_save_path)

            if verbose:
                imp_str = " * (Best)" if improved else ""
                print(
                    f"Epoch {epoch:02d}/{self.config.max_epochs:02d} | "
                    f"Train MAE: {train_loss:.4f} km/h | "
                    f"Val MAE: {val_loss:.4f} km/h | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {elapsed:.1f}s{imp_str}",
                    flush=True,
                )

            if patience_counter >= self.config.patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch} (patience={self.config.patience}).", flush=True)
                break

        history.total_duration_seconds = round(accumulated_duration + (time.time() - start_time_all), 2)
        if history_save_path is not None:
            history.save_json(history_save_path)
        
        # Reload best model weights into memory
        if ckpt.exists():
            checkpoint = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return history


