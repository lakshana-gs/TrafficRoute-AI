"""P4.1 Multi-Epoch Training and Convergence Pipeline for DCRNN."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from typing import Any, Dict
import numpy as np
import torch

from src.models.p4.dcrnn import DCRNNModel
from src.models.p4.diffusion import (
    build_diffusion_supports_from_pkl,
    validate_sensor_graph,
)
from src.models.p4.dataset import create_traffic_dataloaders
from src.models.p4.evaluator import P4Evaluator
from src.models.p4.trainer import ModelTrainer, TrainingConfig, TrainingHistory
from src.models.preprocessing import (
    TrafficStandardScaler,
    chronological_split,
    load_raw_metr_la,
    load_sensor_graph,
    prepare_windowed_dataset,
)


def run_p4_1_training(
    raw_h5_path: str = "data/traffic/raw/metr_la/metr-la.h5",
    adj_pkl_path: str = "data/traffic/raw/metr_la/adj_mx.pkl",
    scaler_path: str = "data/traffic/processed/metr_la/scaler.json",
    p3_results_path: str = "reports/p3_baseline_results.json",
    temporal_ckpt_path: str = "artifacts/models/p4_temporal_only_best.pt",
    max_epochs: int = 20,
    patience: int = 5,
    batch_size: int = 128,
    hidden_dim: int = 32,
    num_layers: int = 2,
    diffusion_order_k: int = 2,
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute P4.1 multi-epoch training and convergence evaluation.
    """
    total_start = time.time()
    print("=" * 70, flush=True)
    print("TRAFFICROUTE AI — P4.1 DCRNN MULTI-EPOCH TRAINING & CONVERGENCE", flush=True)
    print("=" * 70, flush=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device_str.upper()} (PyTorch {torch.__version__})", flush=True)

    # 1. Load Data
    print("\n[1/5] Loading METR-LA Data and Chronological Partitions...", flush=True)
    speed_mx, timestamps, sensor_ids = load_raw_metr_la(raw_h5_path, convert_to_kmh=True)
    split = chronological_split(speed_mx, timestamps, sensor_ids, train_ratio=0.70, val_ratio=0.15)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)
    scaler = TrafficStandardScaler.load_json(scaler_path)

    print(f"  Train windows: {windows.x_train.shape} (70%)", flush=True)
    print(f"  Val windows:   {windows.x_val.shape} (15%)", flush=True)
    print(f"  Test windows:  {windows.x_test.shape} (15%)", flush=True)
    print(f"  Scaler: mean={scaler.mean:.4f} km/h, std={scaler.std:.4f} km/h", flush=True)

    train_loader, val_loader, test_loader = create_traffic_dataloaders(
        windows=windows,
        scaler=scaler,
        batch_size=batch_size,
        shuffle_train=True,
    )

    # 2. Graph Supports
    print("\n[2/5] Loading 207-Node METR-LA Graph & Diffusion Supports...", flush=True)
    supports_tensor = build_diffusion_supports_from_pkl(adj_pkl_path, k=diffusion_order_k, device=device)
    print(f"  Nodes: {len(sensor_ids)}, Supports: {len(supports_tensor)} (K={diffusion_order_k})", flush=True)

    # 3. Train DCRNN
    print(f"\n[3/5] Training DCRNN (max_epochs={max_epochs}, patience={patience}, batch_size={batch_size})...", flush=True)
    config = TrainingConfig(
        model_type="DCRNN",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        diffusion_order_k=diffusion_order_k,
        horizon=12,
        history=12,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        random_seed=random_seed,
        device=device_str,
    )
    model = DCRNNModel(
        in_features=1,
        out_features=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_supports=len(supports_tensor),
        horizon=12,
    )
    trainer = ModelTrainer(
        model=model,
        config=config,
        scaler=scaler,
        supports=supports_tensor,
    )
    best_ckpt_path = "artifacts/models/p4_dcrnn_best.pt"
    latest_ckpt_path = "artifacts/models/p4_dcrnn_latest.pt"
    resume_ckpt = None
    existing_history = None
    hist_path = Path("reports/p4_training_history.json")

    # Prefer latest checkpoint if available, else fall back to best checkpoint
    candidate_ckpt = latest_ckpt_path if Path(latest_ckpt_path).exists() else best_ckpt_path
    if Path(candidate_ckpt).exists():
        try:
            chk = torch.load(candidate_ckpt, map_location=device)
            if "model_state_dict" in chk and "epoch" in chk and chk["epoch"] < max_epochs:
                resume_ckpt = candidate_ckpt
                if hist_path.exists():
                    existing_history = TrainingHistory.load_json(hist_path)
                print(f"Resuming DCRNN training from {Path(candidate_ckpt).name} at epoch {chk['epoch']} (saved val_loss: {chk.get('val_loss'):.4f})...", flush=True)
        except Exception as e:
            print(f"Could not parse checkpoint for resume ({e}), starting fresh.", flush=True)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=best_ckpt_path,
        latest_checkpoint_path=latest_ckpt_path,
        verbose=True,
        resume_checkpoint=resume_ckpt,
        existing_history=existing_history,
        history_save_path=hist_path,
    )

    # Save DCRNN Model Config
    config_path = "artifacts/models/p4_dcrnn_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "milestone": "P4.1",
                "architecture": "Diffusion Convolutional Recurrent Neural Network (DCRNN)",
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "diffusion_order_k": diffusion_order_k,
                "num_supports": len(supports_tensor),
                "history_steps": 12,
                "horizon_steps": 12,
                "optimizer": "Adam",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs_trained": history.total_epochs,
                "best_epoch": history.best_epoch,
                "best_val_loss_kmh": history.best_val_loss,
                "random_seed": random_seed,
                "dataset": "METR-LA",
                "scaler_reference": scaler_path,
                "graph_reference": adj_pkl_path,
                "device": device_str,
            },
            f,
            indent=2,
        )

    # 4. Final Evaluation from Reloaded Checkpoint
    print("\n[4/5] Reloading Best Checkpoint from Disk and Evaluating on Test Partition...", flush=True)
    loaded_model = DCRNNModel(
        in_features=1,
        out_features=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_supports=len(supports_tensor),
        horizon=12,
    ).to(device)
    ckpt_obj = torch.load(best_ckpt_path, map_location=device)
    loaded_model.load_state_dict(ckpt_obj["model_state_dict"])

    test_res, y_true_test, y_pred_dcrnn = P4Evaluator.evaluate_model(
        model=loaded_model,
        data_loader=test_loader,
        scaler=scaler,
        sensor_ids=sensor_ids,
        model_name="DCRNN (Spatio-Temporal)",
        supports=supports_tensor,
        device=device,
    )

    print(f"  Test Overall MAE:  {test_res.overall_mae:.4f} km/h", flush=True)
    print(f"  Test Overall RMSE: {test_res.overall_rmse:.4f} km/h", flush=True)
    print(f"  Test Overall MAPE: {test_res.overall_mape:.4f}%", flush=True)
    print(f"  5-min MAE:  {test_res.horizons['5_min']['mae']:.4f} km/h", flush=True)
    print(f"  15-min MAE: {test_res.horizons['15_min']['mae']:.4f} km/h", flush=True)
    print(f"  30-min MAE: {test_res.horizons['30_min']['mae']:.4f} km/h", flush=True)
    print(f"  45-min MAE: {test_res.horizons['45_min']['mae']:.4f} km/h", flush=True)
    print(f"  60-min MAE: {test_res.horizons['60_min']['mae']:.4f} km/h", flush=True)

    # 5. Assemble Comparison Reports
    print("\n[5/5] Updating Machine-Readable Reports and Comparison Tables...", flush=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    history.save_json(reports_dir / "p4_training_history.json")

    # Load P3 baselines
    p3_data: Dict[str, Any] = {}
    if Path(p3_results_path).exists():
        with open(p3_results_path, "r") as f:
            p3_data = json.load(f).get("baselines", {})

    # Load existing P4 results for earlier models and reference
    existing_p4_path = reports_dir / "p4_results.json"
    temporal_test_dict = {}
    existing_models = {}
    existing_comp_table = {}
    if existing_p4_path.exists():
        with open(existing_p4_path, "r") as f:
            prev_p4 = json.load(f)
            temporal_test_dict = prev_p4.get("models", {}).get("temporal_only_gru", {})
            existing_models = prev_p4.get("models", {})
            existing_comp_table = prev_p4.get("comparison_table", {})

    _, _, raw_adj = load_sensor_graph(adj_pkl_path)
    graph_props = validate_sensor_graph(raw_adj, sensor_ids)

    comparison_table = {
        **existing_comp_table,
        "Persistence": p3_data.get("persistence", {}).get("test", existing_comp_table.get("Persistence", {})),
        "Historical Mean": p3_data.get("historical_mean", {}).get("test", existing_comp_table.get("Historical Mean", {})),
        "XGBoost": p3_data.get("xgboost", {}).get("test", existing_comp_table.get("XGBoost", {})),
        "Temporal-Only GRU (P4)": {
            "overall": temporal_test_dict.get("test_overall", {}),
            "horizons": temporal_test_dict.get("test_horizons", {}),
        },
        "DCRNN (P4.1 Final Best)": {
            "overall": {
                "mae": test_res.overall_mae,
                "rmse": test_res.overall_rmse,
                "mape": test_res.overall_mape,
            },
            "horizons": test_res.horizons,
        },
    }

    # Save training config & reproducibility metadata
    reproducibility_config = {
        "milestone": "P4.1",
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": device_str,
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset": {
            "name": "METR-LA",
            "h5_path": raw_h5_path,
            "adj_path": adj_pkl_path,
            "speed_unit": "km/h",
            "conversion_factor_mph_to_kmh": 1.609344,
            "total_timesteps": 34272,
            "num_sensors": 207,
            "history_steps": 12,
            "horizon_steps": 12,
            "train_windows": windows.x_train.shape[0],
            "val_windows": windows.x_val.shape[0],
            "test_windows": windows.x_test.shape[0],
        },
        "dcrnn_hyperparameters": asdict(config),
    }
    with open(reports_dir / "p4_training_config.json", "w") as f:
        json.dump(reproducibility_config, f, indent=2)

    # Save complete P4.1 results JSON
    p4_1_results = {
        "milestone": "P4.1 — DCRNN Multi-Epoch Training Completion",
        "dataset": "METR-LA",
        "speed_unit": "km/h",
        "graph": graph_props._asdict(),
        "split_timestamps": {
            "train": [str(split.train_timestamps[0]), str(split.train_timestamps[-1])],
            "val": [str(split.val_timestamps[0]), str(split.val_timestamps[-1])],
            "test": [str(split.test_timestamps[0]), str(split.test_timestamps[-1])],
        },
        "models": {
            **existing_models,
            "dcrnn_final_best": {
                "training_duration_seconds": history.total_duration_seconds,
                "total_epochs": history.total_epochs,
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
                "test_overall": {
                    "mae": test_res.overall_mae,
                    "rmse": test_res.overall_rmse,
                    "mape": test_res.overall_mape,
                },
                "test_horizons": test_res.horizons,
                "sensor_summary": test_res.sensor_summary,
            },
            "temporal_only_gru": temporal_test_dict,
        },
        "comparison_table": comparison_table,
        "missing_data_analysis": test_res.missing_data_summary,
        "critical_domain_limitation": (
            "METR-LA loop detectors in Los Angeles County. "
            "These models validate the spatio-temporal ML architecture on authentic traffic data. "
            "They do NOT predict Tamil Nadu road traffic or map to Coimbatore OSM segments."
        ),
        "total_pipeline_duration_seconds": round(time.time() - total_start, 2),
    }
    with open(reports_dir / "p4_results.json", "w") as f:
        json.dump(p4_1_results, f, indent=2)


    print("\n" + "=" * 70, flush=True)
    print("P4.1 DCRNN TRAINING & EVALUATION COMPLETED", flush=True)
    print(f"Total Duration: {p4_1_results['total_pipeline_duration_seconds']}s", flush=True)
    print("=" * 70, flush=True)
    return p4_1_results


if __name__ == "__main__":
    run_p4_1_training()
