"""End-to-End Orchestrator for P4 Spatio-Temporal Prediction Pipeline."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, Optional
import numpy as np
import torch

from src.models.p4.dcrnn import DCRNNModel
from src.models.p4.diffusion import (
    build_diffusion_supports_from_pkl,
    compute_diffusion_supports,
    save_normalized_supports,
    validate_sensor_graph,
)
from src.models.p4.dataset import create_traffic_dataloaders
from src.models.p4.evaluator import P4Evaluator
from src.models.p4.temporal_only import TemporalOnlyGRU
from src.models.p4.trainer import ModelTrainer, TrainingConfig
from src.models.preprocessing import (
    TrafficStandardScaler,
    chronological_split,
    load_raw_metr_la,
    load_sensor_graph,
    prepare_windowed_dataset,
)


def run_p4_pipeline(
    raw_h5_path: str = "data/traffic/raw/metr_la/metr-la.h5",
    adj_pkl_path: str = "data/traffic/raw/metr_la/adj_mx.pkl",
    scaler_path: str = "data/traffic/processed/metr_la/scaler.json",
    p3_results_path: str = "reports/p3_baseline_results.json",
    max_epochs: int = 10,
    patience: int = 4,
    batch_size: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 2,
    diffusion_order_k: int = 2,
    learning_rate: float = 0.001,
    random_seed: int = 42,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute full P4 spatio-temporal modeling and evaluation pipeline.
    """
    total_start = time.time()
    print("=" * 70)
    print("TRAFFICROUTE AI — P4 SPATIO-TEMPORAL SPEED PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Device detection
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device_str.upper()} (PyTorch {torch.__version__})")

    # 2. Sensor Graph Validation & Normalization
    print("\n[1/7] Validating and Normalizing METR-LA Sensor Graph...")
    sensor_ids_graph, sensor_id_to_ind, raw_adj = load_sensor_graph(adj_pkl_path)
    graph_props = validate_sensor_graph(raw_adj, sensor_ids_graph)
    print(f"  Nodes: {graph_props.sensor_count}")
    print(f"  Adjacency shape: {graph_props.adjacency_shape}")
    print(f"  Finite: {graph_props.is_finite}, No NaN: {not graph_props.has_nan}")
    print(f"  Diagonal self-loops: [{graph_props.diagonal_min}, {graph_props.diagonal_max}]")
    print(f"  Directed/Asymmetric: {not graph_props.is_symmetric}")
    print(f"  Non-zero edges: {graph_props.nonzero_edges} (density: {graph_props.density:.4%})")

    # Save derived normalized diffusion supports
    supports_np = compute_diffusion_supports(raw_adj, k=diffusion_order_k)
    save_normalized_supports(
        supports_np,
        output_npz_path="data/traffic/processed/metr_la/normalized_adj_mx.npz",
        metadata_json_path="data/traffic/processed/metr_la/normalized_adj_meta.json",
    )
    supports_tensor = [torch.from_numpy(s).float().to(device) for s in supports_np]
    print(f"  Computed {len(supports_tensor)} diffusion supports for order K={diffusion_order_k}.")

    # 3. Data Loading & Chronological Partitioning
    print("\n[2/7] Loading Raw METR-LA Data and Generating Sliding Windows...")
    speed_mx, timestamps, sensor_ids = load_raw_metr_la(raw_h5_path, convert_to_kmh=True)
    split = chronological_split(speed_mx, timestamps, sensor_ids, train_ratio=0.70, val_ratio=0.15)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)
    scaler = TrafficStandardScaler.load_json(scaler_path)

    print(f"  Train windows: {windows.x_train.shape} [{split.train_timestamps[0]} -> {split.train_timestamps[-1]}]")
    print(f"  Val windows:   {windows.x_val.shape}   [{split.val_timestamps[0]} -> {split.val_timestamps[-1]}]")
    print(f"  Test windows:  {windows.x_test.shape}  [{split.test_timestamps[0]} -> {split.test_timestamps[-1]}]")
    print(f"  Training scaler: mean={scaler.mean:.4f} km/h, std={scaler.std:.4f} km/h")

    train_loader, val_loader, test_loader = create_traffic_dataloaders(
        windows=windows,
        scaler=scaler,
        batch_size=batch_size,
        shuffle_train=True,
    )

    # 4. Train DCRNN Model
    print("\n[3/7] Training DCRNN Spatio-Temporal Model...")
    dcrnn_config = TrainingConfig(
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
    dcrnn_model = DCRNNModel(
        in_features=1,
        out_features=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_supports=len(supports_tensor),
        horizon=12,
    )
    dcrnn_trainer = ModelTrainer(
        model=dcrnn_model,
        config=dcrnn_config,
        scaler=scaler,
        supports=supports_tensor,
    )
    dcrnn_ckpt_path = "artifacts/models/p4_dcrnn_best.pt"
    dcrnn_history = dcrnn_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=dcrnn_ckpt_path,
        verbose=True,
    )

    # Save DCRNN Model Config
    dcrnn_config_path = "artifacts/models/p4_dcrnn_config.json"
    Path(dcrnn_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dcrnn_config_path, "w") as f:
        json.dump(
            {
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
                "epochs_trained": dcrnn_history.total_epochs,
                "best_epoch": dcrnn_history.best_epoch,
                "best_val_loss_kmh": dcrnn_history.best_val_loss,
                "random_seed": random_seed,
                "dataset": "METR-LA",
                "scaler_reference": scaler_path,
                "graph_reference": adj_pkl_path,
                "device": device_str,
            },
            f,
            indent=2,
        )

    # 5. Train Temporal-Only Model (Spatial Ablation)
    print("\n[4/7] Training Temporal-Only GRU Model (Spatial Ablation Baseline)...")
    temporal_config = TrainingConfig(
        model_type="TemporalOnlyGRU",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        diffusion_order_k=0,
        horizon=12,
        history=12,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        random_seed=random_seed,
        device=device_str,
    )
    temporal_model = TemporalOnlyGRU(
        in_features=1,
        out_features=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        horizon=12,
    )
    temporal_trainer = ModelTrainer(
        model=temporal_model,
        config=temporal_config,
        scaler=scaler,
        supports=None,
    )
    temporal_ckpt_path = "artifacts/models/p4_temporal_only_best.pt"
    temporal_history = temporal_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=temporal_ckpt_path,
        verbose=True,
    )

    # 6. Evaluate Models on Unseen Test Partition
    print("\n[5/7] Evaluating Checkpoints on Unseen Test Partition (5,119 Windows)...")
    
    # Reload DCRNN checkpoint to verify reproducibility from disk
    dcrnn_loaded = DCRNNModel(
        in_features=1,
        out_features=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_supports=len(supports_tensor),
        horizon=12,
    ).to(device)
    dcrnn_ckpt_obj = torch.load(dcrnn_ckpt_path, map_location=device)
    dcrnn_loaded.load_state_dict(dcrnn_ckpt_obj["model_state_dict"])

    dcrnn_test_res, y_true_test, y_pred_dcrnn = P4Evaluator.evaluate_model(
        model=dcrnn_loaded,
        data_loader=test_loader,
        scaler=scaler,
        sensor_ids=sensor_ids,
        model_name="DCRNN (Spatio-Temporal)",
        supports=supports_tensor,
        device=device,
    )

    # Reload Temporal checkpoint
    temporal_loaded = TemporalOnlyGRU(
        in_features=1,
        out_features=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        horizon=12,
    ).to(device)
    temporal_ckpt_obj = torch.load(temporal_ckpt_path, map_location=device)
    temporal_loaded.load_state_dict(temporal_ckpt_obj["model_state_dict"])

    temporal_test_res, _, y_pred_temporal = P4Evaluator.evaluate_model(
        model=temporal_loaded,
        data_loader=test_loader,
        scaler=scaler,
        sensor_ids=sensor_ids,
        model_name="Temporal-Only GRU (Ablation)",
        supports=None,
        device=device,
    )

    # 7. Compare Against P3 Baselines
    print("\n[6/7] Assembling Benchmark Comparison...")
    p3_data: Dict[str, Any] = {}
    if Path(p3_results_path).exists():
        with open(p3_results_path, "r") as f:
            p3_data = json.load(f).get("baselines", {})

    comparison_table = {
        "Persistence": p3_data.get("persistence", {}).get("test", {}),
        "Historical Mean": p3_data.get("historical_mean", {}).get("test", {}),
        "XGBoost": p3_data.get("xgboost", {}).get("test", {}),
        "Temporal-Only GRU": {
            "overall": {
                "mae": temporal_test_res.overall_mae,
                "rmse": temporal_test_res.overall_rmse,
                "mape": temporal_test_res.overall_mape,
            },
            "horizons": temporal_test_res.horizons,
        },
        "DCRNN (Spatio-Temporal)": {
            "overall": {
                "mae": dcrnn_test_res.overall_mae,
                "rmse": dcrnn_test_res.overall_rmse,
                "mape": dcrnn_test_res.overall_mape,
            },
            "horizons": dcrnn_test_res.horizons,
        },
    }

    # 8. Save Machine-Readable Reports
    print("\n[7/7] Generating Machine-Readable Reports and Summary...")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save training history
    dcrnn_history.save_json(reports_dir / "p4_training_history.json")

    # Save training config & reproducibility metadata
    reproducibility_config = {
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
        "dcrnn_hyperparameters": asdict(dcrnn_config),
        "temporal_ablation_hyperparameters": asdict(temporal_config),
    }
    with open(reports_dir / "p4_training_config.json", "w") as f:
        json.dump(reproducibility_config, f, indent=2)

    # Save complete P4 results JSON
    p4_results = {
        "milestone": "P4 — Spatio-Temporal Traffic Speed Prediction",
        "dataset": "METR-LA",
        "speed_unit": "km/h",
        "graph": graph_props._asdict(),
        "split_timestamps": {
            "train": [str(split.train_timestamps[0]), str(split.train_timestamps[-1])],
            "val": [str(split.val_timestamps[0]), str(split.val_timestamps[-1])],
            "test": [str(split.test_timestamps[0]), str(split.test_timestamps[-1])],
        },
        "models": {
            "dcrnn": {
                "training_duration_seconds": dcrnn_history.total_duration_seconds,
                "best_epoch": dcrnn_history.best_epoch,
                "best_val_loss": dcrnn_history.best_val_loss,
                "test_overall": {
                    "mae": dcrnn_test_res.overall_mae,
                    "rmse": dcrnn_test_res.overall_rmse,
                    "mape": dcrnn_test_res.overall_mape,
                },
                "test_horizons": dcrnn_test_res.horizons,
                "sensor_summary": dcrnn_test_res.sensor_summary,
            },
            "temporal_only_gru": {
                "training_duration_seconds": temporal_history.total_duration_seconds,
                "best_epoch": temporal_history.best_epoch,
                "best_val_loss": temporal_history.best_val_loss,
                "test_overall": {
                    "mae": temporal_test_res.overall_mae,
                    "rmse": temporal_test_res.overall_rmse,
                    "mape": temporal_test_res.overall_mape,
                },
                "test_horizons": temporal_test_res.horizons,
            },
        },
        "comparison_table": comparison_table,
        "missing_data_analysis": dcrnn_test_res.missing_data_summary,
        "critical_domain_limitation": (
            "METR-LA loop detectors in Los Angeles County. "
            "These models validate the spatio-temporal ML architecture on authentic traffic data. "
            "They do NOT predict Tamil Nadu road traffic or map to Coimbatore OSM segments."
        ),
        "total_pipeline_duration_seconds": round(time.time() - total_start, 2),
    }
    with open(reports_dir / "p4_results.json", "w") as f:
        json.dump(p4_results, f, indent=2)

    print("\n" + "=" * 70)
    print("P4 PREDICTION PIPELINE COMPLETED")
    print(f"Total Duration: {p4_results['total_pipeline_duration_seconds']}s")
    print("=" * 70)
    return p4_results


if __name__ == "__main__":
    run_p4_pipeline()
