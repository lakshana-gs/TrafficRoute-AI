"""Evaluation of Epoch 6 Checkpoint on METR-LA Test Partition."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import torch

from src.models.p4.dcrnn import DCRNNModel
from src.models.p4.diffusion import build_diffusion_supports_from_pkl
from src.models.p4.dataset import create_traffic_dataloaders
from src.models.p4.evaluator import P4Evaluator
from src.models.preprocessing import (
    load_raw_metr_la,
    chronological_split,
    prepare_windowed_dataset,
    TrafficStandardScaler,
)

def main():
    print("Starting evaluation...", flush=True)
    t0 = time.time()
    device = torch.device("cpu")

    speed_mx, timestamps, sensor_ids = load_raw_metr_la(
        "data/traffic/raw/metr_la/metr-la.h5", convert_to_kmh=True
    )
    split = chronological_split(speed_mx, timestamps, sensor_ids, train_ratio=0.70, val_ratio=0.15)
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)
    scaler = TrafficStandardScaler.load_json("data/traffic/processed/metr_la/scaler.json")
    
    # We only need test loader
    test_loader = create_traffic_dataloaders(
        windows=windows, scaler=scaler, batch_size=128, shuffle_train=False
    )[2]

    supports_tensor = build_diffusion_supports_from_pkl(
        "data/traffic/raw/metr_la/adj_mx.pkl", k=2, device=device
    )

    ckpt_path = "artifacts/models/p4_dcrnn_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: Epoch {ckpt.get('epoch')}, Val Loss {ckpt.get('val_loss'):.4f}", flush=True)

    model = DCRNNModel(
        in_features=1,
        out_features=1,
        hidden_dim=32,
        num_layers=2,
        num_supports=len(supports_tensor),
        horizon=12,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_res, y_true_test, y_pred = P4Evaluator.evaluate_model(
        model=model,
        data_loader=test_loader,
        scaler=scaler,
        sensor_ids=sensor_ids,
        model_name="DCRNN (Epoch 6 Checkpoint)",
        supports=supports_tensor,
        device=device,
    )

    results = {
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "overall_mae": test_res.overall_mae,
        "overall_rmse": test_res.overall_rmse,
        "overall_mape": test_res.overall_mape,
        "horizons": test_res.horizons,
        "sensor_summary": test_res.sensor_summary,
        "missing_data_summary": test_res.missing_data_summary,
        "evaluation_duration_seconds": round(time.time() - t0, 2),
    }

    output_path = Path("reports/p4_1_checkpoint_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete in {results['evaluation_duration_seconds']}s", flush=True)
    print(f"Overall MAE:  {test_res.overall_mae:.4f} km/h", flush=True)
    print(f"Overall RMSE: {test_res.overall_rmse:.4f} km/h", flush=True)
    print(f"Overall MAPE: {test_res.overall_mape:.4f}%", flush=True)
    for h in ["5_min", "15_min", "30_min", "45_min", "60_min"]:
        print(f"  {h} MAE: {test_res.horizons[h]['mae']:.4f} km/h", flush=True)

if __name__ == "__main__":
    main()
