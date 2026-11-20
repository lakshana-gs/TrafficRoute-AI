"""Execution orchestrator for P3 baseline speed prediction evaluation."""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_horizons
from src.models.historical_mean import HistoricalTimeOfDayBaseline
from src.models.ml_baseline import XGBoostSpeedBaseline
from src.models.persistence import PersistenceBaseline
from src.models.preprocessing import (
    MPH_TO_KMH,
    chronological_split,
    load_raw_metr_la,
    load_sensor_graph,
    prepare_windowed_dataset,
    TrafficStandardScaler,
)


def run_p3_baseline_pipeline(
    h5_path: str = "data/traffic/raw/metr_la/metr-la.h5",
    adj_path: str = "data/traffic/raw/metr_la/adj_mx.pkl",
    output_report_path: str = "reports/p3_baseline_results.json",
    scaler_output_path: str = "data/traffic/processed/metr_la/scaler.json",
) -> Dict[str, Any]:
    """
    Run complete P3 baseline pipeline across Persistence, Historical Mean, and XGBoost.
    """
    start_time = time.time()
    print("======================================================================")
    print("TRAFFICROUTE AI — P3 BASELINE SPEED PREDICTION PIPELINE")
    print("======================================================================")

    # 1. Load Raw METR-LA Speed Matrix
    print("\n[Step 1/6] Loading raw METR-LA speed matrix and sensor graph...")
    speed_matrix_kmh, timestamps, sensor_ids = load_raw_metr_la(h5_path, convert_to_kmh=True)
    sensor_ids_graph, sensor_id_map, adj_mx = load_sensor_graph(adj_path)
    
    T, N = speed_matrix_kmh.shape
    print(f"  Loaded speed tensor: shape=({T}, {N}), unit=km/h (converted via * {MPH_TO_KMH:.6f})")
    print(f"  Total observations: {T * N:,}")
    print(f"  Temporal range: {timestamps[0]} to {timestamps[-1]}")
    print(f"  Adjacency matrix: shape={adj_mx.shape}, non-zero edges={np.count_nonzero(adj_mx)}")

    # 2. Chronological Partitioning
    print("\n[Step 2/6] Chronological partitioning (70% Train / 15% Val / 15% Test)...")
    split = chronological_split(speed_matrix_kmh, timestamps, sensor_ids, train_ratio=0.70, val_ratio=0.15)
    
    print(f"  TRAIN: {len(split.train_data)} steps | {split.train_timestamps[0]} -> {split.train_timestamps[-1]}")
    print(f"  VAL:   {len(split.val_data)} steps | {split.val_timestamps[0]} -> {split.val_timestamps[-1]}")
    print(f"  TEST:  {len(split.test_data)} steps | {split.test_timestamps[0]} -> {split.test_timestamps[-1]}")

    # 3. Fit & Save Training-Only Scaler
    print("\n[Step 3/6] Fitting Z-score standard scaler strictly on training data...")
    scaler = TrafficStandardScaler(null_val=0.0)
    scaler.fit(split.train_data)
    scaler.save_json(scaler_output_path, extra_meta={"sensors_count": N, "train_steps": len(split.train_data)})
    print(f"  Training Speed Mean: {scaler.mean:.2f} km/h, Std: {scaler.std:.2f} km/h")
    print(f"  Saved scaler metadata to: {scaler_output_path}")

    # 4. Window Generation (12 steps -> 12 steps)
    print("\n[Step 4/6] Generating sliding windows (History=12 [60m], Horizon=12 [60m])...")
    windows = prepare_windowed_dataset(split, history_steps=12, horizon_steps=12)
    print(f"  X_train: {windows.x_train.shape}, Y_train: {windows.y_train.shape}")
    print(f"  X_val:   {windows.x_val.shape}, Y_val:   {windows.y_val.shape}")
    print(f"  X_test:  {windows.x_test.shape}, Y_test:  {windows.y_test.shape}")

    results: Dict[str, Any] = {
        "dataset": "METR-LA",
        "speed_unit": "km/h",
        "conversion_factor_mph_to_kmh": MPH_TO_KMH,
        "tensor_shapes": {
            "total_raw": [T, N],
            "train_windows": list(windows.x_train.shape),
            "val_windows": list(windows.x_val.shape),
            "test_windows": list(windows.x_test.shape),
        },
        "chronological_split_timestamps": {
            "train": [str(split.train_timestamps[0]), str(split.train_timestamps[-1])],
            "val": [str(split.val_timestamps[0]), str(split.val_timestamps[-1])],
            "test": [str(split.test_timestamps[0]), str(split.test_timestamps[-1])],
        },
        "scaler_statistics": {
            "train_mean_kmh": round(scaler.mean, 4),
            "train_std_kmh": round(scaler.std, 4),
        },
        "baselines": {},
    }

    # 5. Baseline 1: Persistence
    print("\n[Step 5a/6] Evaluating Baseline 1: Persistence (Latest Speed Forward)...")
    persistence = PersistenceBaseline(horizon_steps=12)
    pers_val = persistence.evaluate(windows.x_val, windows.y_val, step_minutes=5)
    pers_test = persistence.evaluate(windows.x_test, windows.y_test, step_minutes=5)
    
    results["baselines"]["persistence"] = {
        "description": "Persistence Baseline: carries latest observed speed at t forward across all 12 horizons",
        "validation": pers_val,
        "test": pers_test,
    }
    print(f"  [Persistence] Test 5-min MAE:  {pers_test['horizons']['5_min']['mae']:.2f} km/h, RMSE: {pers_test['horizons']['5_min']['rmse']:.2f} km/h, MAPE: {pers_test['horizons']['5_min']['mape']:.2f}%")
    print(f"  [Persistence] Test 15-min MAE: {pers_test['horizons']['15_min']['mae']:.2f} km/h")
    print(f"  [Persistence] Test 30-min MAE: {pers_test['horizons']['30_min']['mae']:.2f} km/h")
    print(f"  [Persistence] Test 60-min MAE: {pers_test['horizons']['60_min']['mae']:.2f} km/h")
    print(f"  [Persistence] Test Overall MAE: {pers_test['overall']['mae']:.2f} km/h, RMSE: {pers_test['overall']['rmse']:.2f} km/h, MAPE: {pers_test['overall']['mape']:.2f}%")

    # 6. Baseline 2: Historical Time-of-Day Mean
    print("\n[Step 5b/6] Evaluating Baseline 2: Historical Time-of-Day Mean (Train Only)...")
    hist_model = HistoricalTimeOfDayBaseline(horizon_steps=12)
    hist_model.fit(split.train_data, split.train_timestamps)
    
    hist_val = hist_model.evaluate(split.val_timestamps, windows.y_val, history_steps=12, step_minutes=5)
    hist_test = hist_model.evaluate(split.test_timestamps, windows.y_test, history_steps=12, step_minutes=5)
    
    results["baselines"]["historical_mean"] = {
        "description": "Historical Mean Baseline: Day-of-week & 5-min slot speed average strictly from training data",
        "validation": hist_val,
        "test": hist_test,
    }
    print(f"  [Historical Mean] Test 5-min MAE:  {hist_test['horizons']['5_min']['mae']:.2f} km/h")
    print(f"  [Historical Mean] Test 15-min MAE: {hist_test['horizons']['15_min']['mae']:.2f} km/h")
    print(f"  [Historical Mean] Test 30-min MAE: {hist_test['horizons']['30_min']['mae']:.2f} km/h")
    print(f"  [Historical Mean] Test 60-min MAE: {hist_test['horizons']['60_min']['mae']:.2f} km/h")
    print(f"  [Historical Mean] Test Overall MAE: {hist_test['overall']['mae']:.2f} km/h, RMSE: {hist_test['overall']['rmse']:.2f} km/h, MAPE: {hist_test['overall']['mape']:.2f}%")

    # 7. Baseline 3: Simple ML Model (XGBoost)
    print("\n[Step 5c/6] Training & Evaluating Baseline 3: XGBoost on Tabular Lag Features...")
    xgb_model = XGBoostSpeedBaseline(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42)
    xgb_model.fit(
        windows.x_train,
        windows.y_train,
        split.train_timestamps,
        sample_step_stride=4,  # Efficient training stride
        sensor_stride=2,
    )
    
    print("  Evaluating XGBoost on validation and test sets...")
    xgb_val = xgb_model.evaluate(windows.x_val, windows.y_val, split.val_timestamps, step_minutes=5)
    xgb_test = xgb_model.evaluate(windows.x_test, windows.y_test, split.test_timestamps, step_minutes=5)
    
    results["baselines"]["xgboost"] = {
        "description": "XGBoost Regressor: Tabular lag features (t, t-1, t-2, t-3, t-6, t-12, mean, std, hour, dow) per horizon",
        "validation": xgb_val,
        "test": xgb_test,
    }
    print(f"  [XGBoost] Test 5-min MAE:  {xgb_test['horizons']['5_min']['mae']:.2f} km/h, RMSE: {xgb_test['horizons']['5_min']['rmse']:.2f} km/h, MAPE: {xgb_test['horizons']['5_min']['mape']:.2f}%")
    print(f"  [XGBoost] Test 15-min MAE: {xgb_test['horizons']['15_min']['mae']:.2f} km/h")
    print(f"  [XGBoost] Test 30-min MAE: {xgb_test['horizons']['30_min']['mae']:.2f} km/h")
    print(f"  [XGBoost] Test 60-min MAE: {xgb_test['horizons']['60_min']['mae']:.2f} km/h")
    print(f"  [XGBoost] Test Overall MAE: {xgb_test['overall']['mae']:.2f} km/h, RMSE: {xgb_test['overall']['rmse']:.2f} km/h, MAPE: {xgb_test['overall']['mape']:.2f}%")

    # 8. Comparison Summary Table
    print("\n======================================================================")
    print("P3 BASELINE PERFORMANCE COMPARISON TABLE (TEST SET, KM/H)")
    print("======================================================================")
    print(f"{'Model':<20} | {'5-min MAE':<10} | {'15-min MAE':<10} | {'30-min MAE':<10} | {'45-min MAE':<10} | {'60-min MAE':<10} | {'RMSE':<8} | {'MAPE (%)':<8}")
    print("-" * 105)
    for model_name, res in results["baselines"].items():
        t_res = res["test"]
        h_res = t_res["horizons"]
        print(
            f"{model_name.capitalize():<20} | "
            f"{h_res['5_min']['mae']:<10.2f} | "
            f"{h_res['15_min']['mae']:<10.2f} | "
            f"{h_res['30_min']['mae']:<10.2f} | "
            f"{h_res['45_min']['mae']:<10.2f} | "
            f"{h_res['60_min']['mae']:<10.2f} | "
            f"{t_res['overall']['rmse']:<8.2f} | "
            f"{t_res['overall']['mape']:<8.2f}"
        )
    print("======================================================================")

    # 9. Save Execution Report
    out_rep = Path(output_report_path)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    results["elapsed_seconds"] = round(time.time() - start_time, 2)
    with open(out_rep, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved P3 baseline results to: {out_rep}")
    
    return results


if __name__ == "__main__":
    run_p3_baseline_pipeline()
