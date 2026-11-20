import torch
import json
import math
from pathlib import Path

print("=== CHECKPOINT AUDIT ===")
for p in [
    "artifacts/models/p4_dcrnn_1epoch_backup.pt",
    "artifacts/models/p4_dcrnn_epoch6_checkpoint.pt",
    "artifacts/models/p4_dcrnn_epoch9_checkpoint.pt",
    "artifacts/models/p4_dcrnn_best.pt",
    "artifacts/models/p4_dcrnn_latest.pt"
]:
    f = Path(p)
    if f.exists():
        c = torch.load(f, map_location="cpu")
        print(f"{f.name:32s}: epoch={c.get('epoch'):2d}, val_loss={c.get('val_loss'):.4f}, size={f.stat().st_size} bytes")
    else:
        print(f"{f.name:32s}: MISSING!")

print("\n=== HISTORY AUDIT ===")
with open("reports/p4_training_history.json") as f:
    h = json.load(f)

print(f"Model: {h.get('model_name')}")
print(f"Total epochs: {h.get('total_epochs')}")
print(f"Best epoch: {h.get('best_epoch')}")
print(f"Best val loss: {h.get('best_val_loss')}")
print(f"Total duration: {h.get('total_duration_seconds')}s ({h.get('total_duration_seconds')/3600:.2f} hours)")

records = h.get("epoch_records", [])
epochs = [r["epoch"] for r in records]
print(f"Recorded epochs count: {len(epochs)}")
print(f"Recorded epochs sequence: {epochs}")
assert len(epochs) == 20, f"Expected 20 epochs, got {len(epochs)}"
assert epochs == list(range(1, 21)), f"Epochs not sequential 1..20: {epochs}"

# Check for NaN / Inf
for r in records:
    for k in ["train_loss", "val_loss", "learning_rate", "elapsed_seconds"]:
        val = r[k]
        assert math.isfinite(val), f"Non-finite value {k}={val} in epoch {r['epoch']}"

print("All 20 epoch records are finite and valid.")

print("\n=== RESULTS AUDIT ===")
with open("reports/p4_results.json") as f:
    res = json.load(f)

final_best = res["models"]["dcrnn_final_best"]
print(f"Final Best Model Epoch: {final_best['best_epoch']}")
print(f"Final Best Val Loss: {final_best['best_val_loss']:.4f} km/h")
print(f"Test Overall MAE: {final_best['test_overall']['mae']:.4f} km/h")
print(f"Test Overall RMSE: {final_best['test_overall']['rmse']:.4f} km/h")
print(f"Test Overall MAPE: {final_best['test_overall']['mape']:.4f}%")

print("\n=== MULTI-HORIZON TEST MAE (km/h) ===")
horizons = final_best["test_horizons"]
for h_key, h_val in horizons.items():
    print(f"  {h_key:10s}: MAE={h_val['mae']:.4f}, RMSE={h_val['rmse']:.4f}, MAPE={h_val['mape']:.4f}%")
