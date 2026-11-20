import torch
import json
from pathlib import Path

print("=== BACKUPS ===")
for p in [
    "artifacts/models/p4_dcrnn_1epoch_backup.pt",
    "artifacts/models/p4_dcrnn_epoch6_checkpoint.pt",
    "artifacts/models/p4_dcrnn_epoch9_checkpoint.pt"
]:
    f = Path(p)
    if f.exists():
        c = torch.load(f, map_location="cpu")
        print(f"{f.name}: exists=True, size={f.stat().st_size}, epoch={c.get('epoch')}, val_loss={c.get('val_loss'):.4f}")
    else:
        print(f"{f.name}: MISSING!")

print("\n=== ACTIVE CHECKPOINTS ===")
for p in [
    "artifacts/models/p4_dcrnn_best.pt",
    "artifacts/models/p4_dcrnn_latest.pt"
]:
    f = Path(p)
    if f.exists():
        c = torch.load(f, map_location="cpu")
        print(f"{f.name}: epoch={c.get('epoch')}, val_loss={c.get('val_loss'):.4f}, best_val_loss={c.get('best_val_loss')}, patience={c.get('patience_counter')}, has_model={'model_state_dict' in c}, has_opt={'optimizer_state_dict' in c}")
    else:
        print(f"{f.name}: MISSING!")

print("\n=== TRAINING HISTORY ===")
with open("reports/p4_training_history.json") as f:
    h = json.load(f)
print(f"Total epochs: {h.get('total_epochs')}, best_epoch: {h.get('best_epoch')}, best_val_loss: {h.get('best_val_loss')}")
records = h.get("epoch_records", [])
epochs = [r["epoch"] for r in records]
print(f"Recorded epochs: {epochs}")
print(f"Unique epochs check: {len(epochs) == len(set(epochs))}")
for r in records:
    print(f"  Epoch {r['epoch']:02d}: train={r['train_loss']:.4f}, val={r['val_loss']:.4f}, lr={r['learning_rate']:.6f}, time={r['elapsed_seconds']}s, improved={r.get('improved')}")
