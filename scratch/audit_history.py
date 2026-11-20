import json

h = json.load(open("reports/p4_training_history.json"))
records = h["epoch_records"]
print(f"Total records in history: {len(records)}")
assert len(records) == 20, f"Expected 20 records, found {len(records)}"
for r in records:
    print(f"Epoch {r['epoch']:2d}: train_loss={r['train_loss']:6.4f}, val_loss={r['val_loss']:6.4f}, time={r['elapsed_seconds']:7.2f}s, improved={r.get('improved')}")
