import json

res = json.load(open("reports/p4_results.json"))
comp = res["comparison_table"]
print(f"{'Model':42s} | {'MAE (km/h)':10s} | {'RMSE (km/h)':11s} | {'MAPE (%)':8s}")
print("-" * 80)
for k, v in comp.items():
    ov = v.get("overall", {})
    mae = ov.get("mae", 0.0)
    rmse = ov.get("rmse", 0.0)
    mape = ov.get("mape", 0.0)
    print(f"{k:42s} | {mae:10.4f} | {rmse:11.4f} | {mape:7.2f}%")
