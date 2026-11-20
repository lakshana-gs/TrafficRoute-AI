import json

res = json.load(open("reports/p4_results.json"))
p3_res = json.load(open("reports/p3_baseline_results.json"))["baselines"]

models_data = {
    "Persistence (P3)": p3_res["persistence"]["test"]["horizons"],
    "Hist Mean (P3)": p3_res["historical_mean"]["test"]["horizons"],
    "XGBoost (P3)": p3_res["xgboost"]["test"]["horizons"],
    "Temporal GRU (P4)": res["models"]["temporal_only_gru"]["test_horizons"],
    "DCRNN Ep 1": res["models"]["dcrnn_1epoch"]["test_horizons"],
    "DCRNN Ep 6": res["models"]["dcrnn_epoch6"]["test_horizons"],
    "DCRNN Ep 9": res["models"]["dcrnn_epoch9_best"]["test_horizons"],
    "DCRNN Ep 19 (Final)": res["models"]["dcrnn_final_best"]["test_horizons"],
}

print(f"{'Horizon':8s} | {'Pers':7s} | {'HistM':7s} | {'XGB':7s} | {'TempGRU':7s} | {'Ep 1':7s} | {'Ep 6':7s} | {'Ep 9':7s} | {'Ep 19 (Final)':13s}")
print("-" * 88)

for h in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
    key = f"{h}_min"
    vals = [f"{models_data[m][key]['mae']:7.4f}" for m in models_data]
    print(f"{h:2d} min   | " + " | ".join(vals))
