import json
import sys
import os

pr_metrics_path = "metrics.json"
baseline_path = "baseline/metrics.json"

if not os.path.exists(baseline_path):
    raise FileNotFoundError("Baseline metrics not found")

if not os.path.exists(pr_metrics_path):
    raise FileNotFoundError("PR metrics not found")

with open(baseline_path, "r") as f:
    baseline = json.load(f)

with open(pr_metrics_path, "r") as f:
    current = json.load(f)

# Порівняння
results = []

for metric in ["accuracy", "f1"]:
    base_value = float(baseline.get(metric, 0))
    curr_value = float(current.get(metric, 0))
    delta = curr_value - base_value

    results.append({
        "metric": metric,
        "baseline": base_value,
        "current": curr_value,
        "delta": delta
    })

# Вивід таблиці
print("\nRegression Comparison:\n")
print("| Metric | Baseline | Current | Δ |")
print("|--------|----------|----------|------|")

for r in results:
    print(f"| {r['metric']} | {r['baseline']:.4f} | {r['current']:.4f} | {r['delta']:+.4f} |")

# Optional: fail if деградація
for r in results:
    if r["delta"] < -0.01:  # допустимий поріг
        raise ValueError(
            f"Regression detected in {r['metric']}: Δ={r['delta']:.4f}"
        )