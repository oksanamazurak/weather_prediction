import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


# ==============================
# НАЛАШТУВАННЯ
# ==============================

EXPERIMENT_NAME = "weather_experiment"  # <-- заміни
METRIC_NAME = "accuracy"  # якщо у тебе інша метрика — зміни тут


# ==============================
# ЗАВАНТАЖЕННЯ RUNS
# ==============================

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # якщо потрібно
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("Experiment not found!")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.sampler != ''"
)

# ==============================
# ФОРМУВАННЯ DATAFRAME
# ==============================

data = []

for run in runs:
    sampler = run.data.tags.get("sampler")
    metric = run.data.metrics.get(METRIC_NAME)

    if sampler and metric is not None:
        data.append({
            "run_id": run.info.run_id,
            "sampler": sampler,
            "metric": metric,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time
        })

df = pd.DataFrame(data)

if df.empty:
    raise ValueError("No runs with sampler tag found!")


# ==============================
# ОБЧИСЛЕННЯ СТАТИСТИКИ
# ==============================

summary = df.groupby("sampler").agg(
    best_value=("metric", "max"),
    mean_value=("metric", "mean"),
    median_value=("metric", "median"),
    std_value=("metric", "std"),
    trials_count=("metric", "count")
)

print("\n===== SAMPLER COMPARISON =====\n")
print(summary)


# ==============================
# ЧАС ВИКОНАННЯ
# ==============================

df["duration_sec"] = (df["end_time"] - df["start_time"]) / 1000.0

time_summary = df.groupby("sampler")["duration_sec"].sum()

print("\n===== TOTAL TIME (seconds) =====\n")
print(time_summary)


# ==============================
# BEST-SO-FAR ГРАФІК
# ==============================

plt.figure()

for sampler in df["sampler"].unique():
    sampler_df = df[df["sampler"] == sampler].sort_values("start_time")
    
    best_so_far = sampler_df["metric"].cummax()
    
    plt.plot(
        range(1, len(best_so_far) + 1),
        best_so_far,
        label=sampler
    )

plt.xlabel("Trial")
plt.ylabel("Best so far accuracy")
plt.title("Best-so-far Comparison")
plt.legend()
plt.grid(True)

plt.show()