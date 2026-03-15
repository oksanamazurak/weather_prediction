import pandas as pd
import sys
import os
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# =========================
# INPUT / OUTPUT
# =========================

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

# =========================
# CI MODE
# =========================

CI = os.getenv("CI", "false") == "true"

# =========================
# LOAD DATA
# =========================

train = pd.read_csv(os.path.join(input_dir, "train.csv"))
test = pd.read_csv(os.path.join(input_dir, "test.csv"))

# Якщо потрібно — можна обмежити дані в CI
if CI:
    train = train.head(5000)
    test = test.head(2000)

X_train = train.drop("RainTomorrow", axis=1)
y_train = train["RainTomorrow"]

X_test = test.drop("RainTomorrow", axis=1)
y_test = test["RainTomorrow"]

# =========================
# MLflow
# =========================

mlflow.set_experiment("Weather_Rain_Prediction")

with mlflow.start_run():

    mlflow.set_tag("author", "student")
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "WeatherAUS")
    mlflow.set_tag("ci_mode", str(CI))

    # Менша модель у CI
    n_estimators = 50 if CI else 100
    max_depth = 8 if CI else 15

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_score", test_f1)

    # Save model to MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Save model locally (для CI тестів)
    model_path = os.path.join(output_dir, "rf_model.pkl")
    joblib.dump(model, model_path)

    # =========================
    # SAVE METRICS FOR PYTEST
    # =========================

    metrics = {
        "accuracy": float(test_accuracy),
        "f1": float(test_f1)
    }

    # metrics.json у корені проєкту
    metrics_path = os.path.abspath("metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # =========================
    # QUALITY GATE
    # =========================

    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))

    if test_f1 < threshold:
        raise ValueError(
            f"Quality Gate failed: F1={test_f1:.4f} < {threshold}"
        )

    print("Test Accuracy:", test_accuracy)
    print("Test F1 Score:", test_f1)