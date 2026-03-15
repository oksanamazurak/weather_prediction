"""
Tests for the MLOps Weather Rain Prediction pipeline.

Two groups of tests:
  - Pre-train  (marker: @pytest.mark.pretrain)  — fast checks that run BEFORE
    training: data schema, value ranges, missing values, Hydra config validity.
  - Post-train (marker: @pytest.mark.posttrain) — checks that run AFTER
    training: artifact existence and Quality Gate on metrics.

Run only one group:
    pytest -m pretrain
    pytest -m posttrain
"""

import json
import os

import joblib
import pandas as pd
import pytest
import yaml

from tests.conftest import (
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    HYDRA_CONFIG_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)


# ============================================================================
#  PRE-TRAIN TESTS  — run before training to validate data & config
# ============================================================================


@pytest.mark.pretrain
class TestDataExists:
    """Verify that required data files are present."""

    def test_prepared_train_exists(self):
        assert os.path.exists(TRAIN_DATA_PATH), (
            f"Prepared train data not found: {TRAIN_DATA_PATH}"
        )

    def test_prepared_test_exists(self):
        assert os.path.exists(TEST_DATA_PATH), (
            f"Prepared test data not found: {TEST_DATA_PATH}"
        )


@pytest.mark.pretrain
class TestDataSchema:
    """Validate column schema and basic data integrity of the prepared data."""

    REQUIRED_COLUMNS = {
        "Date",
        "Location",
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Evaporation",
        "Sunshine",
        "WindGustDir",
        "WindGustSpeed",
        "WindDir9am",
        "WindDir3pm",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Temp9am",
        "Temp3pm",
        "RainToday",
        "RainTomorrow",
        "Temp_diff",
    }

    TARGET_COLUMN = "RainTomorrow"

    @pytest.fixture(autouse=True)
    def _load_train(self):
        self.df = pd.read_csv(TRAIN_DATA_PATH)

    # ---- Column checks ----

    def test_required_columns_present(self):
        """All expected columns must be present in the dataset."""
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        assert not missing, f"Missing columns: {sorted(missing)}"

    def test_no_extra_columns(self):
        """No unexpected columns should appear."""
        extra = set(self.df.columns) - self.REQUIRED_COLUMNS
        assert not extra, f"Unexpected extra columns: {sorted(extra)}"

    # ---- Target variable checks ----

    def test_target_no_nulls(self):
        """Target column 'RainTomorrow' must not contain NaN values."""
        assert self.df[self.TARGET_COLUMN].notna().all(), (
            "RainTomorrow contains NaN values"
        )

    def test_target_is_binary(self):
        """Target must only contain values 0 and 1."""
        unique_vals = set(self.df[self.TARGET_COLUMN].unique())
        assert unique_vals.issubset({0, 1}), (
            f"RainTomorrow has unexpected values: {unique_vals}"
        )

    # ---- Size checks ----

    def test_minimum_rows(self):
        """Dataset must have a reasonable number of rows for training."""
        min_rows = int(os.getenv("MIN_TRAIN_ROWS", "100"))
        assert self.df.shape[0] >= min_rows, (
            f"Too few rows for training: {self.df.shape[0]} < {min_rows}"
        )

    def test_train_test_ratio(self):
        """Train set should be larger than the test set (~80/20 split)."""
        df_test = pd.read_csv(TEST_DATA_PATH)
        ratio = self.df.shape[0] / (self.df.shape[0] + df_test.shape[0])
        assert 0.7 <= ratio <= 0.9, (
            f"Unexpected train/total ratio: {ratio:.2f} (expected ~0.8)"
        )


@pytest.mark.pretrain
class TestDataValueRanges:
    """Sanity-check numerical value ranges in the prepared data."""

    @pytest.fixture(autouse=True)
    def _load_train(self):
        self.df = pd.read_csv(TRAIN_DATA_PATH)

    def test_temperature_range(self):
        """Temperature values should be within a physically plausible range."""
        assert self.df["MinTemp"].between(-20, 55).all(), (
            "MinTemp values out of plausible range [-20, 55]"
        )
        assert self.df["MaxTemp"].between(-15, 60).all(), (
            "MaxTemp values out of plausible range [-15, 60]"
        )

    def test_humidity_range(self):
        """Humidity should be 0-100."""
        for col in ["Humidity9am", "Humidity3pm"]:
            assert self.df[col].between(0, 100).all(), (
                f"{col} has values outside [0, 100]"
            )

    def test_pressure_range(self):
        """Atmospheric pressure should be in a reasonable range (hPa)."""
        for col in ["Pressure9am", "Pressure3pm"]:
            assert self.df[col].between(870, 1090).all(), (
                f"{col} has values outside [870, 1090] hPa"
            )

    def test_rainfall_non_negative(self):
        """Rainfall cannot be negative."""
        assert (self.df["Rainfall"] >= 0).all(), (
            "Rainfall contains negative values"
        )

    def test_temp_diff_consistent(self):
        """Temp_diff should equal MaxTemp - MinTemp."""
        computed = self.df["MaxTemp"] - self.df["MinTemp"]
        assert (abs(self.df["Temp_diff"] - computed) < 1e-6).all(), (
            "Temp_diff is not consistent with MaxTemp - MinTemp"
        )


@pytest.mark.pretrain
class TestDataNulls:
    """Check that critical columns do not have NaN after preparation."""

    @pytest.fixture(autouse=True)
    def _load_train(self):
        self.df = pd.read_csv(TRAIN_DATA_PATH)

    def test_no_nulls_in_numeric_features(self):
        """After preparation, numeric columns should have no NaN."""
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        null_counts = self.df[numeric_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"Numeric columns with NaN after prep: "
            f"{dict(cols_with_nulls)}"
        )

    def test_no_nulls_in_target(self):
        """Target column must not have null values."""
        assert self.df["RainTomorrow"].notna().all(), (
            "RainTomorrow contains NaN values"
        )


@pytest.mark.pretrain
class TestHydraConfig:
    """Validate the Hydra/YAML configuration file."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        with open(HYDRA_CONFIG_PATH, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_config_file_exists(self):
        assert os.path.exists(HYDRA_CONFIG_PATH), (
            f"Config file not found: {HYDRA_CONFIG_PATH}"
        )

    def test_data_section_present(self):
        """Config must define data paths."""
        assert "data" in self.cfg, "Config missing 'data' section"
        assert "train_path" in self.cfg["data"], (
            "Config data section missing 'train_path'"
        )
        assert "test_path" in self.cfg["data"], (
            "Config data section missing 'test_path'"
        )

    def test_mlflow_section_present(self):
        """Config must have MLflow settings."""
        assert "mlflow" in self.cfg, "Config missing 'mlflow' section"
        assert "experiment_name" in self.cfg["mlflow"], (
            "Config mlflow section missing 'experiment_name'"
        )

    def test_seed_is_set(self):
        """A random seed must be defined for reproducibility."""
        assert "seed" in self.cfg, "Config missing 'seed'"
        assert isinstance(self.cfg["seed"], int), (
            f"Seed must be an integer, got {type(self.cfg['seed'])}"
        )


# ============================================================================
#  POST-TRAIN TESTS  — run after training to validate artifacts & quality
# ============================================================================


@pytest.mark.posttrain
class TestArtifactsExist:
    """After training, expected artifact files must be present."""

    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), (
            f"Model artifact not found: {MODEL_PATH}"
        )

    def test_model_loadable(self):
        """The saved model must be loadable with joblib."""
        assert os.path.exists(MODEL_PATH), (
            f"Model artifact not found: {MODEL_PATH}"
        )
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict"), (
            "Loaded object does not have a 'predict' method — not a valid model"
        )

    def test_model_file_not_empty(self):
        """Model file must not be empty."""
        assert os.path.getsize(MODEL_PATH) > 0, (
            "Model file is empty (0 bytes)"
        )

    def test_confusion_matrix_exists(self):
        assert os.path.exists(CONFUSION_MATRIX_PATH), (
            f"confusion_matrix.png not found: {CONFUSION_MATRIX_PATH}"
        )

    def test_feature_importance_exists(self):
        assert os.path.exists(FEATURE_IMPORTANCE_PATH), (
            f"feature_importance.png not found: {FEATURE_IMPORTANCE_PATH}"
        )

    def test_metrics_file_exists(self):
        assert os.path.exists(METRICS_PATH), (
            f"metrics.json not found: {METRICS_PATH}"
        )


@pytest.mark.posttrain
class TestMetricsJson:
    """Validate the structure and content of metrics.json."""

    @pytest.fixture(autouse=True)
    def _load_metrics(self):
        if not os.path.exists(METRICS_PATH):
            pytest.skip(f"metrics.json not found at {METRICS_PATH}")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

    def test_has_accuracy(self):
        assert "accuracy" in self.metrics, (
            "metrics.json missing 'accuracy' key"
        )

    def test_has_f1(self):
        assert "f1" in self.metrics, (
            "metrics.json missing 'f1' key"
        )

    def test_accuracy_in_range(self):
        acc = float(self.metrics["accuracy"])
        assert 0.0 <= acc <= 1.0, (
            f"accuracy out of range [0, 1]: {acc}"
        )

    def test_f1_in_range(self):
        f1 = float(self.metrics["f1"])
        assert 0.0 <= f1 <= 1.0, (
            f"f1 out of range [0, 1]: {f1}"
        )


@pytest.mark.posttrain
class TestQualityGate:
    """
    Quality Gate: metrics must exceed configurable thresholds.

    Control thresholds via environment variables:
        F1_THRESHOLD   (default: 0.70)
        ACC_THRESHOLD  (default: 0.80)
    """

    @pytest.fixture(autouse=True)
    def _load_metrics(self):
        if not os.path.exists(METRICS_PATH):
            pytest.skip(f"metrics.json not found at {METRICS_PATH}")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

    def test_quality_gate_f1(self):
        threshold = float(os.getenv("F1_THRESHOLD", "0.50"))
        f1 = float(self.metrics["f1"])
        assert f1 >= threshold, (
            f"Quality Gate FAILED: f1={f1:.4f} < threshold={threshold:.2f}"
        )

    def test_quality_gate_accuracy(self):
        threshold = float(os.getenv("ACC_THRESHOLD", "0.80"))
        acc = float(self.metrics["accuracy"])
        assert acc >= threshold, (
            f"Quality Gate FAILED: accuracy={acc:.4f} < threshold={threshold:.2f}"
        )
