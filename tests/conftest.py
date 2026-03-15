"""
Shared fixtures and constants for the test suite.
"""
import os
import pytest

# ---------------------------------------------------------------------------
# Project root (repo root, one level above tests/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Default paths — can be overridden via environment variables
# ---------------------------------------------------------------------------
TRAIN_DATA_PATH = os.getenv(
    "TRAIN_DATA_PATH",
    os.path.join(PROJECT_ROOT, "data", "prepared", "train.csv"),
)
TEST_DATA_PATH = os.getenv(
    "TEST_DATA_PATH",
    os.path.join(PROJECT_ROOT, "data", "prepared", "test.csv"),
)
RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH",
    os.path.join(PROJECT_ROOT, "data", "raw", "weatherAUS.csv"),
)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "data", "models", "rf_model.pkl"),
)
METRICS_PATH = os.getenv(
    "METRICS_PATH",
    os.path.join(PROJECT_ROOT, "metrics.json"),
)
CONFUSION_MATRIX_PATH = os.getenv(
    "CONFUSION_MATRIX_PATH",
    os.path.join(PROJECT_ROOT, "confusion_matrix.png"),
)
FEATURE_IMPORTANCE_PATH = os.getenv(
    "FEATURE_IMPORTANCE_PATH",
    os.path.join(PROJECT_ROOT, "feature_importance.png"),
)
HYDRA_CONFIG_PATH = os.getenv(
    "HYDRA_CONFIG_PATH",
    os.path.join(PROJECT_ROOT, "config", "config.yaml"),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def train_data_path():
    return TRAIN_DATA_PATH


@pytest.fixture(scope="session")
def test_data_path():
    return TEST_DATA_PATH
