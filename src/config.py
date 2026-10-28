"""Central configuration: file paths, constants and logging setup."""
from __future__ import annotations

import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT_DIR / "data" / "raw" / "github_repositories.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed" / "cleaned_github_data.csv"
CLEANING_REPORT = ROOT_DIR / "data" / "processed" / "cleaning_report.json"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
MODEL_PATH = ROOT_DIR / "models" / "github_star_predictor.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
PREDICTIONS_PATH = ROOT_DIR / "models" / "test_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Column order of the raw dataset written by the collector / sample generator.
RAW_COLUMNS = [
    "repository_name", "full_name", "owner", "description", "language",
    "stars", "forks", "watchers", "open_issues", "size",
    "created_at", "updated_at", "pushed_at", "license", "topics",
    "default_branch", "archived", "is_fork", "visibility", "repository_url",
    "owner_login", "owner_type", "followers", "public_repositories",
    "account_created_at", "collected_at",
]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once for CLI scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
