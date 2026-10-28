"""Predict GitHub stars for a new repository.

Example:  python -m src.ml.predict
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import METRICS_PATH, MODEL_PATH
from src.data_processing.clean_data import normalize_language

DISCLAIMER = ("This is a statistical ESTIMATE based on past public data, "
              "not a guarantee of how many stars a repository will receive.")


def load_model(path: Path = MODEL_PATH):
    """Load the trained pipeline saved by ``train_model``."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: python -m src.ml.train_model")
    return joblib.load(path)


def load_metrics(path: Path = METRICS_PATH) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def interpret_prediction(stars: float, thresholds: Optional[dict] = None) -> str:
    """Translate a numeric estimate into Low / Moderate / High popularity."""
    low, high = (thresholds or {}).get("p33", 300), (thresholds or {}).get("p66", 1500)
    if stars < low:
        return "Low predicted popularity"
    if stars < high:
        return "Moderate predicted popularity"
    return "High predicted popularity"


def predict_stars(language: str, forks: int, watchers: int, open_issues: int,
                  repository_age_days: int, days_since_last_update: int,
                  has_license: bool = True, has_description: bool = True, has_topics: bool = True,
                  is_archived: bool = False, model=None, metrics: Optional[dict] = None) -> dict[str, Any]:
    """Predict stars for one repository. Raises ``ValueError`` on invalid input."""
    numeric = {"forks": forks, "watchers": watchers, "open_issues": open_issues,
               "repository_age_days": repository_age_days, "days_since_last_update": days_since_last_update}
    for name, value in numeric.items():
        if value is None or value < 0:
            raise ValueError(f"{name} must be a non-negative number (got {value!r})")

    model = model or load_model()
    metrics = metrics if metrics is not None else load_metrics()
    row = pd.DataFrame([{
        "language": normalize_language(language), **numeric,
        "has_license": int(has_license), "has_description": int(has_description),
        "has_topics": int(has_topics), "is_archived": int(is_archived),
    }])
    estimate = float(max(model.predict(row)[0], 0))
    low = estimate * float(np.exp(metrics.get("residual_q10", -0.5)))     # typical error band (10th-90th pct)
    high = estimate * float(np.exp(metrics.get("residual_q90", 0.5)))
    return {
        "predicted_stars": int(round(estimate)),
        "range_low": int(round(min(low, estimate))),
        "range_high": int(round(max(high, estimate))),
        "popularity": interpret_prediction(estimate, metrics.get("star_quantiles")),
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    result = predict_stars("Python", forks=250, watchers=120, open_issues=15,
                           repository_age_days=900, days_since_last_update=14)
    print(f"Predicted Stars: {result['predicted_stars']:,} "
          f"(typical range {result['range_low']:,} - {result['range_high']:,})")
    print(result["popularity"]); print(result["disclaimer"])
