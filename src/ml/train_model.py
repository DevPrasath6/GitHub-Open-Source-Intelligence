"""Train, evaluate and save the GitHub star predictor.

Run:  python -m src.ml.train_model
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (DATA_PROCESSED, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH,
                        RANDOM_STATE, TEST_SIZE, setup_logging)
from src.data_processing.clean_data import load_cleaned_data
from src.ml.features import add_features, build_preprocessor, get_model_columns, TARGET

logger = logging.getLogger(__name__)


def build_models(columns: list[str]) -> dict[str, TransformedTargetRegressor]:
    """Two pipelines. ``TransformedTargetRegressor`` trains on log1p(stars) and returns
    predictions already converted back to stars with expm1()."""
    def wrap(estimator) -> TransformedTargetRegressor:
        pipe = Pipeline([("prep", build_preprocessor(columns)), ("model", estimator)])
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

    return {
        "Linear Regression": wrap(LinearRegression()),
        "Random Forest": wrap(RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                    n_jobs=-1, random_state=RANDOM_STATE)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics on the original star scale (+ log scale, which is fairer for skewed data)."""
    y_pred = np.clip(y_pred, 0, None)
    mse = mean_squared_error(y_true, y_pred)
    ly_t, ly_p = np.log1p(y_true), np.log1p(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(ly_t, ly_p))),
        "R2_log": float(r2_score(ly_t, ly_p)),
    }


def _feature_importance(model: TransformedTargetRegressor) -> dict[str, float]:
    """Random-Forest importances, with one-hot language columns summed into 'language'."""
    pipe = model.regressor_
    names = pipe.named_steps["prep"].get_feature_names_out()
    grouped: dict[str, float] = {}
    for name, imp in zip(names, pipe.named_steps["model"].feature_importances_):
        key = name.split("__", 1)[-1]
        key = "language" if key.startswith("language_") else key
        grouped[key] = grouped.get(key, 0.0) + float(imp)
    return dict(sorted(grouped.items(), key=lambda kv: kv[1], reverse=True))


def train_models(df: pd.DataFrame, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH,
                 predictions_path: Path = PREDICTIONS_PATH) -> dict[str, Any]:
    """Full training workflow. Returns the metrics dictionary (also saved as JSON)."""
    feats = add_features(df)
    columns = get_model_columns(feats)
    X, y = feats[columns], feats[TARGET]
    logger.info("Features used: %s", columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    results, fitted, preds = {}, {}, {}
    for name, model in build_models(columns).items():
        model.fit(X_train, y_train)
        preds[name] = np.clip(model.predict(X_test), 0, None)
        results[name] = evaluate(y_test.to_numpy(), preds[name])
        fitted[name] = model
        logger.info("%s -> %s", name, {k: round(v, 3) for k, v in results[name].items()})

    # Model selection: lowest RMSE on the log scale (robust to a few mega-popular repos)
    best = min(results, key=lambda m: results[m]["RMSE_log"])
    log_resid = np.log1p(y_test.to_numpy()) - np.log1p(preds[best])

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "log1p(stars) -> expm1 back to stars",
        "features": columns,
        "n_total": int(len(X)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "models": results,
        "best_model": best,
        "feature_importance": _feature_importance(fitted["Random Forest"]),
        "residual_q10": float(np.quantile(log_resid, 0.10)),     # used for the estimate range
        "residual_q90": float(np.quantile(log_resid, 0.90)),
        "star_quantiles": {"p33": float(y_train.quantile(0.33)), "p66": float(y_train.quantile(0.66))},
        "languages": sorted(df["language"].dropna().unique().tolist()),
    }

    model_path, metrics_path, predictions_path = Path(model_path), Path(metrics_path), Path(predictions_path)
    for p in (model_path, metrics_path, predictions_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "full_name": feats.loc[X_test.index, "full_name"].to_numpy(),
        "actual": y_test.to_numpy(),
        "linear_regression": preds["Linear Regression"],
        "random_forest": preds["Random Forest"],
    }).to_csv(predictions_path, index=False)
    logger.info("Best model: %s - saved to %s", best, model_path)
    return metrics


def comparison_table(metrics: dict) -> pd.DataFrame:
    """Model comparison as a DataFrame (rows = models)."""
    return pd.DataFrame(metrics["models"]).T[["MAE", "MSE", "RMSE", "R2", "RMSE_log", "R2_log"]]


def main() -> None:
    setup_logging()
    if not Path(DATA_PROCESSED).exists():
        raise SystemExit("Cleaned data not found. Run: python -m src.data_processing.clean_data")
    metrics = train_models(load_cleaned_data())
    print(f"\nTrain rows: {metrics['n_train']} | Test rows: {metrics['n_test']}\n")
    print(comparison_table(metrics).round(3).to_string())
    print(f"\nBest model: {metrics['best_model']}")


if __name__ == "__main__":
    main()
