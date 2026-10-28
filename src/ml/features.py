"""Feature engineering & preprocessing (no target leakage).

The target is ``stars``. Nothing derived from ``stars`` may be a model input.
Ratio features such as fork_star_ratio contain the target in the denominator, so they
are generated for EDA only (``add_ratio_features_for_eda``) and are blocked from the model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "stars"
CATEGORICAL_FEATURES = ["language"]
COUNT_FEATURES = ["forks", "watchers", "open_issues"]                    # skewed -> log1p + scale
DAY_FEATURES = ["repository_age_days", "days_since_last_update"]        # scale
BINARY_FEATURES = ["has_license", "has_description", "has_topics", "is_archived"]
MODEL_FEATURES = CATEGORICAL_FEATURES + COUNT_FEATURES + DAY_FEATURES + BINARY_FEATURES


def add_features(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Add engineered, leakage-free features to a cleaned dataframe."""
    out = df.copy()
    if reference_date is None:
        if "collected_at" in out and out["collected_at"].notna().any():
            reference_date = out["collected_at"].max()
        else:
            reference_date = out["pushed_at"].max()
    created = pd.to_datetime(out["created_at"], utc=True)
    pushed = pd.to_datetime(out["pushed_at"], utc=True)

    out["repository_age_days"] = (reference_date - created).dt.days.clip(lower=0)
    out["days_since_creation"] = out["repository_age_days"]          # alias requested in the brief
    out["days_since_last_update"] = (reference_date - pushed).dt.days.clip(lower=0)
    out["has_license"] = (~out["license"].fillna("No License").isin(["No License", ""])).astype(int)
    out["has_description"] = (out["description"].fillna("").str.len() > 0).astype(int)
    out["has_topics"] = (out["topics"].fillna("").str.len() > 0).astype(int)
    out["topic_count"] = out["topics"].fillna("").map(lambda s: len([t for t in s.split(",") if t]))
    out["is_archived"] = out["archived"].astype(int) if "archived" in out else 0
    return out


def add_ratio_features_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Star-based ratios. LEAKAGE-PRONE -> for exploration only, never for training."""
    out = df.copy()
    out["fork_star_ratio"] = out["forks"] / (out["stars"] + 1)
    out["issue_star_ratio"] = out["open_issues"] / (out["stars"] + 1)
    if "watchers" in out:
        out["watcher_star_ratio"] = out["watchers"] / (out["stars"] + 1)
    return out


def assert_no_leakage(columns: list[str]) -> None:
    """Raise if any feature name reveals the target."""
    bad = [c for c in columns if TARGET in c.lower()]
    if bad:
        raise ValueError(f"Target leakage: {bad} must not be used as model inputs.")


def get_model_columns(df: pd.DataFrame) -> list[str]:
    """Model features that exist in ``df`` and contain data."""
    cols = [c for c in MODEL_FEATURES if c in df and df[c].notna().any()]
    assert_no_leakage(cols)
    return cols


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for modelling from a cleaned dataframe."""
    feats = add_features(df)
    cols = get_model_columns(feats)
    return feats[cols], feats[TARGET]


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """ColumnTransformer: impute + log1p + scale counts, scale days, one-hot language."""
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    transformers = []
    counts = [c for c in COUNT_FEATURES if c in columns]
    if counts:
        transformers.append(("counts", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                 ("log", log1p), ("scale", StandardScaler())]), counts))
    days = [c for c in DAY_FEATURES if c in columns]
    if days:
        transformers.append(("days", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                ("scale", StandardScaler())]), days))
    binary = [c for c in BINARY_FEATURES if c in columns]
    if binary:
        transformers.append(("binary", "passthrough", binary))
    cats = [c for c in CATEGORICAL_FEATURES if c in columns]
    if cats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                  min_frequency=5, sparse_output=False), cats))
    return ColumnTransformer(transformers, remainder="drop")
