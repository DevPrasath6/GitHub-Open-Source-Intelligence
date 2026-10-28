"""Data cleaning pipeline: raw GitHub CSV -> analysis-ready CSV.

Run:  python -m src.data_processing.clean_data
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CLEANING_REPORT, DATA_PROCESSED, DATA_RAW, setup_logging

logger = logging.getLogger(__name__)

NUMERIC_COLS = ["stars", "forks", "watchers", "open_issues", "size", "followers", "public_repositories"]
DATE_COLS = ["created_at", "updated_at", "pushed_at", "account_created_at", "collected_at"]
BOOL_COLS = ["archived", "is_fork"]
OUTLIER_COLS = ["size", "open_issues", "followers", "public_repositories"]  # capped, NOT stars/forks
DROP_COLS = ["visibility", "default_branch", "owner"]   # constant / duplicated information
OUTLIER_QUANTILE = 0.995

LANGUAGE_ALIASES = {
    "js": "JavaScript", "javascript": "JavaScript", "node": "JavaScript", "ts": "TypeScript",
    "typescript": "TypeScript", "py": "Python", "python": "Python", "python3": "Python",
    "c++": "C++", "cpp": "C++", "c#": "C#", "csharp": "C#", "golang": "Go", "go": "Go",
    "rust": "Rust", "java": "Java", "r": "R", "bash": "Shell", "shell": "Shell", "sh": "Shell",
    "jupyter notebook": "Jupyter Notebook", "ipynb": "Jupyter Notebook", "julia": "Julia",
    "c": "C", "html": "HTML", "css": "CSS", "php": "PHP", "ruby": "Ruby", "swift": "Swift",
    "kotlin": "Kotlin", "scala": "Scala",
}
_MISSING_TOKENS = {"", "none", "nan", "null", "n/a", "unknown"}


# ----------------------------------------------------------------------------- helpers
def normalize_language(value: Any) -> str:
    """Map spelling variants (``python``, ``js``, ``CPP``) to one canonical name."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    text = str(value).strip()
    key = text.lower()
    if key in _MISSING_TOKENS:
        return "Unknown"
    return LANGUAGE_ALIASES.get(key, text)


def clean_description(value: Any) -> str:
    """Strip HTML tags, URLs, emoji short-codes and extra whitespace."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r":[a-z0-9_+\-]+:", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_topics(value: Any) -> str:
    """Lower-case, de-duplicate and tidy a comma/pipe/semicolon separated topic list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    seen: list[str] = []
    for tok in re.split(r"[,|;]", str(value)):
        tok = re.sub(r"\s+", "-", tok.strip().lower())
        if tok and tok not in seen:
            seen.append(tok)
    return ",".join(seen)


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Summary statistics used for the before/after cleaning report."""
    missing = df.isna().sum()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": {c: int(v) for c, v in missing.items() if v > 0},
        "total_missing": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_repositories": int(df["full_name"].duplicated().sum()) if "full_name" in df else 0,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


# ----------------------------------------------------------------------------- pipeline
def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline. Returns ``(clean_df, report)``."""
    steps: dict[str, Any] = {}
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    before = profile_dataframe(df)

    # 1. numeric conversion + invalid values (negative counts are impossible)
    for col in [c for c in NUMERIC_COLS if c in df]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        bad = df[col] < 0
        if bad.any():
            steps[f"invalid_negative_{col}"] = int(bad.sum())
            df.loc[bad, col] = np.nan

    # 2. dates -> timezone-aware datetime
    for col in [c for c in DATE_COLS if c in df]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 3. drop rows that cannot be analysed
    critical = [c for c in ("full_name", "stars", "created_at") if c in df]
    n = len(df)
    df = df.dropna(subset=critical)
    steps["rows_dropped_missing_critical"] = n - len(df)

    # 4. duplicates (exact rows first, then same repository)
    n = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="full_name", keep="first")
    steps["duplicates_removed"] = n - len(df)

    # 5. booleans & categorical normalisation
    for col in [c for c in BOOL_COLS if c in df]:
        df[col] = _to_bool(df[col])
    df["language"] = df["language"].map(normalize_language)
    df["description"] = df["description"].map(clean_description)
    df["topics"] = df["topics"].map(clean_topics)
    df["license"] = df["license"].astype("object").where(df["license"].notna(), "No License")
    df["license"] = df["license"].replace({"NOASSERTION": "Other", "": "No License"})
    if "owner_type" in df:
        df["owner_type"] = df["owner_type"].astype("object").where(df["owner_type"].notna(), "Unknown")

    # 6. date consistency
    df["pushed_at"] = df["pushed_at"].fillna(df["updated_at"]).fillna(df["created_at"])
    df["updated_at"] = df["updated_at"].fillna(df["pushed_at"])
    wrong_order = df["updated_at"] < df["created_at"]
    steps["updated_before_created_fixed"] = int(wrong_order.sum())
    df.loc[wrong_order, "updated_at"] = df.loc[wrong_order, "created_at"]

    # 7. data-leakage guard: Search-API 'watchers' is a copy of stars -> unusable
    if "watchers" in df and df["watchers"].notna().any() and (df["watchers"] == df["stars"]).mean() > 0.95:
        logger.warning("'watchers' equals 'stars' - removing it to avoid target leakage.")
        df["watchers"] = np.nan
        steps["watchers_removed_leakage"] = True

    # 8. missing numeric values -> median (only if the column has any data)
    for col in [c for c in NUMERIC_COLS if c in df and c != "stars"]:
        if df[col].notna().any():
            steps[f"imputed_{col}"] = int(df[col].isna().sum())
            df[col] = df[col].fillna(df[col].median())

    # 9. extreme outliers: winsorise (cap) at the 99.5th percentile
    capped = 0
    for col in [c for c in OUTLIER_COLS if c in df and df[col].notna().any()]:
        limit = df[col].quantile(OUTLIER_QUANTILE)
        mask = df[col] > limit
        capped += int(mask.sum())
        df.loc[mask, col] = limit
    steps["outlier_values_capped"] = capped

    # 10. irrelevant columns, integer types, tidy index
    df = df.drop(columns=[c for c in DROP_COLS if c in df])
    for col in [c for c in NUMERIC_COLS if c in df and df[c].notna().all()]:
        df[col] = df[col].round().astype("int64")
    df = df.sort_values("stars", ascending=False).reset_index(drop=True)

    report = {"before": before, "after": profile_dataframe(df), "steps": steps}
    return df, report


def load_cleaned_data(path: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Load the cleaned CSV with proper dtypes (dates, booleans, empty strings)."""
    df = pd.read_csv(path)
    for col in [c for c in DATE_COLS if c in df]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    for col in ("description", "topics", "license", "language"):
        if col in df:
            df[col] = df[col].fillna("" if col in ("description", "topics") else "Unknown")
    for col in [c for c in BOOL_COLS if c in df]:
        df[col] = _to_bool(df[col])
    return df


def run_cleaning(raw_path: Path = DATA_RAW, out_path: Path = DATA_PROCESSED,
                 report_path: Path = CLEANING_REPORT) -> tuple[pd.DataFrame, dict]:
    """Read raw CSV, clean it, write cleaned CSV + JSON report."""
    raw_path, out_path = Path(raw_path), Path(out_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run data collection first "
                                "(python -m src.data_collection.github_api [--sample]).")
    raw = pd.read_csv(raw_path)
    clean, report = clean_data(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    Path(report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("Cleaned %d -> %d rows; saved %s", report["before"]["rows"], report["after"]["rows"], out_path)
    return clean, report


def main() -> None:
    setup_logging()
    _, rep = run_cleaning()
    for k in ("before", "after"):
        p = rep[k]
        print(f"{k.upper():6s} rows={p['rows']} cols={p['columns']} missing={p['total_missing']} "
              f"duplicates={p['duplicate_rows']}")
    print("Steps:", json.dumps(rep["steps"], indent=2))


if __name__ == "__main__":
    main()
