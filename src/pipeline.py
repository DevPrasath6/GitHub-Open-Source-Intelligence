"""One-command pipeline: collect -> clean -> EDA -> train.

    python -m src.pipeline --sample                      # offline demo with synthetic data
    python -m src.pipeline --keyword "machine learning" --language Python --min-stars 100 --limit 500
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional

from src.analysis.eda import generate_all_figures
from src.config import setup_logging
from src.data_collection.github_api import collect_dataset, save_raw_data
from src.data_collection.sample_data import generate_sample_dataset
from src.data_processing.clean_data import load_cleaned_data, run_cleaning
from src.ml.train_model import train_models

logger = logging.getLogger(__name__)


def run_full_pipeline(sample: bool = True, keyword: str = "machine learning", language: Optional[str] = "Python",
                      min_stars: int = 100, limit: int = 500, token: Optional[str] = None,
                      enrich: bool = True) -> dict:
    """Run every stage and return the model metrics."""
    if sample:
        save_raw_data(generate_sample_dataset(max(limit, 300)), source="sample")
    else:
        df = collect_dataset(keyword, language or None, min_stars, limit, enrich, enrich, token)
        if df.empty:
            raise RuntimeError("No repositories collected - check network/token or use sample=True.")
        save_raw_data(df)
    run_cleaning()
    cleaned = load_cleaned_data()
    generate_all_figures(cleaned)
    return train_models(cleaned)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full GitHub OSINT data science pipeline.")
    p.add_argument("--sample", action="store_true", help="use synthetic data (no API needed)")
    p.add_argument("--keyword", default="machine learning")
    p.add_argument("--language", default="Python")
    p.add_argument("--min-stars", type=int, default=100)
    p.add_argument("--limit", type=int, default=500)
    a = p.parse_args()
    setup_logging()
    m = run_full_pipeline(a.sample, a.keyword, a.language, a.min_stars, a.limit)
    print("Pipeline finished. Best model:", m["best_model"])


if __name__ == "__main__":
    main()
