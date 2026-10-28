"""Exploratory data analysis: static Matplotlib/Seaborn figures + interactive Plotly charts.

Run:  python -m src.analysis.eda      (saves PNGs to outputs/figures/)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend (works without a display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.config import FIGURES_DIR, DATA_PROCESSED, setup_logging
from src.ml.features import add_features, add_ratio_features_for_eda

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="deep")

CORR_COLUMNS = ["stars", "forks", "watchers", "open_issues", "size", "repository_age_days",
                "days_since_last_update", "followers", "topic_count",
                "fork_star_ratio", "issue_star_ratio", "watcher_star_ratio"]


def prepare_eda_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned data + engineered columns (incl. star ratios, which are EDA-only)."""
    return add_ratio_features_for_eda(add_features(df))


def top_languages(df: pd.DataFrame, n: int = 10) -> list[str]:
    return df["language"].value_counts().head(n).index.tolist()


# ----------------------------------------------------------------------------- static (PNG)
def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Create and save every EDA figure. Returns the list of file paths."""
    d = prepare_eda_frame(df)
    paths: list[Path] = []

    # --- popularity
    top = d.nlargest(15, "stars").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["repository_name"], top["stars"], color=sns.color_palette("deep")[0])
    ax.set(title="Top 15 Repositories by Stars", xlabel="Stars")
    paths.append(_save(fig, "top_repositories", out_dir))

    for col, name in [("stars", "stars_distribution"), ("forks", "forks_distribution")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(np.log10(d[col] + 1), bins=30, kde=True, ax=ax)
        ax.set(title=f"Distribution of {col.title()} (log10 scale)", xlabel=f"log10({col} + 1)")
        paths.append(_save(fig, name, out_dir))

    # --- languages
    langs = top_languages(d, 12)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=d[d["language"].isin(langs)], y="language", order=langs, ax=ax)
    ax.set(title="Repository Count by Language", xlabel="Repositories", ylabel="")
    paths.append(_save(fig, "language_repo_count", out_dir))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=d[d["language"].isin(langs[:10])], x="language", y="stars", order=langs[:10], ax=ax)
    ax.set(yscale="log", title="Stars by Programming Language (log scale)", xlabel="")
    ax.tick_params(axis="x", rotation=45)
    paths.append(_save(fig, "stars_by_language", out_dir))

    # --- activity
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    d.groupby(d["created_at"].dt.year).size().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set(title="Repositories Created per Year", xlabel="Year", ylabel="Repositories")
    d.groupby(d["pushed_at"].dt.tz_localize(None).dt.to_period("Q").astype(str)).size().plot(kind="line", marker="o", ax=axes[1])
    axes[1].set(title="Repositories by Last-Push Quarter", xlabel="Quarter", ylabel="Repositories")
    axes[1].tick_params(axis="x", rotation=60)
    paths.append(_save(fig, "repository_activity_over_time", out_dir))

    recent = d.nsmallest(15, "days_since_last_update").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(recent["repository_name"], recent["days_since_last_update"], color="seagreen")
    ax.set(title="Most Recently Updated Repositories", xlabel="Days since last push")
    paths.append(_save(fig, "recently_updated", out_dir))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(d["repository_age_days"] / 365.25, bins=30, ax=ax)
    ax.set(title="Repository Age Distribution", xlabel="Age (years)")
    paths.append(_save(fig, "age_distribution", out_dir))

    # --- relationships (log-log scatter)
    for col, name in [("forks", "stars_vs_forks"), ("watchers", "stars_vs_watchers"),
                      ("open_issues", "stars_vs_open_issues"), ("repository_age_days", "age_vs_stars")]:
        if col not in d or d[col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=d, x=col, y="stars", alpha=0.5, ax=ax)
        ax.set(xscale="symlog", yscale="log", title=f"Stars vs {col.replace('_', ' ').title()}")
        paths.append(_save(fig, name, out_dir))

    # --- correlation heatmap (Spearman: robust to skew)
    cols = [c for c in CORR_COLUMNS if c in d and d[c].notna().any()]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(d[cols].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman Correlation Heatmap (ratio columns are EDA-only)")
    paths.append(_save(fig, "correlation_heatmap", out_dir))

    logger.info("Saved %d figures to %s", len(paths), out_dir)
    return paths


# ----------------------------------------------------------------------------- interactive (Plotly)
def plotly_top_repos(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.nlargest(n, "stars").iloc[::-1]
    return px.bar(top, x="stars", y="repository_name", orientation="h", color="language",
                  hover_data=["forks", "full_name"], title=f"Top {n} Repositories by Stars")


def plotly_stars_by_language(df: pd.DataFrame, n: int = 10) -> go.Figure:
    sub = df[df["language"].isin(top_languages(df, n))]
    return px.box(sub, x="language", y="stars", log_y=True, color="language",
                  title="Stars by Programming Language (log scale)")


def plotly_stars_vs_forks(df: pd.DataFrame) -> go.Figure:
    return px.scatter(df, x="forks", y="stars", color="language", log_x=True, log_y=True, opacity=0.6,
                      hover_name="full_name", title="Stars vs Forks (log-log)")


def plotly_stars_distribution(df: pd.DataFrame) -> go.Figure:
    return px.histogram(df, x="stars", nbins=40, log_x=True, marginal="box",
                        title="Stars Distribution (log x-axis)")


def plotly_activity(df: pd.DataFrame) -> go.Figure:
    created = df.groupby(df["created_at"].dt.year).size().rename("Created")
    pushed = df.groupby(df["pushed_at"].dt.year).size().rename("Last pushed")
    act = pd.concat([created, pushed], axis=1).fillna(0).reset_index(names="year")
    return px.line(act, x="year", y=["Created", "Last pushed"], markers=True,
                   title="Repository Activity by Year", labels={"value": "Repositories"})


def plotly_correlation(df: pd.DataFrame) -> go.Figure:
    d = prepare_eda_frame(df)
    cols = [c for c in CORR_COLUMNS if c in d and d[c].notna().any()]
    return px.imshow(d[cols].corr(method="spearman").round(2), text_auto=True, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, title="Spearman Correlation (ratio columns are EDA-only)")


def main() -> None:
    from src.data_processing.clean_data import load_cleaned_data
    setup_logging()
    if not Path(DATA_PROCESSED).exists():
        raise SystemExit("Cleaned data not found. Run: python -m src.data_processing.clean_data")
    for p in generate_all_figures(load_cleaned_data()):
        print(p)


if __name__ == "__main__":
    main()
