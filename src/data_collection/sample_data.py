"""Synthetic GitHub-like dataset so the whole project works without API access.

The data is artificial (not real repositories). It is deliberately a little messy
(duplicates, missing values, wrong casing, invalid numbers) so the cleaning step has work to do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, RAW_COLUMNS

LANGUAGES = ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "C#",
             "Jupyter Notebook", "R", "Shell", "Julia"]
LANG_PROBS = np.array([30, 14, 10, 8, 7, 6, 5, 4, 8, 3, 3, 2], dtype=float)
LANG_EFFECT = dict(zip(LANGUAGES, [0.15, 0.05, 0.10, -0.05, 0.0, 0.20, 0.25, -0.15, 0.0, -0.20, -0.25, -0.10]))
TOPICS = ["machine-learning", "deep-learning", "nlp", "computer-vision", "data-science", "api",
          "cli", "web", "automation", "visualization", "security", "devops", "pytorch", "tensorflow"]
PREFIX = ["deep", "auto", "fast", "open", "neural", "smart", "data", "quick", "ml", "vision"]
SUFFIX = ["learn", "kit", "flow", "net", "lab", "hub", "pipe", "engine", "tools", "studio"]
LICENSES = ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause"]


def generate_sample_dataset(n_repos: int = 800, seed: int = RANDOM_STATE, inject_noise: bool = True) -> pd.DataFrame:
    """Create a synthetic raw dataset with the same columns as the real collector."""
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp("2025-01-01", tz="UTC")
    n = n_repos

    lang = rng.choice(LANGUAGES, n, p=LANG_PROBS / LANG_PROBS.sum())
    age_days = rng.integers(90, 3650, n)
    q = rng.normal(0, 1.1, n)                                    # hidden "quality" driving popularity
    has_license = rng.random(n) < (0.55 + 0.15 * (q > 0))
    since_push = np.minimum(rng.exponential(150, n).astype(int), age_days - 1)

    forks = np.exp(3.0 + 0.95 * q + rng.normal(0, 0.45, n)).astype(int)
    watchers = np.exp(2.6 + 0.85 * q + rng.normal(0, 0.45, n)).astype(int) + 1
    open_issues = rng.poisson(np.exp(1.4 + 0.45 * q))
    log_stars = (6.0 + 0.85 * q + 0.0002 * age_days + 0.25 * has_license - 0.0015 * since_push
                 + np.array([LANG_EFFECT[x] for x in lang]) + rng.normal(0, 0.35, n))
    stars = np.maximum(100, np.exp(log_stars).astype(int))

    # owners
    n_owners = max(50, n // 3)
    owner_idx = rng.integers(0, n_owners, n)
    owner_followers = np.exp(rng.normal(5, 1.5, n_owners)).astype(int)
    owner_repos = np.exp(rng.normal(3.5, 1.0, n_owners)).astype(int) + 1
    owner_org = rng.random(n_owners) < 0.3
    owner_age = rng.integers(365, 5000, n_owners)
    owner_login = np.array([f"dev{i:03d}" for i in owner_idx])

    names = [f"{rng.choice(PREFIX)}-{rng.choice(SUFFIX)}-{i}" for i in range(n)]
    topics = [",".join(rng.choice(TOPICS, rng.integers(1, 6), replace=False)) if rng.random() > 0.15 else ""
              for _ in range(n)]
    desc = [f"A {rng.choice(['fast', 'simple', 'modular', 'lightweight'])} {l} toolkit for "
            f"{rng.choice(['data analysis', 'model training', 'automation', 'visualization', 'APIs'])}."
            for l in lang]
    created = ref - pd.to_timedelta(age_days, unit="D")
    pushed = ref - pd.to_timedelta(since_push, unit="D")
    updated = np.minimum(pushed + pd.to_timedelta(rng.integers(0, 4, n), unit="D"), ref)
    fmt = "%Y-%m-%dT%H:%M:%SZ"

    df = pd.DataFrame({
        "repository_name": names,
        "full_name": [f"{o}/{nm}" for o, nm in zip(owner_login, names)],
        "owner": owner_login,
        "description": desc,
        "language": lang,
        "stars": stars, "forks": forks, "watchers": watchers, "open_issues": open_issues,
        "size": np.exp(rng.normal(8, 1.5, n)).astype(int),
        "created_at": created.strftime(fmt),
        "updated_at": pd.DatetimeIndex(updated).strftime(fmt),
        "pushed_at": pushed.strftime(fmt),
        "license": np.where(has_license, rng.choice(LICENSES, n), None),
        "topics": topics,
        "default_branch": rng.choice(["main", "master"], n, p=[0.7, 0.3]),
        "archived": rng.random(n) < 0.04,
        "is_fork": rng.random(n) < 0.08,
        "visibility": "public",
        "repository_url": [f"https://github.com/{o}/{nm}" for o, nm in zip(owner_login, names)],
        "owner_login": owner_login,
        "owner_type": np.where(owner_org[owner_idx], "Organization", "User"),
        "followers": owner_followers[owner_idx],
        "public_repositories": owner_repos[owner_idx],
        "account_created_at": (ref - pd.to_timedelta(owner_age[owner_idx], unit="D")).strftime(fmt),
        "collected_at": ref.strftime(fmt),
    })

    if inject_noise:
        df = _inject_noise(df, rng)
    return df[RAW_COLUMNS].reset_index(drop=True)


def _inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic data-quality problems for the cleaning pipeline to fix."""
    df = df.copy()
    n = len(df)

    def pick(frac: float) -> np.ndarray:
        return rng.choice(n, max(1, int(n * frac)), replace=False)

    df["description"] = df["description"].astype(object)
    df["language"] = df["language"].astype(object)
    for col, frac in [("description", 0.06), ("language", 0.04), ("watchers", 0.03), ("followers", 0.03)]:
        df.loc[pick(frac), col] = np.nan
    html_rows = pick(0.05)
    df.loc[html_rows, "description"] = ("<b>" + df.loc[html_rows, "description"].fillna("demo") +
                                        "</b> https://example.com  ")
    messy = {"Python": "python", "JavaScript": "js", "C++": "cpp", "TypeScript": "TYPESCRIPT"}
    for idx in pick(0.08):
        lang = df.at[idx, "language"]
        if lang in messy:
            df.at[idx, "language"] = messy[lang]
    df.loc[rng.choice(n, 3, replace=False), "open_issues"] = -1          # invalid values
    df.loc[rng.choice(n, 2, replace=False), "forks"] = -5
    df.loc[rng.choice(n, 2, replace=False), "size"] = 50_000_000          # extreme outliers
    df = pd.concat([df, df.iloc[pick(0.03)]], ignore_index=True)         # exact duplicates
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)
