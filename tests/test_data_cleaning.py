import numpy as np
import pandas as pd
import pytest

from src.data_collection.sample_data import generate_sample_dataset
from src.data_processing.clean_data import (clean_data, clean_description, clean_topics,
                                            normalize_language, profile_dataframe)


@pytest.fixture(scope="module")
def raw():
    return generate_sample_dataset(300)


def test_language_normalization():
    assert normalize_language("python") == "Python"
    assert normalize_language("JS") == "JavaScript"
    assert normalize_language("cpp") == "C++"
    assert normalize_language(None) == "Unknown"
    assert normalize_language(np.nan) == "Unknown"


def test_text_cleaners():
    assert clean_description("<b>Fast</b>  lib https://x.io  :rocket:") == "Fast lib"
    assert clean_description(None) == ""
    assert clean_topics("ML, ml ,Deep Learning;nlp") == "ml,deep-learning,nlp"


def test_cleaning_pipeline(raw):
    clean, report = clean_data(raw)
    assert report["before"]["duplicate_rows"] > 0
    assert clean["full_name"].is_unique
    assert clean.duplicated().sum() == 0
    assert (clean[["stars", "forks", "open_issues"]] >= 0).all().all()
    assert str(clean["created_at"].dtype).startswith("datetime64")
    assert clean["language"].isin(["python", "js", "cpp"]).sum() == 0
    assert clean["language"].notna().all()
    assert not {"visibility", "default_branch"} & set(clean.columns)
    assert clean["size"].max() < 50_000_000                       # outlier capped
    assert report["after"]["rows"] < report["before"]["rows"]


def test_leakage_guard_removes_fake_watchers():
    df = generate_sample_dataset(200, inject_noise=False)
    df["watchers"] = df["stars"]                                   # what the Search API would give
    clean, _ = clean_data(df)
    assert clean["watchers"].isna().all()


def test_profile_counts_missing():
    p = profile_dataframe(pd.DataFrame({"full_name": ["a", "a"], "x": [1, None]}))
    assert p["total_missing"] == 1 and p["duplicate_repositories"] == 1
