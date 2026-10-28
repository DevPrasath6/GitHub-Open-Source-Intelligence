import numpy as np
import pytest

from src.data_collection.sample_data import generate_sample_dataset
from src.data_processing.clean_data import clean_data
from src.ml.features import add_features, assert_no_leakage, get_model_columns
from src.ml.predict import interpret_prediction, predict_stars
from src.ml.train_model import train_models


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    clean, _ = clean_data(generate_sample_dataset(400))
    metrics = train_models(clean, tmp / "m.pkl", tmp / "metrics.json", tmp / "pred.csv")
    import joblib
    return clean, metrics, joblib.load(tmp / "m.pkl")


def test_features_have_no_leakage(trained):
    clean, metrics, _ = trained
    cols = get_model_columns(add_features(clean))
    assert "stars" not in cols and not any("ratio" in c for c in cols)
    with pytest.raises(ValueError):
        assert_no_leakage(["forks", "stars_per_fork"])


def test_split_sizes_and_metrics(trained):
    _, m, _ = trained
    assert m["n_train"] + m["n_test"] == m["n_total"]
    assert abs(m["n_test"] / m["n_total"] - 0.2) < 0.02
    assert set(m["models"]) == {"Linear Regression", "Random Forest"}
    assert m["models"][m["best_model"]]["R2_log"] > 0.3


def test_prediction_output(trained):
    _, m, model = trained
    res = predict_stars("Python", 200, 80, 10, 900, 10, model=model, metrics=m)
    assert res["predicted_stars"] > 0
    assert res["range_low"] <= res["predicted_stars"] <= res["range_high"]
    assert "popularity" in res["popularity"]
    assert "ESTIMATE" in res["disclaimer"]


def test_more_forks_means_more_stars(trained):
    _, m, model = trained
    lo = predict_stars("Python", 5, 3, 1, 900, 10, model=model, metrics=m)["predicted_stars"]
    hi = predict_stars("Python", 2000, 800, 50, 900, 10, model=model, metrics=m)["predicted_stars"]
    assert hi > lo


def test_invalid_input_and_unknown_language(trained):
    _, m, model = trained
    with pytest.raises(ValueError):
        predict_stars("Python", -1, 1, 1, 10, 10, model=model, metrics=m)
    assert predict_stars("Brainfuck", 10, 5, 1, 100, 5, model=model, metrics=m)["predicted_stars"] >= 0


def test_interpretation_thresholds():
    q = {"p33": 200, "p66": 1000}
    assert interpret_prediction(50, q).startswith("Low")
    assert interpret_prediction(500, q).startswith("Moderate")
    assert interpret_prediction(5000, q).startswith("High")
