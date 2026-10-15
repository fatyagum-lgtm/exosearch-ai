import json

import numpy as np
import pandas as pd
import pytest

from exosearch_ai.experiment import _scores, group_splits, infer_saved, train_experiment
from exosearch_ai.schema import FEATURES, labeled_and_candidates


def test_group_disjoint_split(catalog):
    _, frame = catalog
    labeled, _ = labeled_and_candidates(frame)
    splits = group_splits(labeled)
    stars = [set(labeled.iloc[idx]["tid"]) for idx in splits.values()]
    assert not (stars[0] & stars[1] or stars[0] & stars[2] or stars[1] & stars[2])
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(labeled)))


def test_reproducible_split(catalog):
    _, frame = catalog
    labeled, _ = labeled_and_candidates(frame)
    a, b = group_splits(labeled), group_splits(labeled)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_split_rejects_too_few_stars(catalog):
    _, frame = catalog
    labeled, _ = labeled_and_candidates(frame)
    with pytest.raises(ValueError, match="Not enough unique"):
        group_splits(labeled.iloc[:7])


def test_metric_calculation():
    result = _scores(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["f1"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["average_precision"] == 1.0


def test_train_artifacts_and_no_candidate_leakage(catalog, tmp_path):
    path, frame = catalog
    output = tmp_path / "models"
    exp = train_experiment(frame, path, output, is_synthetic=True)
    assert exp["data_type"] == "SYNTHETIC_DEMO_NOT_SCIENTIFIC"
    assert exp["winner"] in {"logistic_regression", "xgboost"}
    assert all(0 <= v["test"]["f1"] <= 1 for v in exp["models"].values())
    files = ("model.joblib", "experiment.json", "split_train.csv", "split_validation.csv",
             "split_test.csv", "test_predictions.csv", "candidate_ranking.csv")
    assert all((output / name).is_file() for name in files)
    # Holdout identifiers must be completely disjoint from training and validation.
    ids = [set(pd.read_csv(output / f"split_{split}.csv").tid) for split in ("train", "validation", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    ranks = pd.read_csv(output / "candidate_ranking.csv")
    assert set(ranks.tfopwg_disp).issubset({"PC", "APC"})
    assert (ranks.planet_likeness_score.diff().dropna() <= 0).all()
    scores = infer_saved(output / "model.joblib", ranks)
    assert np.allclose(scores, ranks.planet_likeness_score)
    assert json.loads((output / "experiment.json").read_text())["input_sha256"]


def test_tiny_dataset_rejected(catalog, tmp_path):
    path, frame = catalog
    with pytest.raises(ValueError, match="Too few labeled"):
        train_experiment(frame.iloc[:15], path, tmp_path)


def test_feature_whitelist_excludes_outcome_columns():
    forbidden = {"tfopwg_disp", "toi", "tid", "rowupdate", "toi_created", "pl_pnum"}
    assert not (forbidden & set(FEATURES))


def test_grouped_bootstrap_and_top_k():
    from exosearch_ai.experiment import grouped_bootstrap_intervals, precision_at_fraction
    y = np.array([0, 1] * 35)
    prob = np.where(y == 1, .8, .2).astype(float)
    groups = np.array([f"star-{i//2}" for i in range(len(y))])
    conf = grouped_bootstrap_intervals(y, prob, groups, repeats=75)
    assert conf["available"]
    assert conf["average_precision_95pct"] == [1.0, 1.0]
    rank = precision_at_fraction(y, prob, .2)
    assert rank["count"] == 14 and rank["top_positive_fraction"] == 1.0
