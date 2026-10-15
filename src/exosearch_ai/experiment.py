"""Leakage-aware experiments and deterministic ranking for NASA TOI catalog data."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import sha256
from .schema import FEATURES, catalog_summary, labeled_and_candidates

SEED = 2026


def _models():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=SEED)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", XGBClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.055, subsample=0.85,
                colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
                n_jobs=2, random_state=SEED, tree_method="hist",
            )),
        ]),
    }


def group_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, np.ndarray]:
    """Disjoint TIC IDs across train/validation/test; deterministic for a CSV row ordering."""
    if df["tid"].nunique() < 15:
        raise ValueError("Not enough unique TIC star groups for group-disjoint evaluation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    remain, test = next(splitter.split(df, df["label"], groups=df["tid"]))
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    subtrain, subval = next(splitter2.split(df.iloc[remain], df.iloc[remain]["label"], groups=df.iloc[remain]["tid"]))
    train, val = remain[subtrain], remain[subval]
    splits = {"train": np.array(train), "validation": np.array(val), "test": np.array(test)}
    sets = [set(df.iloc[ids]["tid"]) for ids in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError("Group leakage: TIC IDs overlap")
    if any(df.iloc[ids]["label"].nunique() != 2 for ids in splits.values()):
        raise ValueError("At least one disjoint split lacks both classes; use a larger dataset")
    return splits


def _scores(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    predicted = (prob >= threshold).astype(int)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, predicted, labels=[0, 1]).ravel()]
    precision, recall, _ = precision_recall_curve(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    return {
        "n": int(len(y)), "positives": int(np.sum(y)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "pr_curve": {"recall": recall.tolist(), "precision": precision.tolist()},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def grouped_bootstrap_intervals(y: np.ndarray, prob: np.ndarray, star_ids: np.ndarray, *, repeats: int = 160) -> dict:
    """Exploratory uncertainty for test-set metrics, resampling independent host stars.

    These intervals only reflect finite held-out star sampling. They do not account for
    label noise, catalog revisions, selection bias, or distribution shift.
    """
    rng = np.random.default_rng(SEED + 19)
    stars = np.unique(star_ids.astype(str))
    members = {star: np.flatnonzero(star_ids.astype(str) == star) for star in stars}
    ap, f1s = [], []
    for _ in range(repeats):
        chosen = rng.choice(stars, size=len(stars), replace=True)
        ix = np.concatenate([members[s] for s in chosen])
        if len(np.unique(y[ix])) != 2:
            continue
        ap.append(average_precision_score(y[ix], prob[ix]))
        f1s.append(f1_score(y[ix], (prob[ix] >= 0.5).astype(int), zero_division=0))
    if len(ap) < 30:
        return {"available": False, "note": "Too few valid bootstrap resamples"}
    return {
        "available": True, "resampling_unit": "TIC host star", "valid_resamples": len(ap),
        "average_precision_95pct": list(map(float, np.quantile(ap, [.025, .975]))),
        "f1_95pct": list(map(float, np.quantile(f1s, [.025, .975]))),
        "note": "Exploratory grouped bootstrap only; does not address temporal or label-selection bias.",
    }


def precision_at_fraction(y: np.ndarray, prob: np.ndarray, fraction: float = 0.20) -> dict:
    """Positive disposition share among top ranked holdout objects (not candidate discovery rate)."""
    k = max(1, int(np.ceil(len(y) * fraction)))
    top = np.argsort(-prob, kind="stable")[:k]
    share = float(y[top].mean())
    prevalence = float(y.mean())
    return {"fraction": fraction, "count": k, "top_positive_fraction": share,
            "baseline_prevalence": prevalence,
            "lift_over_prevalence": float(share / prevalence) if prevalence else None}


def feature_importance(fitted: Pipeline) -> list[dict]:
    estimator = fitted.named_steps["classifier"]
    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        weights = np.abs(estimator.coef_[0])
    else:
        return []
    return sorted(
        ({"feature": key, "importance": float(value)} for key, value in zip(FEATURES, weights)),
        key=lambda r: -r["importance"],
    )


def _evaluate_random_baseline(df: pd.DataFrame) -> dict:
    """Comparison only; never tune final model using this split."""
    X = df[list(FEATURES)]
    y = df["label"].to_numpy(dtype=int)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=SEED, stratify=y
    )
    # Baseline random split intentionally permits shared TIC IDs; this is its limitation.
    model = _models()["logistic_regression"]
    model.fit(X.iloc[train_idx], y[train_idx])
    return {
        "metrics": _scores(y[test_idx], model.predict_proba(X.iloc[test_idx])[:, 1]),
        "shared_tic_ids": len(set(df.iloc[train_idx]["tid"]) & set(df.iloc[test_idx]["tid"])),
        "note": "Exploratory random-split reference; training and test may share TIC IDs.",
    }


def _write_research_summary(out: Path, experiment: dict) -> None:
    """Reproducible Markdown report generated strictly from measured experiment outputs."""
    label = "SYNTHETIC DEMO (NOT SCIENTIFIC VALIDATION)" if experiment["data_type"].startswith("SYNTHETIC") else "NASA TOI CATALOG EXPERIMENT"
    lines = [f"# ExoSearch AI — {label}", "", "## Experimental setup", "",
             f"- Input SHA-256: `{experiment['input_sha256']}`",
             f"- Labeled examples: {experiment['labeled_rows']}",
             f"- Held-out test rows: {experiment['split_counts']['test']['rows']}",
             f"- Split protocol: {experiment['split_protocol']}",
             f"- Model selection: {experiment['model_selection']}", "", "## Held-out scores", "",
             "| Model | Precision | Recall | F1 | Average precision | ROC AUC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, records in experiment["models"].items():
        m = records["test"]
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['average_precision']:.3f} | {m['roc_auc']:.3f} | {m['brier_score']:.3f} |")
    a=experiment['feature_ablation']['metrics']; rnd=experiment['random_split_reference']
    lines += ["", "## Ablation — remove derived radius", "",
              "Re-train Logistic Regression without `pl_rade` on the same star-disjoint training split; holdout classification:",
              f"- Test F1 = {a['f1']:.3f}; average precision = {a['average_precision']:.3f}.",
              "", "## Random split diagnostic", "",
              f"- Stratified random-split reference AP: {rnd['metrics']['average_precision']:.3f}.",
              f"- Host stars appearing in both train and test under random split: {rnd['shared_tic_ids']}.",
              "- This diagnostic is not a matched scientific comparison with the selected model and is not used for selection.",
              "", "## Limitations", "",
              "- Follow-up selections and retrospective parameter revisions can bias every model metric.",
              "- No true-time prospective validation is established.",
              "- Model scores do not confirm planets or represent calibrated discovery probabilities."]
    if experiment["data_type"].startswith("SYNTHETIC"):
        lines += ["", "**WARNING: All observations and reported metrics derive from a simulated dataset, not from NASA TESS.**"]
    (out / "research_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def train_experiment(frame: pd.DataFrame, input_csv: str | Path, output: str | Path, *, is_synthetic: bool = False) -> dict:
    labeled, candidates = labeled_and_candidates(frame)
    if len(labeled) < 75:
        raise ValueError("Too few labeled records (<75) for the experimental pipeline")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    splits = group_splits(labeled)
    X = labeled[list(FEATURES)]
    y = labeled["label"].to_numpy(dtype=int)
    for split, indices in splits.items():
        subset = labeled.iloc[indices][["toi", "tid", "tfopwg_disp", "label"]]
        subset.to_csv(out / f"split_{split}.csv", index=False)
    models = _models()
    results = {}
    trained = {}
    for name, pipeline in models.items():
        fitted = clone(pipeline).fit(X.iloc[splits["train"]], y[splits["train"]])
        validation_prob = fitted.predict_proba(X.iloc[splits["validation"]])[:, 1]
        results[name] = {"validation": _scores(y[splits["validation"]], validation_prob)}
        trained[name] = fitted
    # Selection is based only on validation AP, with lexical tie-breaking for determinism.
    best = sorted(models, key=lambda n: (-results[n]["validation"]["average_precision"], n))[0]
    fitted_best = trained[best]
    # Test samples are not used to tune models or decision threshold.
    for name, fitted in trained.items():
        test_probs = fitted.predict_proba(X.iloc[splits["test"]])[:, 1]
        results[name]["test"] = _scores(y[splits["test"]], test_probs)
        results[name]["importance"] = feature_importance(fitted)
        results[name]["test_star_bootstrap"] = grouped_bootstrap_intervals(
            y[splits["test"]], test_probs, labeled.iloc[splits["test"]]["tid"].to_numpy()
        )
        results[name]["precision_at_top_20pct"] = precision_at_fraction(
            y[splits["test"]], test_probs, 0.20
        )
    candidate_df = candidates[["toi", "tid", "tfopwg_disp", *FEATURES]].copy()
    if not candidate_df.empty:
        candidate_df["planet_likeness_score"] = fitted_best.predict_proba(candidate_df[list(FEATURES)])[:, 1]
        candidate_df = candidate_df.sort_values("planet_likeness_score", ascending=False)
    else:
        candidate_df["planet_likeness_score"] = pd.Series(dtype=float)
    candidate_df.to_csv(out / "candidate_ranking.csv", index=False)
    joblib.dump(fitted_best, out / "model.joblib")
    # Controlled ablation on the SAME splits; it is descriptive and does not select the winner.
    no_radius = [k for k in FEATURES if k != "pl_rade"]
    ablation_model = clone(models["logistic_regression"]).fit(
        X.iloc[splits["train"]][no_radius], y[splits["train"]]
    )
    ablated_prob = ablation_model.predict_proba(X.iloc[splits["test"]][no_radius])[:, 1]
    ablation = {
        "research_question": "Sensitivity of Logistic Regression to removal of derived planet radius",
        "removed_feature": "pl_rade", "remaining_features": no_radius,
        "protocol": "Same TIC-disjoint train/test split; training-only preprocessing; not used for model selection",
        "metrics": _scores(y[splits["test"]], ablated_prob),
    }
    random_ref = _evaluate_random_baseline(labeled)
    test_ids = splits["test"]
    holdout = labeled.iloc[test_ids][["toi", "tid", "tfopwg_disp", "label", *FEATURES]].copy()
    holdout["model_score"] = fitted_best.predict_proba(X.iloc[test_ids])[:, 1]
    holdout["predicted"] = (holdout["model_score"] >= 0.5).astype(int)
    holdout.to_csv(out / "test_predictions.csv", index=False)
    manifest = {
        "project": "ExoSearch AI",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": "SYNTHETIC_DEMO_NOT_SCIENTIFIC" if is_synthetic else "NASA_TOI_CATALOG_USER_SUPPLIED",
        "input_filename": str(input_csv),
        "input_sha256": sha256(Path(input_csv)),
        "schema_summary": catalog_summary(frame),
        "labeled_rows": len(labeled),
        "unlabeled_candidates": len(candidates),
        "disposition_mapping": {"CP": 1, "KP": 1, "FP": 0, "FA": 0, "PC": "rank_only", "APC": "rank_only"},
        "positive_label": "Known or confirmed disposition, not exoplanet existence proof",
        "features": list(FEATURES),
        "excluded_from_training": ["toi", "tid", "tfopwg_disp", "toi_created", "rowupdate", "pl_pnum"],
        "split_protocol": "TIC-disjoint train/validation/test = ~60/20/20 by groups",
        "seed": SEED,
        "split_counts": {k: {"rows": len(v), "stars": int(labeled.iloc[v]["tid"].nunique())} for k, v in splits.items()},
        "model_selection": "highest validation average precision, then lexical model name",
        "winner": best,
        "models": results,
        "random_split_reference": random_ref,
        "feature_ablation": ablation,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "scientific_limitations": [
            "This is catalog disposition prediction, not photometric transit discovery.",
            "Outcome categories can be revised and are biased by follow-up selection.",
            "Model scores are not calibrated astrophysical discovery probabilities.",
            "Features may be revised after follow-up; retrospective evaluation can be optimistic.",
            "True candidates (PC/APC) are never used as labeled positive or negative classes.",
            "Random split can share host stars and should not be used as final generalization evidence.",
        ],
    }
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_research_summary(out, manifest)
    return manifest


def infer_saved(model_path: str | Path, frame: pd.DataFrame) -> np.ndarray:
    """Load only models generated by your own trusted run, never arbitrary third-party files."""
    model = joblib.load(model_path)
    return model.predict_proba(frame[list(FEATURES)])[:, 1]
