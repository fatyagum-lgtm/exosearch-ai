"""TESS TOI table schema and hard guardrails against outcome leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

# All predictor fields are numeric catalog values observable without outcome labels.
# Object IDs, statuses, and status updates MUST NOT become predictors.
FEATURES = (
    "pl_orbper",    # days
    "pl_trandurh",  # hours
    "pl_trandep",   # ppm
    "pl_rade",      # Earth radii; model-derived and may reflect follow-up biases
    "st_tmag",      # TESS apparent magnitude
    "st_teff",      # K
    "st_logg",      # log(g)
    "st_rad",       # Solar radii
    "st_dist",      # parsec
)
IDENTITY = ("toi", "tid", "tfopwg_disp")
REQUIRED = IDENTITY + FEATURES
POSITIVE = frozenset(("CP", "KP"))
NEGATIVE = frozenset(("FP", "FA"))
UNLABELED = frozenset(("PC", "APC"))


def normalize_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, normalize status, and deduplicate TOI IDs conservatively."""
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required NASA TOI columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Empty TOI dataset")
    df = frame.loc[:, list(REQUIRED)].copy()
    df["tfopwg_disp"] = df["tfopwg_disp"].astype("string").str.strip().str.upper()
    df["toi"] = df["toi"].astype("string").str.strip()
    df["tid"] = df["tid"].astype("string").str.strip()
    for name in FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.loc[df["toi"].notna() & df["tid"].notna() & (df["toi"] != "")].copy()
    # NASA TOI and TIC IDs are numeric; reject foreign or injected strings.
    if not df["toi"].str.fullmatch(r"[0-9]{1,8}\.[0-9]{1,3}").all():
        raise ValueError("TOI identifiers must be numeric NNNN.DD style")
    if not df["tid"].str.fullmatch(r"[0-9]{1,15}(?:\.0)?").all():
        raise ValueError("TIC star IDs must be numeric")
    # Duplicates with incompatible dispositions make provenance ambiguous: don't pick arbitrarily.
    conflicting = df.groupby("toi")["tfopwg_disp"].nunique(dropna=False)
    if (conflicting > 1).any():
        raise ValueError("Contradictory dispositions for the same TOI; resolve source records")
    return df.drop_duplicates(subset="toi", keep="first").reset_index(drop=True)


def labeled_and_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = df["tfopwg_disp"]
    train = df.loc[labels.isin(POSITIVE | NEGATIVE)].copy()
    train["label"] = train["tfopwg_disp"].isin(POSITIVE).astype(int)
    candidates = df.loc[labels.isin(UNLABELED)].copy()
    if train["label"].nunique() != 2:
        raise ValueError("Training requires both known planet (CP/KP) and false positive (FP/FA)")
    return train.reset_index(drop=True), candidates.reset_index(drop=True)


def catalog_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "unique_tic_ids": int(df["tid"].nunique()),
        "dispositions": {str(k): int(v) for k, v in df["tfopwg_disp"].value_counts().items()},
        "feature_null_percent": {
            k: round(float(df[k].isna().mean() * 100), 2) for k in FEATURES
        },
    }
