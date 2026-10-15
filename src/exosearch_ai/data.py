"""NASA TAP source acquisition plus manifest (network not used during inference)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .schema import FEATURES, catalog_summary, normalize_catalog

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
NASA_COLUMNS = ["toi", "tid", "tfopwg_disp", *FEATURES]
NASA_SQL = "select " + ",".join(NASA_COLUMNS) + " from toi"
DATASET_PAGE = "https://exoplanetarchive.ipac.caltech.edu/docs/TESSMission.html"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_catalog(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TOI CSV not found: {path}")
    if path.stat().st_size > 150_000_000:
        raise ValueError("CSV exceeds 150 MB safety limit")
    return normalize_catalog(pd.read_csv(path, low_memory=False))


def fetch_toi(destination: str | Path, *, timeout: int = 45) -> dict:
    """User-triggered download; never silently fetch live catalog during tests/training."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(TAP_URL, params={"query": NASA_SQL, "format": "csv"}, timeout=timeout)
    response.raise_for_status()
    if len(response.content) > 150_000_000:
        raise ValueError("NASA response too large")
    if not response.text.lstrip().startswith("toi,tid,"):
        raise ValueError("TAP returned unexpected schema or non-CSV response")
    staged = target.with_suffix(target.suffix + ".partial")
    staged.write_bytes(response.content)
    try:
        df = load_catalog(staged)
        digest = sha256(staged)
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    manifest = {
        "source": DATASET_PAGE,
        "api_endpoint": TAP_URL,
        "adql": NASA_SQL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "file": target.name,
        "summary": catalog_summary(df),
        "dataset_type": "nasa-tess-toi",
    }
    (target.parent / f"{target.stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
