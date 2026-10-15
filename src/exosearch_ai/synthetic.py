"""SIMULATED catalogs for smoke tests; never scientific evidence about NASA/TESS."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FEATURES


def generate_synthetic(path: str | Path, n_stars: int = 350, seed: int = 2026) -> pd.DataFrame:
    if n_stars < 60:
        raise ValueError("At least 60 artificial stars required for split validation")
    rng = np.random.default_rng(seed)
    rows = []
    for star in range(n_stars):
        star_teff = rng.normal(5300, 920)
        star_mag = rng.uniform(8, 16)
        count = 1 if rng.random() < 0.82 else 2
        for j in range(count):
            planet = rng.random() < 0.43
            r = float(np.exp(rng.normal(0.6 if planet else 2.5, 0.55)))
            depth = float(np.exp(rng.normal(5.7 if planet else 7.5, 0.85)))
            orbital = float(np.exp(rng.normal(1.9 if planet else 1.5, 1.1)))
            duration = float(np.clip(rng.normal(3.6 if planet else 7.2, 1.6), 0.2, 22))
            status = ("CP" if rng.random() < 0.7 else "KP") if planet else ("FP" if rng.random() < 0.85 else "FA")
            if rng.random() < 0.15:
                status = "PC" if rng.random() < 0.65 else "APC"
            record = {
                "toi": f"{1000 + star}.{j+1:02d}",
                "tid": str(800000000 + star),
                "tfopwg_disp": status,
                "pl_orbper": orbital,
                "pl_trandurh": duration,
                "pl_trandep": depth,
                "pl_rade": r,
                "st_tmag": star_mag,
                "st_teff": star_teff,
                "st_logg": rng.normal(4.3, 0.25),
                "st_rad": np.clip(rng.normal(1.1, 0.3), 0.3, 3),
                "st_dist": rng.uniform(25, 1100),
            }
            if rng.random() < 0.06:
                record[rng.choice(FEATURES)] = np.nan
            rows.append(record)
    df = pd.DataFrame(rows)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df
