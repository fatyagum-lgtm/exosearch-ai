# ExoSearch AI — Retrospective NASA TOI catalog experiment (2026-10-15)

## Dataset and provenance

- **Source form supplied by user:** `цифры.pdf`, a 9-page text PDF containing a CSV-formatted table with NASA TOI column names from the requested TAP query. Independent access to the NASA endpoint was unavailable during this run, so the current snapshot was **not independently re-downloaded or authenticated**.
- **PDF SHA-256:** `e6b0a1386f608cb82bf35b31d85020b37a588ee002e99db522b6cb6275f4a07c`.
- **Reconstructed UTF-8 CSV SHA-256:** `33eefa58e7b60ee6af27d55f1679269ddd7a95ce1193e181bd4fa043fb7e2360`.
- Parsed 8,148 data rows, each with exactly 12 CSV fields; 12 names match the project schema. The CSV data was recovered from PDF text, *not* an original downloadable CSV response. No row-completeness assertion against a live NASA snapshot can be made.
- 7,826 unique host-star TIC identifiers; 2 rows have empty dispositions (excluded from labeled and candidate groups).
- TFOPWG status counts: CP=814, KP=606, FP=1,314, FA=100, PC=4,824, APC=488, blank=2.
- 2,834 labeled records (CP/KP=positive, FP/FA=negative), 5,312 unresolved candidates (PC/APC; **never labeled for training**).
- Input to experimental run: `exosearch_nasa_toi_recovered.csv` (original converted CSV, hash matches `experiment.json`).

## Evaluation protocol

- Split by TIC host-star ID (GroupShuffleSplit), training **1702** / validation **572** / final held-out test **560** rows.
- Training-only imputation; model selected by **validation average precision**, then evaluated once on the star-disjoint held-out test data.
- Both Logistic Regression and XGBoost compared; fixed classification threshold of 0.5, not optimized on test.

| Model | Precision | Recall | F1 | Average precision | ROC-AUC | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.629 | 0.731 | 0.676 | 0.638 | 0.740 | 0.205 |
| XGBoost | 0.751 | 0.881 | 0.811 | 0.851 | 0.889 | 0.137 |

Winner selected on validation: **xgboost** (validation AP 0.910).

XGBoost hold-out confusion matrix (rows: actual negative, positive; columns: predicted negative, positive): `[[224, 76], [31, 229]]`.

Grouped star-bootstrap 95% interval (160 resamples) for XGBoost average precision: **0.802–0.894**. This interval does not capture label-selection, retrospective, or distribution-shift uncertainty.

- Logistic Regression ablation without estimated planet radius: test F1=0.679, AP=0.647.
- Exploratory random-split reference: 38 TIC IDs shared between train and test. This uses a different split and does not support a direct causal comparison.
- Candidate ranking produced for 5,312 PC/APC rows; scores are **statistical resemblances to labeled dispositions, not verified planetary probabilities or discoveries**.

![Actual held-out test measurements](screenshots/nasa-toi-real-holdout.png)

## Exact local validation

- `PYTHONPATH=src python -m exosearch_ai.cli inspect --input /mnt/data/exosearch_nasa_toi_recovered.csv` — successful.
- `PYTHONPATH=src python -m exosearch_ai.cli train --input /mnt/data/exosearch_nasa_toi_recovered.csv --output artifacts/real` — successful; `index.html`, model, experiment JSON, ranked candidates, split manifests and raw predictions generated.
- `PYTHONPATH=src python -m pytest -q` — **20 tests passed** in this environment.
- The generated HTML was **not independently visually verified in a browser during this run**; it is bundled with the offline Plotly library and originates from actual experiment files.

## Scientific limits (mandatory disclosure)

This is a **retrospective catalog-disposition prediction task**, not transit detection from light curves and not exoplanet confirmation. NASA catalog measurements may have been updated after scientific follow-up, potentially encoding outcome information. Target verification is strongly selected and non-random, unresolved candidates may differ from the labeled cohort, and this experiment uses neither a historical early-epoch snapshot nor a prospective test. Current scores must not be described as expected accuracy on future exoplanet discoveries. The candidate rankings are useful for a reproducible teaching/research prototype only.
