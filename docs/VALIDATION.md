# Validation record

## Environment

Performed in a Linux-based development container with Python 3.13 and installed pandas, scikit-learn, XGBoost, Plotly and pytest. GitHub CI will target Python 3.11/3.12 on Linux and macOS after publication, but those online jobs have **not** run yet.

## Completed local checks

- Full executable synthetic end-to-end experiment through both classifiers, held-out evaluation, ranking and offline HTML report.
- Local unit/integration test suite: **20 tests passed** in this runtime.
- SQL API endpoint and disposition schema checked against official NASA Exoplanet Archive documentation.
- Standalone offline HTML is generated and its structure validated by automated tests. Browser screenshot validation could not be completed in this runtime (headless Chromium stalled).

## NASA retrospective evaluation update — 2026-10-15

A user-supplied 9-page PDF export of a NASA TOI-style query was recovered into an 8,148-row CSV; all rows parsed into the required 12 fields. Both models trained on real-looking catalog values, the TIC-disjoint held-out evaluation executed, and a real-data dashboard and ranking were generated. PDF provenance and the inability to independently verify the snapshot against the live NASA host are recorded in [REAL_NASA_RESULTS.md](REAL_NASA_RESULTS.md). This is **not** prospective astronomical validation.

- Unit/integration suite was rerun: **20 passed**.

## Not completed

- Independent direct download/verification of the NASA dataset from its source and prospective external validation.
- Performance comparison with ExoMiner/ExoMiner++: requires comparable cohorts and input modalities.
- Temporal evaluation: requires historical feature snapshots.
- CI job success status: not available until the user publishes to GitHub and sees completed workflows.

## Key commands

```
PYTHONPATH=src python -m exosearch_ai.cli demo --output artifacts/demo --stars 350
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m exosearch_ai.cli --help
```

Synthetic classification statistics, if generated, must never be quoted as astronomy research results.
