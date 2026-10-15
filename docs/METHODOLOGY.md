# ExoSearch AI — Experimental Methodology

## Research question

How effectively can ordinary catalog parameters distinguish *already confirmed or known* TOIs from historically flagged false positives, and which unresolved objects resemble the former? This does not answer whether an object is actually an exoplanet.

## Source data

Use NASA Exoplanet Archive's TAP `toi` table (see README); preserve a dated raw snapshot and SHA-256 manifest. NASA updates TOIs and their dispositions. **This report does not include a completed NASA-catalog experiment in the delivered workspace** because the network request was blocked; synthetic input is purely for software verification.

## Training labels

Binary outcome `1`: `CP` or `KP`. Binary outcome `0`: `FP` or `FA`. `PC` and `APC` are unlabeled and must not be mixed into training or evaluation. All other dispositions are excluded rather than guessed. CP/KP vs FP/FA is not a sample of independently confirmed truth at a single historical observation time.

## Feature set and leakage controls

Nine numeric catalog attributes: orbital period, transit duration, transit depth, inferred planet radius, TESS magnitude, stellar temperature, surface gravity, stellar radius, and distance. The model explicitly excludes `tfopwg_disp`, `toi`, `tid`, catalog update dates, and related identifier/status fields. Missing values are imputed using medians computed **only from the training subset**.

Potential issue: `pl_rade` and other parameters may be revised after follow-up; this creates retrospective information that might not have existed when a candidate was first cataloged. The current study therefore must not claim prospective performance.

## Split protocol

- First remove duplicate TOI rows, rejecting contradictory duplicate statuses.
- Use `GroupShuffleSplit`, group = `tid` (TESS Input Catalog host star ID).
- Hold out ~20% of host stars for final test (seed 2026).
- Split the remaining groups into ~75% training and ~25% validation (~60/20/20 overall).
- Explicitly verify star groups are disjoint across the three splits.
- Save each split as a CSV with TOI, TIC, disposition, and label.
- Additionally report a **separate** stratified random-split Logistic Regression reference; this can share host stars and is not used to choose the final model.

## Models

- Logistic Regression: training-median imputer, StandardScaler, balanced class weights, regularized logistic model.
- XGBoost: training-median imputer, 120 histogram-based boosting trees, max depth 3, learning rate 0.055, random state 2026.

The model with highest **validation average precision** is selected. The final holdout is never used for tuning. Other test metrics are reported for scientific transparency: F1, precision, recall, ROC AUC, average precision, Brier score, and confusion matrix (fixed threshold 0.5). Exploratory 95% confidence intervals for AP and F1 resample held-out stars 160 times; these intervals do not include label, selection, or distribution-shift uncertainty. A controlled Logistic Regression ablation excludes derived radius `pl_rade` and reports the same held-out performance for sensitivity analysis. Precision among the highest-scored 20% of labeled test objects is also reported (this is not an actual candidate-discovery rate). These probabilities are **not calibrated** and must not be interpreted as accurate planet-existence probabilities.

## Candidate ranking

Apply selected model to only `PC` and `APC` rows. Sort by prediction score; the score measures statistical resemblance to labeled catalog classes, not the probability of exoplanet confirmation. Missing fields remain imputed from training medians.

## Reproducibility

All splits use fixed seeds. Save raw NASA table hash and source manifest, versions, test results, model artifact, predicted test samples, ranked candidates, plots, and generated dashboard. If catalog version changes, regenerate all research artifacts and disclose that the cohort changed.

## Threats to validity

1. Disposition labels and positive/negative classes may be uncertain or later revised.
2. Confirmation is nonrandom (follow-up selection and instrument availability biases).
3. Retrospective features can reflect follow-up decisions (especially derived radii).
4. Catalog classification differs from analyzing raw TESS light curves.
5. Multiple TOIs can share TIC stars; random splits can leak star-level signal.
6. Base rates differ between labeled and truly unresolved candidates; model scores are not observational priors.
7. No temporal holdout can be claimed without historical parameter snapshots.
8. Synthetic tests prove code functionality, not astrophysical predictive performance.

## Next scientific iteration

Freeze at least two dated catalog snapshots; construct a temporally faithful feature table reflecting original observation dates, collect vetted adjudication histories, and evaluate prospective prioritization. Consider light-curve signals as a separate, properly sourced research branch.
