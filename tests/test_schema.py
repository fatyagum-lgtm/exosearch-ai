import pandas as pd
import pytest

from exosearch_ai.schema import FEATURES, labeled_and_candidates, normalize_catalog


def test_dataset_schema_and_states(catalog):
    _, frame = catalog
    labeled, candidates = labeled_and_candidates(frame)
    assert labeled["tfopwg_disp"].isin(("CP", "KP", "FP", "FA")).all()
    assert candidates["tfopwg_disp"].isin(("PC", "APC")).all()
    assert set(labeled.label) == {0, 1}
    assert len(labeled) + len(candidates) == len(frame)


def test_missing_column_fails(catalog):
    _, frame = catalog
    with pytest.raises(ValueError, match="Missing required"):
        normalize_catalog(frame.drop(columns=[FEATURES[0]]))


def test_conflicting_duplicate_status_fails(catalog):
    _, frame = catalog
    altered = frame.iloc[[0]].copy()
    altered.loc[:, "tfopwg_disp"] = "KP" if frame.iloc[0].tfopwg_disp == "FP" else "FP"
    with pytest.raises(ValueError, match="Contradictory"):
        normalize_catalog(pd.concat([frame, altered], ignore_index=True))


def test_exact_duplicate_is_removed(catalog):
    _, frame = catalog
    one = frame.iloc[[0]].copy()
    out = normalize_catalog(pd.concat([frame, one], ignore_index=True))
    assert len(out) == len(frame)


def test_unknown_disposition_does_not_enter_training(catalog):
    _, frame = catalog
    row = frame.iloc[0].copy()
    row.loc["tfopwg_disp"] = "UNKNOWN"
    row.loc["toi"] = "999.88"
    out = normalize_catalog(pd.concat([frame, row.to_frame().T], ignore_index=True))
    trained, ranked = labeled_and_candidates(out)
    assert "999.88" not in set(trained.toi)
    assert "999.88" not in set(ranked.toi)


def test_non_numeric_input_becomes_missing(catalog):
    _, frame = catalog
    frame = frame.copy()
    frame[FEATURES[0]] = frame[FEATURES[0]].astype(object)
    frame.loc[0, FEATURES[0]] = "not available"
    norm = normalize_catalog(frame)
    assert pd.isna(norm.loc[0, FEATURES[0]])


def test_injected_toi_id_is_rejected(catalog):
    _, frame = catalog
    frame = frame.copy()
    frame["toi"] = frame["toi"].astype(object)
    frame.loc[0, "toi"] = "<script>alert(1)</script>"
    with pytest.raises(ValueError, match="TOI identifiers"):
        normalize_catalog(frame)
