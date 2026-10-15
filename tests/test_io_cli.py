import pandas as pd
import pytest

from exosearch_ai.cli import main
from exosearch_ai.data import load_catalog, sha256


def test_csv_validation_rejects_missing_columns(tmp_path):
    file = tmp_path / "wrong.csv"
    pd.DataFrame({"foo": [1]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        load_catalog(file)


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "none.csv")


def test_sha256_is_stable(tmp_path):
    file = tmp_path / "a.dat"
    file.write_bytes(b"test")
    assert sha256(file) == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


def test_local_demo_cli_and_html(tmp_path):
    output = tmp_path / "demo"
    assert main(["demo", "--output", str(output), "--stars", "90"]) == 0
    page = output / "index.html"
    assert page.is_file()
    text = page.read_text(encoding="utf-8")
    assert "SYNTHETIC DEMONSTRATION" in text
    assert "Candidate prioritization" in text
    assert "<script src=\"plotly.min.js\"></script>" in text
    assert (output / "plotly.min.js").stat().st_size > 1_000_000
    assert main(["report", "--experiment-dir", str(output)]) == 0
    assert main(["rank", "--experiment-dir", str(output), "--top", "4"]) == 0


def test_cli_invalid_rank_count_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["rank", "--experiment-dir", str(tmp_path), "--top", "0"])
    assert exc.value.code == 2
