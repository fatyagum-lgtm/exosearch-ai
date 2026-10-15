import pytest

from exosearch_ai.data import load_catalog
from exosearch_ai.synthetic import generate_synthetic


@pytest.fixture(scope="session")
def catalog(tmp_path_factory):
    path = tmp_path_factory.mktemp("toidata") / "simulated.csv"
    generate_synthetic(path, 140, seed=12)
    return path, load_catalog(path)
