"""Shared pytest fixtures.

Tests are kept fast and hermetic: synthetic datasets are small, file logging is
disabled, and all artifacts are written to pytest's ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RANDOM_STATE = 0


@pytest.fixture(autouse=True)
def _isolate_side_effects(tmp_path, monkeypatch):
    """Keep tests from touching the developer's real logs/ and model dirs."""
    monkeypatch.setenv("AUTOML_LOG_TO_FILE", "0")
    monkeypatch.setenv("AUTOML_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("AUTOML_MODEL_DIR", str(tmp_path / "models"))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(RANDOM_STATE)


@pytest.fixture
def classification_df(rng) -> pd.DataFrame:
    """A small, learnable binary classification dataset with mixed dtypes."""
    n = 120
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "num_a": signal,
            "num_b": rng.normal(size=n),
            "cat": rng.choice(["x", "y", "z"], size=n),
            "target": (signal > 0).astype(int),
        }
    )


@pytest.fixture
def regression_df(rng) -> pd.DataFrame:
    """A small, learnable regression dataset."""
    n = 120
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "num_a": x,
            "num_b": rng.normal(size=n),
            "cat": rng.choice(["p", "q"], size=n),
            "price": 3.0 * x + rng.normal(scale=0.1, size=n) + 100.0,
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Data exercising the cleaning paths: currency, percents, NaNs, rare levels."""
    return pd.DataFrame(
        {
            "salary": ["$1,000", "$2,000", "$3,000", "$4,000"],
            "growth": ["10%", "20%", "30%", "40%"],
            "grade": ["A", "A", "B", "RARE"],
            "with_nan": [1.0, np.nan, 3.0, 4.0],
            "label": ["yes", "no", "yes", "no"],
        }
    )
