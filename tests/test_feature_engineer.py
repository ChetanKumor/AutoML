"""Tests for the preprocessing pipeline.

The most important test here is :func:`test_preprocessor_is_not_prefitted` plus
:func:`test_statistics_come_from_training_fold_only`, which together pin the
no-leakage guarantee that the training flow depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from utils.feature_engineer import (
    FeatureSelector,
    FeatureTypeCleaner,
    OutlierRemover,
    RareCategoryGrouper,
    SkewnessCorrector,
    build_preprocessor,
)


# --------------------------------------------------------------------------
# Leakage guarantees
# --------------------------------------------------------------------------
def test_preprocessor_is_not_prefitted(classification_df):
    """build_preprocessor must return an UNFITTED pipeline.

    If it came back fitted, the caller could not control which rows the
    transformers learn their statistics from.
    """
    X = classification_df.drop(columns=["target"])
    preprocessor = build_preprocessor(X)

    with pytest.raises(NotFittedError):
        preprocessor.transform(X)


def test_statistics_come_from_training_fold_only():
    """Scaler statistics must reflect the training fold, never the full data."""
    X = pd.DataFrame({"a": np.arange(100.0), "b": list("xy") * 50})
    train = X.iloc[:50]

    preprocessor = build_preprocessor(X).fit(train)
    scaler = (
        preprocessor.named_steps["column_transform"]
        .named_transformers_["num"]
        .named_steps["scaler"]
    )
    learned_mean = float(scaler.mean_[0])

    # Ground truth: what the upstream steps actually feed the scaler for `train`.
    upstream = Pipeline([(n, clone(s)) for n, s in preprocessor.steps[:-1]])
    expected_mean = float(upstream.fit_transform(train)["a"].mean())

    assert learned_mean == pytest.approx(expected_mean)
    # And it must be nowhere near the full-dataset mean of 49.5.
    assert abs(learned_mean - X["a"].mean()) > 1.0


def test_transform_does_not_change_row_count(classification_df):
    """Outlier handling caps values; it must never drop rows."""
    X = classification_df.drop(columns=["target"])
    preprocessor = build_preprocessor(X).fit(X)
    assert len(preprocessor.transform(X)) == len(X)


# --------------------------------------------------------------------------
# get_feature_names_out: regression test for the pandas-output wrapper
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "transformer",
    [FeatureTypeCleaner(), OutlierRemover(), RareCategoryGrouper(), SkewnessCorrector()],
)
def test_column_preserving_names_without_input_features(transformer):
    """Called with no arguments (as sklearn does), names come from fit."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["p", "q", "p", "q"]})
    transformer.fit(X)
    assert list(transformer.get_feature_names_out()) == ["a", "b"]


@pytest.mark.parametrize(
    "transformer",
    [FeatureTypeCleaner(), OutlierRemover(), RareCategoryGrouper(), SkewnessCorrector()],
)
def test_names_before_fit_raise_clearly(transformer):
    with pytest.raises(ValueError, match="must be fitted"):
        transformer.get_feature_names_out()


def test_feature_selector_reports_selected_columns():
    X = pd.DataFrame(
        {
            "keep": [1.0, 2.0, 3.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],  # zero variance -> dropped
            "cat": ["a", "b", "a", "b"],
        }
    )
    selector = FeatureSelector().fit(X)
    names = list(selector.get_feature_names_out())

    assert "constant" not in names
    assert "keep" in names and "cat" in names
    assert list(selector.transform(X).columns) == names


def test_feature_selector_drops_correlated_columns():
    base = np.arange(50.0)
    X = pd.DataFrame({"a": base, "b": base * 2.0 + 1.0})  # perfectly correlated
    names = list(FeatureSelector(corr_threshold=0.9).fit(X).get_feature_names_out())
    assert len(names) == 1


# --------------------------------------------------------------------------
# Individual transformer behaviour
# --------------------------------------------------------------------------
def test_currency_and_percent_strings_become_numeric():
    X = pd.DataFrame({"salary": ["$1,000", "$2,000"], "growth": ["10%", "20%"]})
    out = FeatureTypeCleaner().fit(X).transform(X)

    assert pd.api.types.is_numeric_dtype(out["salary"])
    assert out["salary"].tolist() == [1000.0, 2000.0]
    assert out["growth"].tolist() == [10.0, 20.0]


def test_rare_categories_grouped_into_other():
    X = pd.DataFrame({"c": ["common"] * 99 + ["rare"]})
    out = RareCategoryGrouper(threshold=0.05).fit(X).transform(X)

    assert "rare" not in out["c"].values
    assert "Other" in out["c"].values


def test_outlier_capping_widens_integer_column_for_fractional_median():
    """A fractional median must not raise when written to an int64 column.

    pandas 3 refuses the narrowing assignment, so the column is widened first.
    The detector is forced to flag a row to make the path deterministic.
    """
    X = pd.DataFrame({"v": [1, 2, 3, 4]})  # median 2.5, dtype int64
    remover = OutlierRemover().fit(X)

    class _FlagFirstRow:
        @staticmethod
        def predict(frame):
            return np.array([-1] + [1] * (len(frame) - 1))

    remover.detectors["v"] = _FlagFirstRow()

    out = remover.transform(X)
    assert len(out) == len(X)
    assert out["v"].iloc[0] == pytest.approx(2.5)


def test_outlier_capping_preserves_row_count():
    X = pd.DataFrame({"v": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 10_000.0] * 5})
    out = OutlierRemover(contamination=0.1).fit(X).transform(X)
    assert len(out) == len(X)


def test_pipeline_handles_missing_values(messy_df):
    X = messy_df.drop(columns=["label"])
    processed = build_preprocessor(X).fit(X).transform(X)

    assert len(processed) == len(X)
    assert not pd.DataFrame(processed).isnull().any().any()
