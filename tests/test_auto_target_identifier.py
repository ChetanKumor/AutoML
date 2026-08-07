"""Tests for task-type inference and target-column heuristics.

Getting ``detect_task_type`` wrong silently switches the whole pipeline between
classifiers and regressors, so the boundary cases are pinned explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.auto_target_identifier import (
    clean_numeric_string,
    clean_target_column,
    detect_task_type,
    is_potential_target,
)


class TestDetectTaskType:
    def test_binary_integers_are_classification(self):
        assert detect_task_type(pd.Series([0, 1, 0, 1, 1])) == "classification"

    def test_small_multiclass_is_classification(self):
        assert detect_task_type(pd.Series([0, 1, 2, 1, 0, 2])) == "classification"

    def test_string_labels_are_classification(self):
        assert detect_task_type(pd.Series(["a", "b", "a"])) == "classification"

    def test_boolean_is_classification(self):
        assert detect_task_type(pd.Series([True, False, True])) == "classification"

    def test_continuous_values_are_regression(self):
        assert detect_task_type(pd.Series([1.5, 2.7, 3.14, 9.9])) == "regression"

    def test_house_prices_are_regression(self):
        """Large-magnitude integers must not be mistaken for class labels."""
        prices = pd.Series([250000, 310000, 425000, 199000, 555000, 610000])
        assert detect_task_type(prices) == "regression"

    def test_many_distinct_integers_are_regression(self):
        assert detect_task_type(pd.Series(range(100))) == "regression"

    def test_ignores_nans(self):
        assert detect_task_type(pd.Series([0, 1, np.nan, 1])) == "classification"


class TestCleanNumericString:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$1,000", 1000.0),
            ("50%", 50.0),
            ("₹2,500", 2500.0),
            ("  42  ", 42.0),
            (3.5, 3.5),
        ],
    )
    def test_parses_decorated_numbers(self, raw, expected):
        assert clean_numeric_string(raw) == pytest.approx(expected)

    def test_nan_passes_through(self):
        assert pd.isna(clean_numeric_string(np.nan))

    def test_unparseable_becomes_nan(self):
        assert pd.isna(clean_numeric_string("not-a-number"))


class TestCleanTargetColumn:
    def test_numeric_target_returns_no_encoder(self):
        values, encoder = clean_target_column(pd.Series([1.0, 2.0, 3.0]))
        assert encoder is None
        assert values.tolist() == [1.0, 2.0, 3.0]

    def test_categorical_target_returns_encoder(self):
        values, encoder = clean_target_column(pd.Series(["yes", "no", "yes"]))
        assert encoder is not None
        assert set(np.unique(values)) == {0, 1}

    def test_currency_target_is_cleaned_to_numeric(self):
        values, encoder = clean_target_column(pd.Series(["$100", "$200"]))
        assert encoder is None
        assert values.tolist() == [100.0, 200.0]


class TestIsPotentialTarget:
    def test_rejects_constant_column(self):
        assert not is_potential_target(pd.Series([1, 1, 1, 1]))

    def test_rejects_all_unique_id_column(self):
        assert not is_potential_target(pd.Series(range(50)))

    def test_rejects_mostly_missing_column(self):
        assert not is_potential_target(pd.Series([1.0] + [np.nan] * 9))

    def test_accepts_reasonable_label_column(self):
        assert is_potential_target(pd.Series([0, 1] * 25))
