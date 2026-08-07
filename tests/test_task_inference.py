"""Tests for task-type inference and target-column heuristics.

Getting ``detect_task_type`` wrong silently switches the whole pipeline between
classifiers and regressors, so the boundary cases are pinned explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.task_inference import detect_task_type, is_potential_target


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


class TestIsPotentialTarget:
    def test_rejects_constant_column(self):
        assert not is_potential_target(pd.Series([1, 1, 1, 1]))

    def test_rejects_all_unique_id_column(self):
        assert not is_potential_target(pd.Series(range(50)))

    def test_rejects_mostly_missing_column(self):
        assert not is_potential_target(pd.Series([1.0] + [np.nan] * 9))

    def test_accepts_reasonable_label_column(self):
        assert is_potential_target(pd.Series([0, 1] * 25))
