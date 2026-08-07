"""Tests for dataset loading, target detection and target preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.data_utils import (
    analyze_and_prepare_target,
    clean_numeric_string,
    clean_target_column,
    detect_target_column,
    load_dataset,
)


class TestLoadDataset:
    def test_loads_csv_by_path(self, classification_df, tmp_path):
        path = tmp_path / "data.csv"
        classification_df.to_csv(path, index=False)

        loaded = load_dataset(str(path))
        assert list(loaded.columns) == list(classification_df.columns)
        assert len(loaded) == len(classification_df)

    def test_loads_excel_by_path(self, classification_df, tmp_path):
        path = tmp_path / "data.xlsx"
        classification_df.to_excel(path, index=False)

        loaded = load_dataset(str(path))
        assert len(loaded) == len(classification_df)

    def test_rejects_unsupported_extension(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2\n")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(str(path))

    def test_rejects_invalid_input_type(self):
        with pytest.raises(ValueError, match="Invalid input type"):
            load_dataset(42)

    def test_accepts_file_like_object(self, classification_df, tmp_path):
        """Streamlit hands over an UploadedFile exposing a .name attribute."""
        path = tmp_path / "upload.csv"
        classification_df.to_csv(path, index=False)

        with open(path, "rb") as handle:
            loaded = load_dataset(handle)  # has .name
        assert len(loaded) == len(classification_df)


class TestDetectTargetColumn:
    def test_prefers_low_cardinality_trailing_column(self, classification_df):
        assert detect_target_column(classification_df) == "target"

    def test_finds_string_label_column(self):
        """Regression test: pandas 3 types text columns as `str`, and a dtype
        allowlist of [object, int, float, bool] silently skipped them."""
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0] * 5,
                "f2": [5.0, 6.0, 7.0, 8.0] * 5,
                "label": ["yes", "no"] * 10,
            }
        )
        assert detect_target_column(df) == "label"

    @pytest.mark.parametrize("name", ["target", "label", "class", "outcome", "y"])
    def test_prefers_conventionally_named_column(self, name):
        df = pd.DataFrame({"a": range(20), name: [0, 1] * 10, "b": range(20)})
        assert detect_target_column(df) == name

    def test_skips_identifier_column(self):
        df = pd.DataFrame(
            {"id": range(20), "f": [1.0, 2.0] * 10, "class": ["p", "q"] * 10}
        )
        assert detect_target_column(df) == "class"

    def test_skips_constant_column(self):
        df = pd.DataFrame(
            {"f": [1.0, 2.0] * 10, "label": ["a", "b"] * 10, "constant": [7] * 20}
        )
        assert detect_target_column(df) == "label"

    def test_falls_back_to_last_column(self):
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        assert detect_target_column(df) == "b"

    def test_always_returns_a_real_column(self, classification_df):
        assert detect_target_column(classification_df) in classification_df.columns

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="empty dataset"):
            detect_target_column(pd.DataFrame())

    def test_does_not_mutate_input(self, classification_df):
        before = classification_df.copy()
        detect_target_column(classification_df)
        pd.testing.assert_frame_equal(classification_df, before)


class TestCleanNumericString:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$1,000", 1000.0),
            ("50%", 50.0),
            ("₹2,500", 2500.0),
            ("€1.5", 1.5),
            ("£99", 99.0),
            ("  42  ", 42.0),
            ("-3.5", -3.5),
            (3.5, 3.5),
        ],
    )
    def test_parses_decorated_numbers(self, raw, expected):
        assert clean_numeric_string(raw) == pytest.approx(expected)

    def test_nan_passes_through(self):
        assert pd.isna(clean_numeric_string(np.nan))

    @pytest.mark.parametrize("raw", ["not-a-number", "1.0.0", "12abc", ""])
    def test_unparseable_becomes_nan(self, raw):
        assert pd.isna(clean_numeric_string(raw))


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

    def test_one_malformed_value_does_not_recategorise_the_column(self):
        """Regression: a whole-column cast raised on a single bad cell, and the
        fallback encoded an otherwise-numeric target as categorical."""
        values, encoder = clean_target_column(pd.Series(["1.0.0", "2.5", "3.5"]))

        assert encoder is None, "numeric target was misread as categorical"
        assert pd.isna(values[0])
        assert values[1] == pytest.approx(2.5)


class TestAnalyzeAndPrepareTarget:
    def test_splits_features_and_target(self, classification_df):
        X, y, encoder = analyze_and_prepare_target(classification_df.copy(), "target")

        assert "target" not in X.columns
        assert len(X) == len(classification_df)
        assert len(y) == len(classification_df)
        assert encoder is None  # numeric target needs no encoding

    def test_encodes_string_target(self):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0], "label": ["a", "b", "a", "b"]})
        _, y, encoder = analyze_and_prepare_target(df, "label")

        assert encoder is not None
        assert set(np.unique(y)) == {0, 1}
        assert set(encoder.inverse_transform([0, 1])) == {"a", "b"}

    def test_does_not_mutate_caller_frame(self, classification_df):
        before = classification_df.copy()
        analyze_and_prepare_target(classification_df.copy(), "target")
        pd.testing.assert_frame_equal(classification_df, before)
