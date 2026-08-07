"""Tests for dataset validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.validation import (
    MIN_ROWS,
    DatasetValidationError,
    validate_dataframe,
    validate_target,
    validate_upload,
)


class _Upload:
    """Stand-in for Streamlit's UploadedFile."""

    def __init__(self, name: str, size: int = 1024):
        self.name = name
        self.size = size


class TestValidateUpload:
    @pytest.mark.parametrize("name", ["a.csv", "a.xlsx", "a.xls", "A.CSV"])
    def test_accepts_supported_extensions(self, name):
        validate_upload(_Upload(name))

    @pytest.mark.parametrize("name", ["a.txt", "a.json", "a.parquet", "a"])
    def test_rejects_unsupported_extensions(self, name):
        with pytest.raises(DatasetValidationError, match="Unsupported file type"):
            validate_upload(_Upload(name))

    def test_rejects_oversized_upload(self, monkeypatch):
        import utils.validation as validation

        monkeypatch.setattr(validation, "MAX_FILE_SIZE_MB", 1)
        with pytest.raises(DatasetValidationError, match="exceeds the"):
            validate_upload(_Upload("big.csv", size=5 * 1024 * 1024))

    def test_accepts_path_within_limit(self, classification_df, tmp_path):
        path = tmp_path / "data.csv"
        classification_df.to_csv(path, index=False)
        validate_upload(str(path))

    def test_rejects_non_file_input(self):
        with pytest.raises(DatasetValidationError, match="Expected a file path"):
            validate_upload(123)


class TestValidateDataframe:
    def test_accepts_valid_dataset(self, classification_df):
        validate_dataframe(classification_df)

    def test_rejects_empty_dataset(self):
        with pytest.raises(DatasetValidationError, match="no rows"):
            validate_dataframe(pd.DataFrame({"a": [], "b": []}))

    def test_rejects_too_few_rows(self):
        tiny = pd.DataFrame({"a": range(MIN_ROWS - 1), "b": range(MIN_ROWS - 1)})
        with pytest.raises(DatasetValidationError, match="at least"):
            validate_dataframe(tiny)

    def test_rejects_single_column(self):
        single = pd.DataFrame({"only": range(MIN_ROWS + 5)})
        with pytest.raises(DatasetValidationError, match="at least 2"):
            validate_dataframe(single)

    def test_rejects_duplicate_column_names(self):
        df = pd.DataFrame(np.arange(40).reshape(20, 2), columns=["dup", "dup"])
        with pytest.raises(DatasetValidationError, match="Duplicate column names"):
            validate_dataframe(df)


class TestValidateTarget:
    def test_accepts_valid_target(self, classification_df):
        validate_target(classification_df, "target")

    def test_rejects_missing_column(self, classification_df):
        with pytest.raises(DatasetValidationError, match="is not in the dataset"):
            validate_target(classification_df, "nope")

    def test_rejects_constant_target(self, classification_df):
        df = classification_df.copy()
        df["target"] = 1
        with pytest.raises(DatasetValidationError, match="single distinct value"):
            validate_target(df, "target")

    def test_rejects_all_missing_target(self, classification_df):
        df = classification_df.copy()
        df["target"] = np.nan
        with pytest.raises(DatasetValidationError, match="entirely missing"):
            validate_target(df, "target")

    def test_error_lists_available_columns(self, classification_df):
        with pytest.raises(DatasetValidationError) as exc_info:
            validate_target(classification_df, "typo")
        assert "num_a" in str(exc_info.value)
