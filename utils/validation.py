"""Validation for user-supplied datasets.

Uploads arrive from an untrusted source and are handed straight to pandas and
then to scikit-learn. Rejecting a bad dataset here, with a message that says
what is wrong, is far better than surfacing an opaque failure from deep inside
the model search -- or spending several minutes training on a dataset that was
never going to work.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

#: A model cannot be trained and evaluated meaningfully below this many rows:
#: an 80/20 split of 10 rows leaves 2 test rows.
MIN_ROWS = 10
#: Features plus target.
MIN_COLUMNS = 2


class DatasetValidationError(ValueError):
    """Raised when an input dataset cannot be used for training."""


def _describe_size(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def validate_upload(file) -> None:
    """Validate an uploaded file before it is parsed.

    Args:
        file: A filesystem path, or a file-like object with ``name`` and
            ``size`` attributes (as Streamlit's ``UploadedFile`` provides).

    Raises:
        DatasetValidationError: If the extension is unsupported or the payload
            exceeds the configured size limit.
    """
    name = getattr(file, "name", file if isinstance(file, str) else None)
    if name is None:
        raise DatasetValidationError("Expected a file path or an uploaded file object.")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise DatasetValidationError(
            f"Unsupported file type {suffix or '(none)'!r}. "
            f"Supported types: {', '.join(ALLOWED_EXTENSIONS)}."
        )

    size = getattr(file, "size", None)
    if size is None and isinstance(file, str) and Path(file).exists():
        size = Path(file).stat().st_size

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size is not None and size > max_bytes:
        raise DatasetValidationError(
            f"File is {_describe_size(size)}, which exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit. Set AUTOML_MAX_UPLOAD_MB to raise it."
        )


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate a parsed dataset is usable for training.

    Args:
        df: The loaded dataset.

    Raises:
        DatasetValidationError: If the dataset is empty, too small, has too few
            columns, or has duplicate column names.
    """
    if df.empty:
        raise DatasetValidationError("The dataset contains no rows.")

    if len(df) < MIN_ROWS:
        raise DatasetValidationError(
            f"The dataset has {len(df)} rows; at least {MIN_ROWS} are needed to "
            "train and evaluate a model."
        )

    if df.shape[1] < MIN_COLUMNS:
        raise DatasetValidationError(
            f"The dataset has {df.shape[1]} column(s); at least {MIN_COLUMNS} "
            "are needed (one or more features plus a target)."
        )

    duplicates = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicates:
        raise DatasetValidationError(
            f"Duplicate column names are not supported: {', '.join(map(str, duplicates))}."
        )


def validate_target(df: pd.DataFrame, target_col: str) -> None:
    """Validate that ``target_col`` can actually be learned.

    Args:
        df: The loaded dataset.
        target_col: The chosen target column.

    Raises:
        DatasetValidationError: If the column is absent, entirely missing, or
            constant.
    """
    if target_col not in df.columns:
        raise DatasetValidationError(
            f"Target column {target_col!r} is not in the dataset. "
            f"Available columns: {', '.join(map(str, df.columns))}."
        )

    target = df[target_col]
    if target.isna().all():
        raise DatasetValidationError(f"Target column {target_col!r} is entirely missing.")

    if target.nunique(dropna=True) < 2:
        raise DatasetValidationError(
            f"Target column {target_col!r} has a single distinct value, so "
            "there is nothing to predict."
        )

    if df.shape[1] < 2:
        raise DatasetValidationError(
            "The dataset has no feature columns once the target is removed."
        )
