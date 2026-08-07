"""Inference about the *shape* of a supervised learning problem.

Answers two questions about a candidate target column: is it usable as a target
at all (:func:`is_potential_target`), and does it imply classification or
regression (:func:`detect_task_type`)? Getting the second one wrong silently
swaps the entire model family, so its boundaries are pinned by tests.

Target *cleaning* lives in :mod:`utils.data_utils`; this module only inspects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A column missing more than this fraction of its values is a poor target.
MAX_TARGET_MISSING_RATIO = 0.3


def is_potential_target(series: pd.Series) -> bool:
    """Report whether ``series`` is plausible as a prediction target.

    Rejects constant columns (nothing to learn), mostly-missing columns, and
    all-unique columns (which are almost always identifiers).
    """
    distinct = series.nunique()
    return (
        distinct > 1
        and distinct != len(series)
        and series.isnull().mean() <= MAX_TARGET_MISSING_RATIO
    )


#: A numeric target with more distinct values than this is treated as continuous.
MAX_CLASSES = 15
#: Encoded class labels are normally small integers; values above this magnitude
#: look like measurements (prices, counts) rather than category codes.
MAX_LABEL_MAGNITUDE = 1000
#: ...unless the values repeat often enough to behave like labels regardless.
MAX_UNIQUE_RATIO = 0.05


def detect_task_type(y_processed: pd.Series) -> str:
    """Infer whether a prepared target implies classification or regression.

    Expects a series that has already been through
    :func:`utils.data_utils.clean_target_column`.

    A target is classification when it is non-numeric, or when it is numeric but
    behaves like a set of discrete codes: few distinct values, all integral, and
    either small in magnitude or repeating often enough to be labels. Everything
    else -- notably continuous quantities such as prices -- is regression.

    Args:
        y_processed: The cleaned target series.

    Returns:
        ``"classification"`` or ``"regression"``.
    """
    # Anything non-numeric (str, object, category, bool) is categorical, hence
    # classification. Testing numeric-ness rather than enumerating dtype names
    # is essential: pandas 3 gives string columns the `str` dtype, so an
    # ['object', 'category', 'bool'] allowlist silently misses text labels and
    # falls through to regression.
    if not pd.api.types.is_numeric_dtype(y_processed) or pd.api.types.is_bool_dtype(
        y_processed
    ):
        return "classification"

    values = y_processed.dropna()
    unique_values = values.unique()
    num_unique = len(unique_values)

    if num_unique == 0 or num_unique > MAX_CLASSES:
        return "regression"

    # Fractional values (1.5, 2.7, ...) are never class codes.
    if not np.all(np.mod(unique_values, 1) == 0):
        return "regression"

    # Small integers look like labels. Large ones only do so when they repeat
    # often, which distinguishes {100000, 200000} used as codes from a handful
    # of distinct house prices.
    unique_ratio = num_unique / len(values)
    if np.max(np.abs(unique_values)) <= MAX_LABEL_MAGNITUDE:
        return "classification"
    if unique_ratio <= MAX_UNIQUE_RATIO:
        return "classification"

    return "regression"
