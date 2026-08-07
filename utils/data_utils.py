import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.task_inference import is_potential_target


def load_dataset(file) -> pd.DataFrame:
    """
    Loads a CSV or Excel file from Streamlit's UploadedFile object or file path.
    Supports both .csv and .xlsx/.xls.
    """
    if hasattr(file, "name"):  # UploadedFile from Streamlit
        file_name = file.name
    elif isinstance(file, str):
        file_name = file
    else:
        raise ValueError("Invalid input type for file. Must be a path or UploadedFile.")

    if file_name.endswith(".csv"):
        return pd.read_csv(file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel are supported.")


#: Column names that conventionally denote a prediction target.
TARGET_NAME_HINTS = frozenset(
    {"target", "label", "class", "y", "outcome", "output", "result"}
)
#: Above this many distinct values a column looks like an identifier or a
#: free-text field rather than a label.
MAX_TARGET_CARDINALITY = 20


def detect_target_column(df: pd.DataFrame) -> str:
    """Guess which column is the prediction target.

    Prefers a conventionally-named column, then the right-most plausible label
    column (datasets overwhelmingly put the target last), and finally falls back
    to the last column.

    Args:
        df: The loaded dataset.

    Returns:
        The name of the chosen column. Always a real column of ``df``.
    """
    if df.empty or df.shape[1] == 0:
        raise ValueError("Cannot detect a target column in an empty dataset.")

    # 1. An explicitly named target wins, if it is actually usable.
    for col in df.columns:
        if str(col).strip().lower() in TARGET_NAME_HINTS and is_potential_target(df[col]):
            return col

    # 2. Otherwise take the right-most low-cardinality, usable column. Testing
    #    is_potential_target rather than an allowlist of dtype objects matters:
    #    pandas 3 types text columns as `str`, so a check like
    #    `dtype in [object, int, float, bool]` skips every string label and
    #    silently selects the wrong column.
    for col in reversed(df.columns):
        if is_potential_target(df[col]) and df[col].nunique() <= MAX_TARGET_CARDINALITY:
            return col

    # 3. Nothing looked like a label; fall back to the conventional position.
    return df.columns[-1]


#: Symbols stripped from numeric-looking values before parsing.
_DECORATIONS = ("$", "%", "₹", "€", "£", ",")


def clean_numeric_string(val):
    """Parse a single decorated numeric value, or return NaN.

    Handles currency symbols, percent signs and thousands separators. Values
    that are not recognisably numeric become NaN rather than raising, so one
    malformed cell cannot change how the whole column is interpreted.

    Args:
        val: A scalar of any type.

    Returns:
        A float, the original value if it was already non-string, or NaN.
    """
    if pd.isnull(val):
        return np.nan
    if not isinstance(val, str):
        return val

    cleaned = val.strip()
    for symbol in _DECORATIONS:
        cleaned = cleaned.replace(symbol, "")

    # More than one decimal point is not a number (e.g. a version string).
    if cleaned.count(".") > 1:
        return np.nan

    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_target_column(series: pd.Series) -> tuple[np.ndarray, object | None]:
    """Clean and, when necessary, encode a target column.

    A numeric-looking target -- including one decorated with currency or percent
    symbols -- is converted to float and needs no encoder. Anything else is
    treated as categorical and label-encoded.

    Parsing is per-value and NaN-tolerant. A whole-column regex-and-cast would
    raise on a single malformed cell, and the resulting fallback would encode an
    otherwise-numeric target as categorical, silently turning a regression
    problem into a classification one.

    Args:
        series: The raw target series.

    Returns:
        A ``(values, encoder)`` pair, where ``encoder`` is ``None`` for numeric
        targets and a fitted ``LabelEncoder`` otherwise.
    """
    cleaned = series.apply(clean_numeric_string)

    if pd.api.types.is_numeric_dtype(cleaned) and not cleaned.isnull().all():
        return cleaned.values, None

    encoder = LabelEncoder()
    return encoder.fit_transform(series.astype(str).fillna("__MISSING__")), encoder


#: Backwards-compatible alias. Prefer :func:`clean_target_column`.
preprocess_target_column = clean_target_column


def analyze_and_prepare_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, np.ndarray, object | None]:
    """Split a dataset into features and a cleaned target.

    Args:
        df: The loaded dataset, including the target column.
        target_col: Name of the column to predict.

    Returns:
        A ``(X, y, encoder)`` triple: the feature frame with ``target_col``
        removed, the cleaned target values, and the fitted ``LabelEncoder`` if
        the target was categorical (otherwise ``None``).

    Raises:
        KeyError: If ``target_col`` is not a column of ``df``.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} is not in the dataset.")

    y, encoder = clean_target_column(df[target_col])
    return df.drop(columns=[target_col]), y, encoder
