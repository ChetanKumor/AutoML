import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.auto_target_identifier import is_potential_target


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


def clean_currency_symbols(column: pd.Series) -> pd.Series:
    """
    Detects and removes currency symbols, percentages, or non-numeric characters
    from numeric-looking columns.
    """
    return column.replace(r"[^0-9.-]", "", regex=True).astype(float)


def preprocess_target_column(y: pd.Series) -> tuple[np.ndarray, object | None]:
    """Clean and, when necessary, encode the target column.

    A numeric-looking target (including one decorated with currency or percent
    symbols) is converted to float and needs no encoder. Anything else is
    treated as categorical and label-encoded.

    Args:
        y: The raw target series.

    Returns:
        A ``(values, encoder)`` pair, where ``encoder`` is ``None`` for numeric
        targets and a fitted ``LabelEncoder`` otherwise.
    """
    try:
        return clean_currency_symbols(y).values, None
    except (ValueError, TypeError, AttributeError):
        # Not numeric-like: fall back to categorical encoding. The exception
        # types are named explicitly so that genuine bugs (KeyboardInterrupt,
        # MemoryError, typos raising NameError) are not silently swallowed.
        encoder = LabelEncoder()
        return encoder.fit_transform(y.astype(str)), encoder


def analyze_and_prepare_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, np.ndarray, object]:
    """
    Drops the target column from features, preprocesses it, and returns:
    - cleaned X (features)
    - cleaned y (target)
    - encoder used (or None)
    """
    y_raw = df[target_col]
    y, encoder = preprocess_target_column(y_raw)
    df = df.drop(columns=[target_col])
    return df, y, encoder
