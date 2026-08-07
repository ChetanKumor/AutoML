import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Automatically selects the most likely target column by checking columns
    with fewer unique values and suitable data types.
    """
    for col in reversed(df.columns):
        if df[col].nunique() < 20 and df[col].dtype in [object, int, float, bool]:
            return col
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
