# utils/feature_engineer.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer

# Import set_config from sklearn to enforce DataFrame output
from sklearn import set_config
# Ensure all transformers output pandas DataFrames where possible
set_config(transform_output="pandas")


class _ColumnPreservingMixin:
    """Shared ``get_feature_names_out`` for transformers that keep the schema.

    These transformers rewrite *values* but never add or drop columns, so their
    output names equal their input names. scikit-learn's pandas-output wrapper
    calls ``get_feature_names_out()`` with **no arguments**, so the column names
    seen during ``fit`` must be recorded rather than demanded from the caller.
    """

    def _record_input_features(self, X: pd.DataFrame) -> None:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        raise ValueError(
            f"{type(self).__name__} must be fitted before calling "
            "get_feature_names_out(), or input_features must be provided."
        )


class FeatureTypeCleaner(_ColumnPreservingMixin, BaseEstimator, TransformerMixin):
    """
    Cleans features by converting currency/percentage strings to numeric.
    Handles 'object' type columns that contain numeric-like strings.
    """

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._record_input_features(X)
        return self

    def transform(self, X):
        # Ensure X is a DataFrame; if it's a NumPy array, convert it (though set_config should handle this)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X) # This conversion might lose column names if X was from a previous array output

        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            # Check if the column contains any currency or percentage symbols
            if X[col].astype(str).str.contains(r'[\$%₹,]').any(): # Added comma for thousands separator
                # Remove symbols and convert to numeric, coercing errors to NaN
                X[col] = X[col].astype(str).replace(r'[\$,%₹]', '', regex=True) # Remove symbols
                X[col] = pd.to_numeric(X[col], errors='coerce') # Convert to numeric
        return X


class OutlierRemover(_ColumnPreservingMixin, BaseEstimator, TransformerMixin):
    """
    Handles outliers using IsolationForest from numeric columns by capping them
    with the column's median, instead of removing rows.
    Contamination is set to 0.01 (1% of data are outliers).
    """
    def __init__(self, contamination=0.01, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.detectors = {} # Stores fitted IsolationForest models for each numeric column
        self.medians = {} # Stores median values for each numeric column

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._record_input_features(X)
        self.detectors, self.medians = {}, {}

        for col in X.select_dtypes(include=np.number).columns:
            # Fit IsolationForest only on the current numeric column
            # Reshape to 2D array as IsolationForest expects
            self.detectors[col] = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            ).fit(X[[col]])
            # Store the median of the column for capping
            self.medians[col] = X[col].median()
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()

        # Apply outlier detection and capping/imputation
        for col, detector in self.detectors.items():
            if col not in X_transformed.columns:
                continue
            # Predict outliers (1 for inlier, -1 for outlier)
            outlier_mask = detector.predict(X_transformed[[col]]) == -1
            if not outlier_mask.any():
                continue

            replacement = self.medians.get(col, X_transformed[col].median())
            # The median of an integer column can be fractional; widen to float
            # rather than letting pandas raise on the narrowing assignment.
            if pd.api.types.is_integer_dtype(X_transformed[col]) and replacement != int(
                replacement
            ):
                X_transformed[col] = X_transformed[col].astype("float64")
            # Replace outliers with the column median, keeping the row count fixed.
            X_transformed.loc[outlier_mask, col] = replacement
        return X_transformed


class RareCategoryGrouper(_ColumnPreservingMixin, BaseEstimator, TransformerMixin):
    """
    Groups rare categories in object/category/bool columns into an 'Other' category.
    Rare categories are those with a frequency below a specified threshold.
    """
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.mappings = {} # Stores rare labels for each categorical column

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._record_input_features(X)
        self.mappings = {}

        for col in X.select_dtypes(include=['object', 'category', 'bool']).columns:
            # Calculate value frequencies
            freq = X[col].value_counts(normalize=True)
            # Identify rare labels (frequency below threshold)
            rare_labels = set(freq[freq < self.threshold].index)
            self.mappings[col] = rare_labels # Store rare labels for transformation
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        for col, rares in self.mappings.items():
            # Replace rare labels with 'Other'
            X_transformed[col] = X_transformed[col].apply(lambda x: 'Other' if x in rares else x)
        return X_transformed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on variance threshold and removes highly correlated features.
    """
    def __init__(self, corr_threshold=0.9, var_threshold=0.0):
        self.corr_threshold = corr_threshold
        self.var_threshold = var_threshold
        self.selected_columns = [] # Stores the names of selected columns

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Step 1: Remove low variance features
        # Create a temporary selector to find columns with variance above threshold
        temp_selector = VarianceThreshold(self.var_threshold)
        # Fit on numeric columns only
        numeric_cols = X.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            temp_selector.fit(X[numeric_cols])
            # Get names of columns retained after variance thresholding
            retained_by_variance = X[numeric_cols].columns[temp_selector.get_support()]
        else:
            retained_by_variance = pd.Index([]) # No numeric columns

        # Step 2: Remove highly correlated features from the variance-retained numeric columns
        to_drop_corr = []
        if not retained_by_variance.empty:
            corr_matrix = X[retained_by_variance].corr().abs()
            # Select upper triangle of correlation matrix to avoid duplicates
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find columns with correlation greater than threshold
            to_drop_corr = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]

        # Combine selected numeric and original non-numeric columns
        # All original columns that are not numeric are kept as is,
        # then numeric columns after variance and correlation selection are added.
        initial_non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        final_numeric_cols = [col for col in retained_by_variance if col not in to_drop_corr]

        self.selected_columns = initial_non_numeric_cols + final_numeric_cols
        self._fitted = True
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Return only the selected columns
        # Use .copy() to ensure it's a new DataFrame and avoid SettingWithCopyWarning
        return X[self.selected_columns].copy()

    def get_feature_names_out(self, input_features=None):
        # This transformer selects a subset of columns during fit, so the
        # selection recorded at fit time is authoritative.
        if not hasattr(self, "_fitted"):
            raise ValueError(
                "FeatureSelector must be fitted before calling get_feature_names_out()."
            )
        return np.asarray(self.selected_columns, dtype=object)


class SkewnessCorrector(_ColumnPreservingMixin, BaseEstimator, TransformerMixin):
    """
    Applies PowerTransformer (Yeo-Johnson) to highly skewed numeric columns.
    """
    def __init__(self):
        self.transformers = {}

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._record_input_features(X)
        self.transformers = {}

        for col in X.select_dtypes(include=np.number).columns:
            # Check for skewness (absolute skewness > 1 is a common heuristic)
            skew = X[col].skew()
            if pd.notna(skew) and abs(skew) > 1:
                transformer = PowerTransformer(method='yeo-johnson')
                # Fit the transformer on the column (reshaped to 2D)
                transformer.fit(X[[col]])
                self.transformers[col] = transformer
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        for col, transformer in self.transformers.items():
            # Apply the fitted transformer to the column
            X_transformed[col] = transformer.transform(X_transformed[[col]])
        return X_transformed


class PreprocessorBuilder:
    """
    Builds a comprehensive preprocessing pipeline using custom and scikit-learn transformers.
    """
    def __init__(self):
        self.pipeline = None # Stores the complete preprocessing pipeline

    def build_pipeline(self, X_raw_for_inference: pd.DataFrame):
        """
        Builds the scikit-learn pipeline based on the input data's column types.
        This method is called during fit to define the pipeline structure.

        Args:
            X_raw_for_inference (pd.DataFrame): A DataFrame used to infer column types
                                                for setting up the ColumnTransformer.
                                                This is typically the raw feature DataFrame.
        Returns:
            Pipeline: The scikit-learn Pipeline object.
        """
        # Step 1: Infer column types AFTER FeatureTypeCleaner and RareCategoryGrouper
        # Create a temporary pipeline for type inference for ColumnTransformer
        # This temporary pipeline ensures that num_cols and cat_cols are based
        # on the data types after the initial cleaning and grouping steps.
        initial_processing_temp_pipeline = Pipeline([
            ("cleaner_temp", FeatureTypeCleaner()),
            ("rare_grouper_temp", RareCategoryGrouper()),
            # OutlierRemover is part of the main pipeline, but for initial column type inference,
            # we consider the columns as they are after cleaner and rare_grouper.
        ])
        # Apply fit_transform to a copy of the raw data to get column types for ColumnTransformer setup
        X_temp_processed = initial_processing_temp_pipeline.fit_transform(X_raw_for_inference.copy())

        # Determine numeric and categorical columns based on the temporary processed data
        num_cols = X_temp_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X_temp_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Define pipelines for numeric and categorical features
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("skew_correct", SkewnessCorrector()), # Apply skewness correction
            ("scaler", StandardScaler()) # Scale numeric features
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")), # Impute missing categorical values
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # One-hot encode
        ])

        # Create a ColumnTransformer to apply different transformations to different column types
        preprocessor_transformer = ColumnTransformer(
            [
                ("num", num_pipeline, num_cols), # Apply numeric pipeline to numeric columns
                ("cat", cat_pipeline, cat_cols)  # Apply categorical pipeline to categorical columns
            ],
            remainder='passthrough' # Keep other columns (e.g., IDs) as they are
        )

        # Define the complete preprocessing pipeline
        self.pipeline = Pipeline([
            ("cleaner", FeatureTypeCleaner()),         # Custom cleaning (currency, etc.)
            ("rare_grouper", RareCategoryGrouper()),   # Group rare categories
            ("outlier", OutlierRemover()),             # Handle outliers (now by capping)
            ("selector", FeatureSelector()),           # Feature selection (variance, correlation)
            ("column_transform", preprocessor_transformer) # Apply ColumnTransformer (impute, scale, encode)
        ])

        return self.pipeline

    def fit_transform(self, X: pd.DataFrame):
        """
        Fits the preprocessing pipeline to X and transforms X.

        Args:
            X (pd.DataFrame): The input features DataFrame.

        Returns:
            Tuple[pd.DataFrame, Pipeline]: A tuple containing:
                - X_processed (pd.DataFrame): The transformed feature DataFrame.
                - pipeline (Pipeline): The fitted preprocessing pipeline object.
        """
        # Build the pipeline, passing X for type inference for ColumnTransformer
        pipeline = self.build_pipeline(X.copy()) # Pass a copy to avoid modifying original X

        # Fit and transform the data using the built pipeline
        # With set_config(transform_output="pandas"), this should return a DataFrame.
        X_processed = pipeline.fit_transform(X)

        return X_processed, pipeline


def build_preprocessor(X: pd.DataFrame) -> Pipeline:
    """Build an **unfitted** preprocessing pipeline shaped for ``X``.

    Only the *structure* of the pipeline is derived from ``X`` (which columns
    are numeric vs. categorical). No statistics are learned here, so callers are
    responsible for calling ``fit`` on the training fold alone. Keeping build
    and fit separate is what allows the trainer to split before fitting and
    thereby avoid leaking the evaluation fold into the transformers.

    Args:
        X: A representative feature frame used for column-type inference.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline`.
    """
    return PreprocessorBuilder().build_pipeline(X)
