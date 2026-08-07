"""Model search: train a family of estimators and rank them on a held-out set.

The public entry point is :func:`train_models`, which takes **raw** features and
targets. It performs the train/test split *before* fitting any transformer, so
the preprocessing pipeline never observes the evaluation fold. This is the
critical guarantee that keeps reported metrics honest.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.constants import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from utils.feature_engineer import build_preprocessor
from utils.logging_utils import get_logger
from utils.model_artifact import CLASSIFICATION, REGRESSION

logger = get_logger("trainer")

warnings.filterwarnings("ignore")

#: Metric used to rank models, per task type.
PRIMARY_METRIC = {CLASSIFICATION: "Accuracy", REGRESSION: "R2 Score"}


def _lazy_boosters(task_type: str) -> dict[str, Any]:
    """Import optional gradient-boosting libraries, skipping any that are absent.

    xgboost/lightgbm/catboost are heavy, platform-sensitive dependencies. A
    missing one should degrade the model search gracefully rather than take the
    whole application down at import time.
    """
    models: dict[str, Any] = {}
    is_clf = task_type == CLASSIFICATION

    try:
        from xgboost import XGBClassifier, XGBRegressor

        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=DEFAULT_RANDOM_STATE)
            if is_clf
            else XGBRegressor(random_state=DEFAULT_RANDOM_STATE)
        )
    except ImportError:  # pragma: no cover - depends on environment
        logger.warning("xgboost not installed; skipping XGBoost.")

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        models["LightGBM"] = (
            LGBMClassifier(random_state=DEFAULT_RANDOM_STATE, verbose=-1)
            if is_clf
            else LGBMRegressor(random_state=DEFAULT_RANDOM_STATE, verbose=-1)
        )
    except ImportError:  # pragma: no cover
        logger.warning("lightgbm not installed; skipping LightGBM.")

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        # allow_writing_files=False stops CatBoost from littering the working
        # directory with a catboost_info/ telemetry folder on every fit.
        catboost_kwargs = {
            "verbose": 0,
            "random_state": DEFAULT_RANDOM_STATE,
            "allow_writing_files": False,
        }
        models["CatBoost"] = (
            CatBoostClassifier(**catboost_kwargs)
            if is_clf
            else CatBoostRegressor(**catboost_kwargs)
        )
    except ImportError:  # pragma: no cover
        logger.warning("catboost not installed; skipping CatBoost.")

    return models


def _base_models(task_type: str) -> dict[str, Any]:
    """Return the candidate estimators for ``task_type``."""
    if task_type == CLASSIFICATION:
        models: dict[str, Any] = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=DEFAULT_RANDOM_STATE
            ),
            "DecisionTree": DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE),
            "RandomForest": RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE),
            "KNN": KNeighborsClassifier(),
            "NaiveBayes": GaussianNB(),
            "SVM": SVC(probability=True, random_state=DEFAULT_RANDOM_STATE),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=DEFAULT_RANDOM_STATE
            ),
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(random_state=DEFAULT_RANDOM_STATE),
            "RandomForest": RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
            "GradientBoosting": GradientBoostingRegressor(
                random_state=DEFAULT_RANDOM_STATE
            ),
        }
    models.update(_lazy_boosters(task_type))
    return models


#: Hyperparameter grids keyed by model name. Keys are shared between task types
#: because the underlying estimators expose the same parameter names.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "LogisticRegression": {"C": [0.01, 0.1, 1], "solver": ["liblinear"]},
    "LinearRegression": {},
    "DecisionTree": {"max_depth": [None, 10], "min_samples_split": [2, 5]},
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "KNN": {"n_neighbors": [3, 5]},
    "NaiveBayes": {},
    "SVM": {"C": [0.1, 1], "kernel": ["linear", "rbf"]},
    "SVR": {"C": [0.1, 1], "kernel": ["linear", "rbf"]},
    "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
    "XGBoost": {"n_estimators": [50], "learning_rate": [0.01, 0.1]},
    "LightGBM": {"n_estimators": [50], "learning_rate": [0.01, 0.1]},
    "CatBoost": {"depth": [4, 6], "learning_rate": [0.01, 0.1]},
}


@dataclass
class TrainingResult:
    """Outcome of the model search.

    ``leaderboard`` is ordered best-first. ``best_pipeline`` is a full
    :class:`~sklearn.pipeline.Pipeline` (preprocessor + estimator) fitted on the
    training fold, so callers can predict directly from raw input.
    """

    leaderboard: pd.DataFrame
    best_model_name: str | None
    best_pipeline: Pipeline | None
    best_estimator: Any | None
    preprocessor: Any | None
    task_type: str = ""
    feature_names: list[str] = field(default_factory=list)
    best_metrics: dict[str, Any] = field(default_factory=dict)


def evaluate_model(model, X_test, y_test, task_type: str) -> dict[str, Any]:
    """Compute task-appropriate evaluation metrics on a held-out set."""
    y_pred = model.predict(X_test)

    if task_type == CLASSIFICATION:
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    if task_type == REGRESSION:
        mse = mean_squared_error(y_test, y_pred)
        return {
            "R2 Score": r2_score(y_test, y_pred),
            "Mean Squared Error": mse,
            "Root Mean Squared Error": float(np.sqrt(mse)),
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        }
    raise ValueError(f"Unsupported task type: {task_type}")


def _make_split(X: pd.DataFrame, y: pd.Series, task_type: str):
    """Split into train/test, stratifying when the class balance permits."""
    stratify = None
    if task_type == CLASSIFICATION:
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            stratify = y
            logger.info("Using stratified split (min class count: %d).", min_class_count)
        else:
            logger.warning(
                "Min class count is %d; falling back to a non-stratified split.",
                min_class_count,
            )
    return train_test_split(
        X,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=stratify,
    )


def _make_cv(y_train: pd.Series, task_type: str):
    """Build a cross-validation splitter appropriate for the training fold."""
    if task_type == CLASSIFICATION:
        min_class_count = int(y_train.value_counts().min())
        n_splits = max(2, min(3, min_class_count))
        if min_class_count >= n_splits:
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE
            )
        return KFold(n_splits=2, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    return KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
) -> TrainingResult:
    """Train and rank candidate models on raw features.

    The train/test split happens *before* the preprocessor is fitted, so
    imputation statistics, scaling parameters, outlier detectors and feature
    selection are all learned from the training fold alone. Metrics are then
    computed on a genuinely held-out set.

    Args:
        X: Raw feature frame (no target column).
        y: Target values, already cleaned/encoded.
        task_type: ``"classification"`` or ``"regression"``.

    Returns:
        A :class:`TrainingResult` with the leaderboard and the best fitted
        pipeline.

    Raises:
        ValueError: If ``task_type`` is unsupported.
    """
    if task_type not in (CLASSIFICATION, REGRESSION):
        raise ValueError(f"Unsupported task type: {task_type}")

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    logger.info("Task type: %s | dataset shape: %s", task_type.upper(), X.shape)

    # --- Split FIRST, so the preprocessor never sees the evaluation fold. ---
    X_train_raw, X_test_raw, y_train, y_test = _make_split(X, y, task_type)

    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = [str(c) for c in getattr(X_train, "columns", [])]
    logger.info(
        "Preprocessor fitted on the training fold only (train=%s, test=%s, features=%d).",
        X_train_raw.shape,
        X_test_raw.shape,
        len(feature_names) or X_train.shape[1],
    )

    cv = _make_cv(y_train, task_type)
    scoring = "accuracy" if task_type == CLASSIFICATION else "r2"
    primary_metric = PRIMARY_METRIC[task_type]

    results: dict[str, dict[str, Any]] = {}
    best_score = -np.inf
    best_model_name: str | None = None
    best_estimator: Any | None = None

    logger.info("Training candidate models with hyperparameter search...")
    for name, model in _base_models(task_type).items():
        params = PARAM_GRIDS.get(name, {})
        try:
            if params:
                search = GridSearchCV(
                    model, params, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
                )
                search.fit(X_train, y_train)
                trained = search.best_estimator_
            else:
                trained = model.fit(X_train, y_train)

            metrics = evaluate_model(trained, X_test, y_test, task_type)
            score = metrics[primary_metric]
            results[name] = {"Model": trained, "Details": metrics}

            if score > best_score:
                best_score, best_model_name, best_estimator = score, name, trained
            logger.info("  %s -> %s: %.4f", name, primary_metric, score)
        except Exception as exc:
            logger.error("  Skipping %s: %s", name, exc, exc_info=True)
            results[name] = {"Model": None, "Details": {"Error": str(exc)}}

    # --- Ensembles, built from the successfully trained base estimators. ---
    if task_type == CLASSIFICATION:
        ensembles = _build_classification_ensembles(results)
        for ens_name, ensemble in ensembles.items():
            try:
                ensemble.fit(X_train, y_train)
                metrics = evaluate_model(ensemble, X_test, y_test, task_type)
                score = metrics[primary_metric]
                results[ens_name] = {"Model": ensemble, "Details": metrics}
                if score > best_score:
                    best_score, best_model_name, best_estimator = (
                        score,
                        ens_name,
                        ensemble,
                    )
                logger.info("  %s -> %s: %.4f", ens_name, primary_metric, score)
            except Exception as exc:
                logger.error("  Skipping %s: %s", ens_name, exc, exc_info=True)
                results[ens_name] = {"Model": None, "Details": {"Error": str(exc)}}

    leaderboard = _build_leaderboard(results, primary_metric)

    best_pipeline = None
    if best_estimator is not None:
        best_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("model", best_estimator)]
        )
        logger.info(
            "Best model: %s (%s=%.4f)", best_model_name, primary_metric, best_score
        )
    else:
        logger.warning("No model trained successfully.")

    return TrainingResult(
        leaderboard=leaderboard,
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        best_estimator=best_estimator,
        preprocessor=preprocessor,
        task_type=task_type,
        feature_names=feature_names,
        best_metrics=results.get(best_model_name, {}).get("Details", {})
        if best_model_name
        else {},
    )


def _build_classification_ensembles(results: dict[str, dict]) -> dict[str, Any]:
    """Assemble voting/stacking ensembles from base models that trained cleanly."""
    ensembles: dict[str, Any] = {}

    def fitted(*names: str) -> list[tuple[str, Any]]:
        return [
            (n.lower(), results[n]["Model"])
            for n in names
            if results.get(n, {}).get("Model") is not None
        ]

    voting_members = fitted("RandomForest", "XGBoost", "LogisticRegression")
    if len(voting_members) >= 2:
        ensembles["VotingEnsemble"] = VotingClassifier(
            estimators=voting_members, voting="soft", n_jobs=-1
        )

    stacking_members = fitted("SVM", "LightGBM", "KNN")
    if len(stacking_members) >= 2:
        ensembles["StackingEnsemble"] = StackingClassifier(
            estimators=stacking_members,
            final_estimator=LogisticRegression(random_state=DEFAULT_RANDOM_STATE),
            n_jobs=-1,
        )

    return ensembles


def _build_leaderboard(results: dict[str, dict], primary_metric: str) -> pd.DataFrame:
    """Flatten per-model metrics into a leaderboard sorted best-first."""
    rows = []
    for name, payload in results.items():
        details = payload.get("Details", {})
        row: dict[str, Any] = {"Model": name}
        for key, value in details.items():
            if key == "Confusion Matrix":
                row[key] = str(value)
            elif isinstance(value, (int, float, np.floating)):
                row[key] = round(float(value), 4)
            else:
                row[key] = value
        row.setdefault(primary_metric, np.nan)
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and primary_metric in leaderboard.columns:
        # Higher is better for both Accuracy and R2; failures (NaN) sink last.
        leaderboard = leaderboard.sort_values(
            by=primary_metric, ascending=False, na_position="last"
        ).reset_index(drop=True)
    return leaderboard
