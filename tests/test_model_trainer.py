"""Tests for the model search.

The search is expensive, so most tests restrict the candidate set to a couple of
cheap estimators via monkeypatching. Correctness of the *orchestration* -- the
split, the leak-free fit, ranking and failure handling -- is what matters here.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from utils import model_trainer
from utils.model_trainer import (
    PRIMARY_METRIC,
    evaluate_model,
    train_models,
)


@pytest.fixture
def cheap_classifiers(monkeypatch):
    """Restrict the search to two fast classifiers."""
    monkeypatch.setattr(
        model_trainer,
        "_base_models",
        lambda task_type: {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "Dummy": DummyClassifier(strategy="most_frequent"),
        },
    )
    monkeypatch.setattr(model_trainer, "PARAM_GRIDS", {})


@pytest.fixture
def cheap_regressors(monkeypatch):
    monkeypatch.setattr(
        model_trainer,
        "_base_models",
        lambda task_type: {
            "LinearRegression": LinearRegression(),
            "Dummy": DummyRegressor(),
        },
    )
    monkeypatch.setattr(model_trainer, "PARAM_GRIDS", {})


class TestTrainModelsClassification:
    def test_returns_ranked_leaderboard(self, classification_df, cheap_classifiers):
        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]

        result = train_models(X, y, "classification")

        assert not result.leaderboard.empty
        scores = result.leaderboard["Accuracy"].dropna()
        assert list(scores) == sorted(scores, reverse=True), "not ranked best-first"

    def test_selects_best_model(self, classification_df, cheap_classifiers):
        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]

        result = train_models(X, y, "classification")

        assert result.best_model_name is not None
        assert result.best_estimator is not None
        assert result.best_pipeline is not None
        assert result.task_type == "classification"

    def test_best_pipeline_predicts_from_raw_features(
        self, classification_df, cheap_classifiers
    ):
        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]

        result = train_models(X, y, "classification")
        predictions = result.best_pipeline.predict(X)

        assert len(predictions) == len(X)

    def test_reports_classification_metrics(self, classification_df, cheap_classifiers):
        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]

        result = train_models(X, y, "classification")

        for metric in ("Accuracy", "Precision", "Recall", "F1 Score"):
            assert metric in result.leaderboard.columns


class TestTrainModelsRegression:
    def test_returns_r2_leaderboard(self, regression_df, cheap_regressors):
        X = regression_df.drop(columns=["price"])
        y = regression_df["price"]

        result = train_models(X, y, "regression")

        assert "R2 Score" in result.leaderboard.columns
        assert result.best_model_name is not None
        assert result.task_type == "regression"

    def test_learns_a_linear_signal(self, regression_df, cheap_regressors):
        """price is ~3*num_a + noise, so a linear model should fit it well."""
        X = regression_df.drop(columns=["price"])
        y = regression_df["price"]

        result = train_models(X, y, "regression")

        assert result.best_metrics["R2 Score"] > 0.9


class TestLeakFreeTraining:
    def test_preprocessor_is_fitted_on_the_training_fold_only(
        self, classification_df, cheap_classifiers, monkeypatch
    ):
        """The preprocessor must never see all the rows it is evaluated on."""
        seen_row_counts = []
        original = model_trainer.build_preprocessor

        def spy(X):
            pipeline = original(X)
            # The trainer fits via fit_transform; wrap both entry points so the
            # assertion cannot be silently bypassed by a future refactor.
            for method_name in ("fit", "fit_transform"):
                real_method = getattr(pipeline, method_name)

                def recording(X_fit, *args, _real=real_method, **kwargs):
                    seen_row_counts.append(len(X_fit))
                    return _real(X_fit, *args, **kwargs)

                setattr(pipeline, method_name, recording)
            return pipeline

        monkeypatch.setattr(model_trainer, "build_preprocessor", spy)

        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]
        train_models(X, y, "classification")

        assert seen_row_counts, "preprocessor was never fitted"
        # 80/20 split: the fit must see strictly fewer rows than the dataset.
        assert all(count < len(X) for count in seen_row_counts)


class TestFailureHandling:
    def test_failing_model_is_recorded_not_raised(self, classification_df, monkeypatch):
        class Exploding:
            def fit(self, X, y):
                raise RuntimeError("boom")

            def get_params(self, deep=True):
                return {}

        monkeypatch.setattr(
            model_trainer,
            "_base_models",
            lambda task_type: {
                "Good": DummyClassifier(strategy="most_frequent"),
                "Exploding": Exploding(),
            },
        )
        monkeypatch.setattr(model_trainer, "PARAM_GRIDS", {})

        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]
        result = train_models(X, y, "classification")

        rows = result.leaderboard.set_index("Model")
        assert "boom" in str(rows.loc["Exploding", "Error"])
        # A single failure must not prevent a winner from being chosen.
        assert result.best_model_name == "Good"

    def test_unsupported_task_type_rejected(self, classification_df):
        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]

        with pytest.raises(ValueError, match="Unsupported task type"):
            train_models(X, y, "clustering")


class TestEvaluateModel:
    def test_classification_metrics(self, classification_df):
        X = classification_df.drop(columns=["target"]).select_dtypes("number")
        y = classification_df["target"]
        model = DummyClassifier(strategy="most_frequent").fit(X, y)

        metrics = evaluate_model(model, X, y, "classification")

        assert set(metrics) == {
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Confusion Matrix",
        }
        assert 0.0 <= metrics["Accuracy"] <= 1.0

    def test_regression_metrics(self, regression_df):
        X = regression_df.drop(columns=["price"]).select_dtypes("number")
        y = regression_df["price"]
        model = LinearRegression().fit(X, y)

        metrics = evaluate_model(model, X, y, "regression")

        assert set(metrics) == {
            "R2 Score",
            "Mean Squared Error",
            "Root Mean Squared Error",
            "Mean Absolute Error",
        }
        assert metrics["Root Mean Squared Error"] == pytest.approx(
            np.sqrt(metrics["Mean Squared Error"])
        )

    def test_unsupported_task_type(self, classification_df):
        X = classification_df.drop(columns=["target"]).select_dtypes("number")
        y = classification_df["target"]
        model = DummyClassifier().fit(X, y)

        with pytest.raises(ValueError, match="Unsupported task type"):
            evaluate_model(model, X, y, "ranking")


def test_primary_metric_mapping():
    assert PRIMARY_METRIC["classification"] == "Accuracy"
    assert PRIMARY_METRIC["regression"] == "R2 Score"
