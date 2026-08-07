"""Tests for inference, including the full train -> save -> load -> predict path."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression

from utils import model_trainer
from utils.model_artifact import ModelArtifact
from utils.model_trainer import train_models
from utils.predict import PREDICTION_COLUMN, make_prediction


@pytest.fixture
def trained_artifact_path(classification_df, tmp_path, monkeypatch):
    """Train a cheap model end-to-end and persist it."""
    monkeypatch.setattr(
        model_trainer,
        "_base_models",
        lambda task_type: {"Dummy": DummyClassifier(strategy="most_frequent")},
    )
    monkeypatch.setattr(model_trainer, "PARAM_GRIDS", {})

    X = classification_df.drop(columns=["target"])
    y = classification_df["target"]
    result = train_models(X, y, "classification")

    path = tmp_path / "model.pkl"
    ModelArtifact(
        model=result.best_estimator,
        preprocessor=result.preprocessor,
        task_type="classification",
        target_column="target",
        model_name=result.best_model_name or "",
        feature_names=result.feature_names,
        metrics=result.best_metrics,
    ).save(str(path))
    return path


def test_round_trip_train_save_predict(trained_artifact_path, classification_df):
    """The regression test for the save/load contract drift bug."""
    X = classification_df.drop(columns=["target"])

    predictions = make_prediction(X, str(trained_artifact_path))

    assert PREDICTION_COLUMN in predictions.columns
    assert len(predictions) == len(X)
    assert predictions[PREDICTION_COLUMN].notna().all()


def test_original_columns_are_preserved(trained_artifact_path, classification_df):
    X = classification_df.drop(columns=["target"])
    predictions = make_prediction(X, str(trained_artifact_path))

    for column in X.columns:
        assert column in predictions.columns


def test_input_frame_is_not_mutated(trained_artifact_path, classification_df):
    X = classification_df.drop(columns=["target"])
    before = X.copy()

    make_prediction(X, str(trained_artifact_path))

    pd.testing.assert_frame_equal(X, before)


def test_missing_model_file_raises(classification_df, tmp_path):
    with pytest.raises(FileNotFoundError):
        make_prediction(classification_df, str(tmp_path / "absent.pkl"))


def test_labels_are_decoded_back_to_original_classes(tmp_path):
    """A LabelEncoder in the artifact must be applied in reverse."""
    df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0] * 10, "label": ["yes", "no"] * 20})
    from utils.data_utils import analyze_and_prepare_target

    X, y, encoder = analyze_and_prepare_target(df.copy(), "label")

    from utils.feature_engineer import build_preprocessor

    preprocessor = build_preprocessor(X).fit(X)
    model = DummyClassifier(strategy="most_frequent").fit(preprocessor.transform(X), y)

    path = tmp_path / "encoded.pkl"
    ModelArtifact(
        model=model,
        preprocessor=preprocessor,
        task_type="classification",
        target_column="label",
        label_encoder=encoder,
    ).save(str(path))

    predictions = make_prediction(X, str(path))
    assert set(predictions[PREDICTION_COLUMN]) <= {"yes", "no"}


def test_regression_predictions_are_continuous(regression_df, tmp_path):
    from utils.feature_engineer import build_preprocessor

    X = regression_df.drop(columns=["price"])
    y = regression_df["price"]
    preprocessor = build_preprocessor(X).fit(X)
    model = LinearRegression().fit(preprocessor.transform(X), y)

    path = tmp_path / "reg.pkl"
    ModelArtifact(
        model=model,
        preprocessor=preprocessor,
        task_type="regression",
        target_column="price",
    ).save(str(path))

    predictions = make_prediction(X, str(path))
    assert predictions[PREDICTION_COLUMN].dtype.kind == "f"
