"""Tests for the model persistence contract.

Regression coverage for the bug where training saved a 4-tuple while inference
unpacked a 3-tuple, breaking every prediction.
"""

from __future__ import annotations

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.model_artifact import CLASSIFICATION, REGRESSION, ModelArtifact


@pytest.fixture
def artifact(classification_df) -> ModelArtifact:
    X = classification_df.drop(columns=["target"]).select_dtypes("number")
    y = classification_df["target"]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)
    return ModelArtifact(
        model=model,
        preprocessor=scaler,
        task_type=CLASSIFICATION,
        target_column="target",
        model_name="LogisticRegression",
        feature_names=list(X.columns),
    )


def test_round_trip_preserves_fields(artifact, tmp_path):
    path = tmp_path / "artifact.pkl"
    artifact.save(str(path))
    loaded = ModelArtifact.load(str(path))

    assert loaded.task_type == artifact.task_type
    assert loaded.target_column == artifact.target_column
    assert loaded.model_name == artifact.model_name
    assert loaded.feature_names == artifact.feature_names
    assert loaded.is_classification


def test_loaded_model_predicts(artifact, classification_df, tmp_path):
    path = tmp_path / "artifact.pkl"
    artifact.save(str(path))
    loaded = ModelArtifact.load(str(path))

    X = classification_df.drop(columns=["target"]).select_dtypes("number")
    predictions = loaded.model.predict(loaded.preprocessor.transform(X))
    assert len(predictions) == len(X)


def test_load_rejects_legacy_tuple(artifact, tmp_path):
    """The old (model, preprocessor, encoder, task_type) tuple must not load."""
    path = tmp_path / "legacy.pkl"
    joblib.dump(
        (artifact.model, artifact.preprocessor, None, CLASSIFICATION), str(path)
    )

    with pytest.raises(ValueError, match="does not contain a ModelArtifact"):
        ModelArtifact.load(str(path))


def test_invalid_task_type_rejected(artifact):
    with pytest.raises(ValueError, match="task_type must be one of"):
        ModelArtifact(
            model=artifact.model,
            preprocessor=artifact.preprocessor,
            task_type="clustering",
            target_column="target",
        )


def test_is_classification_flag(artifact):
    assert artifact.is_classification

    regressor = ModelArtifact(
        model=artifact.model,
        preprocessor=artifact.preprocessor,
        task_type=REGRESSION,
        target_column="price",
    )
    assert not regressor.is_classification


def test_created_at_is_populated(artifact):
    assert artifact.created_at
    assert "T" in artifact.created_at  # ISO-8601
