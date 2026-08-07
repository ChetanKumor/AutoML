"""Serializable container bundling everything needed to serve predictions.

Historically the trained model, its preprocessing pipeline, the label encoder
and the task type were persisted as a bare tuple and unpacked positionally at
inference time. The producer (the Streamlit app) and the consumer
(:mod:`utils.predict`) each hard-coded the tuple's shape independently, so when
the app started saving a fourth element (``task_type``) the predictor -- still
unpacking three -- broke for every saved model.

``ModelArtifact`` makes that contract explicit and single-sourced: training
builds one, inference loads one, and adding a field can never silently break
the unpacking on the other side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib

CLASSIFICATION = "classification"
REGRESSION = "regression"
VALID_TASK_TYPES = (CLASSIFICATION, REGRESSION)


@dataclass
class ModelArtifact:
    """Bundle of the fitted objects and metadata required for inference.

    Attributes:
        model: The fitted estimator selected as best.
        preprocessor: The fitted feature-engineering pipeline. Applied to raw
            input before ``model.predict``.
        task_type: Either ``"classification"`` or ``"regression"``.
        target_column: Name of the column the model was trained to predict.
        label_encoder: The :class:`~sklearn.preprocessing.LabelEncoder` used on
            the target for classification, or ``None`` when the target was
            numeric (regression / already-encoded).
        model_name: Human-readable identifier of the winning model.
        feature_names: Feature column names produced by the preprocessor, in
            order. Useful for introspection and drift checks.
        metrics: Evaluation metrics for the winning model.
        created_at: UTC ISO-8601 timestamp of when the artifact was created.
    """

    model: Any
    preprocessor: Any
    task_type: str
    target_column: str
    label_encoder: Any | None = None
    model_name: str = ""
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {VALID_TASK_TYPES}, got {self.task_type!r}"
            )

    @property
    def is_classification(self) -> bool:
        return self.task_type == CLASSIFICATION

    def save(self, path: str) -> None:
        """Serialize this artifact to ``path`` using joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> ModelArtifact:
        """Load an artifact from ``path``.

        Raises:
            ValueError: If the file does not contain a :class:`ModelArtifact`.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise ValueError(
                f"{path!r} does not contain a ModelArtifact "
                f"(got {type(obj).__name__}). It may have been produced by an "
                "incompatible version; please retrain the model."
            )
        return obj
