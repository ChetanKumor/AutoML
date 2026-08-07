"""Inference utilities: load a saved :class:`ModelArtifact` and predict."""

from __future__ import annotations

import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.logging_utils import setup_logger
from utils.model_artifact import ModelArtifact

logger = setup_logger("PredictModule")

PREDICTION_COLUMN = "Predicted_Target"


def make_prediction(input_df: pd.DataFrame, model_filepath: str) -> pd.DataFrame:
    """Load a saved model artifact and make predictions on ``input_df``.

    Args:
        input_df: Raw input data. Must share the feature column structure of the
            training data (excluding the target column).
        model_filepath: Path to a ``.pkl`` file containing a
            :class:`~utils.model_artifact.ModelArtifact`.

    Returns:
        A copy of ``input_df`` with an added ``Predicted_Target`` column.

    Raises:
        FileNotFoundError: If ``model_filepath`` does not exist.
        ValueError: If the file does not contain a valid artifact.
    """
    if not os.path.exists(model_filepath):
        logger.error("Model file not found: %s", model_filepath)
        raise FileNotFoundError(f"Model file not found at {model_filepath}")

    artifact = ModelArtifact.load(model_filepath)
    logger.info(
        "Loaded artifact '%s' (task=%s) from %s",
        artifact.model_name or type(artifact.model).__name__,
        artifact.task_type,
        model_filepath,
    )

    data_to_predict = input_df.copy()

    logger.info("Applying preprocessor to input of shape %s", data_to_predict.shape)
    processed = artifact.preprocessor.transform(data_to_predict)

    logger.info("Predicting on %s processed rows", getattr(processed, "shape", ["?"])[0])
    predictions = artifact.model.predict(processed)

    # Decode integer labels back to their original classes for classification.
    if isinstance(artifact.label_encoder, LabelEncoder):
        logger.info("Inverse-transforming predictions via LabelEncoder")
        predictions = artifact.label_encoder.inverse_transform(predictions)

    result_df = input_df.copy()
    result_df[PREDICTION_COLUMN] = predictions
    logger.info("Prediction complete; produced %d predictions", len(result_df))
    return result_df
