"""Command-line entrypoint for reproducible, headless model training.

Trains the full model search on a dataset and writes a versioned
:class:`~utils.model_artifact.ModelArtifact` plus a leaderboard CSV. Having a
scriptable path (rather than only the Streamlit UI) is what makes training
reproducible in CI, cron jobs and container entrypoints.

Examples:
    python train.py --data heart-disease.csv --target target
    python train.py --data data/input.csv --output-dir artifacts
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.auto_target_identifier import detect_task_type
from utils.constants import ensure_directories
from utils.data_utils import (
    analyze_and_prepare_target,
    detect_target_column,
    load_dataset,
)
from utils.logging_utils import configure_logging, get_logger
from utils.model_artifact import ModelArtifact
from utils.model_trainer import PRIMARY_METRIC, train_models
from utils.validation import (
    DatasetValidationError,
    validate_dataframe,
    validate_target,
    validate_upload,
)

logger = get_logger("train_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and rank models on a tabular dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="Path to a CSV or Excel dataset."
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saved_models"),
        help="Directory for the model artifact and leaderboard.",
    )
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default=None,
        help="Override the inferred task type.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the training pipeline. Returns a process exit code."""
    args = parse_args(argv)
    # Configure logging only after argument parsing, so --help/usage errors
    # exit without creating a log directory.
    configure_logging()
    ensure_directories()

    if not args.data.exists():
        logger.error("Dataset not found: %s", args.data)
        return 1

    logger.info("Loading dataset from %s", args.data)
    try:
        validate_upload(str(args.data))
        df = load_dataset(str(args.data))
        validate_dataframe(df)
        target_col = args.target or detect_target_column(df)
        validate_target(df, target_col)
    except DatasetValidationError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("Could not read %s: %s", args.data, exc)
        return 1

    logger.info("Target column: %s", target_col)

    X, y, label_encoder = analyze_and_prepare_target(df.copy(), target_col)
    task_type = args.task_type or detect_task_type(pd.Series(y))
    logger.info("Task type: %s", task_type)

    result = train_models(X, pd.Series(y), task_type)
    if result.best_estimator is None:
        logger.error("No model trained successfully; nothing to save.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifact_path = args.output_dir / f"model_{result.best_model_name}_{timestamp}.pkl"
    ModelArtifact(
        model=result.best_estimator,
        preprocessor=result.preprocessor,
        task_type=task_type,
        target_column=target_col,
        label_encoder=label_encoder,
        model_name=result.best_model_name or "",
        feature_names=result.feature_names,
        metrics=result.best_metrics,
    ).save(str(artifact_path))

    leaderboard_path = args.output_dir / f"leaderboard_{timestamp}.csv"
    result.leaderboard.to_csv(leaderboard_path, index=False)

    metric_name = PRIMARY_METRIC[task_type]
    best_score = result.best_metrics.get(metric_name)
    logger.info(
        "Best model: %s (%s=%s)",
        result.best_model_name,
        metric_name,
        f"{best_score:.4f}" if isinstance(best_score, float) else best_score,
    )
    logger.info("Artifact:    %s", artifact_path)
    logger.info("Leaderboard: %s", leaderboard_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
