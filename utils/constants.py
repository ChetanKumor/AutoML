"""Application configuration.

All paths and tunables are read from the environment with sensible defaults, so
the same code runs unchanged locally, in CI and in a container.

Environment variables:
    AUTOML_MODEL_DIR    directory for saved model artifacts (default: saved_models)
    AUTOML_ENCODER_DIR  directory for saved encoders       (default: saved_encoders)
    AUTOML_LOG_DIR      directory for log files            (default: logs)
    AUTOML_MAX_UPLOAD_MB  maximum accepted upload size in MB (default: 200)
    AUTOML_RANDOM_STATE   global random seed              (default: 42)
    AUTOML_TEST_SIZE      held-out test fraction          (default: 0.2)

This module intentionally performs **no** filesystem side effects at import
time; call :func:`ensure_directories` from an application entrypoint instead.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _path_from_env(var: str, default: str) -> Path:
    """Resolve a directory from ``var``, relative to the project root."""
    value = Path(os.getenv(var, default))
    return value if value.is_absolute() else BASE_DIR / value


# === Directories ===
MODEL_DIR = _path_from_env("AUTOML_MODEL_DIR", "saved_models")
ENCODER_DIR = _path_from_env("AUTOML_ENCODER_DIR", "saved_encoders")
LOG_DIR = _path_from_env("AUTOML_LOG_DIR", "logs")


def ensure_directories() -> None:
    """Create the runtime directories. Call once from an entrypoint."""
    for directory in (MODEL_DIR, ENCODER_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# === Upload limits ===
MAX_FILE_SIZE_MB = int(os.getenv("AUTOML_MAX_UPLOAD_MB", "200"))
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# === App info ===
APP_TITLE = "🤖 Robo Data Scientist"
APP_DESCRIPTION = (
    "Upload a tabular dataset, train and rank a family of models automatically, "
    "then use the winner to make predictions."
)

# === ML configuration ===
DEFAULT_RANDOM_STATE = int(os.getenv("AUTOML_RANDOM_STATE", "42"))
DEFAULT_TEST_SIZE = float(os.getenv("AUTOML_TEST_SIZE", "0.2"))
