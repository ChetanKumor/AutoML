"""Centralised logging configuration.

Handlers are attached once to a single root application logger; named loggers
are children of it. This replaces the previous approach of giving every module
its own file handler, which produced a new timestamped log file per module per
process (the reason 45 stray log files had accumulated in the repository).

Configuration is environment-driven:
    AUTOML_LOG_DIR    directory for the rotating log file (default: ./logs)
    AUTOML_LOG_LEVEL  console/file level, e.g. DEBUG, INFO (default: INFO)
    AUTOML_LOG_TO_FILE set to "0"/"false" to disable file logging entirely
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER_NAME = "automl"
LOG_FILENAME = "automl.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 3

_configured = False


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", ""}


def configure_logging(force: bool = False) -> logging.Logger:
    """Configure and return the root application logger.

    Idempotent: repeated calls (common under Streamlit's script reruns) will not
    stack duplicate handlers.

    Args:
        force: Reconfigure even if logging was already set up.

    Returns:
        The configured ``automl`` logger.
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    if _configured and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    level_name = os.getenv("AUTOML_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    # Application logs are emitted by our handlers only; don't double-log
    # through the root logger's handlers.
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(console)

    if _env_flag("AUTOML_LOG_TO_FILE", default=True):
        log_dir = Path(os.getenv("AUTOML_LOG_DIR", "logs"))
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            # A single rotating file, rather than one file per process/module.
            file_handler = RotatingFileHandler(
                log_dir / LOG_FILENAME,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            logger.addHandler(file_handler)
        except OSError as exc:  # read-only filesystem, permissions, ...
            logger.warning("File logging disabled (%s): %s", log_dir, exc)

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-scoped child of the application logger.

    Deliberately does **not** configure handlers: library modules acquire their
    logger at import time, and creating log directories or files as a side
    effect of an import is surprising and untestable. Entrypoints
    (``app.py``, ``train.py``) call :func:`configure_logging` instead.

    Args:
        name: Short component name, e.g. ``"train_cli"``. ``None`` returns the
            application root logger.
    """
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


#: Backwards-compatible alias. Prefer :func:`get_logger`.
setup_logger = get_logger
