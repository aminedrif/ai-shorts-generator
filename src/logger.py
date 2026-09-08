import logging
import os
import sys
import traceback
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

from src.config import config


class ErrorTracker:
    """Thread-safe in-memory tracker for application exceptions and errors."""

    def __init__(self, max_history: int = 100):
        self._history: deque = deque(maxlen=max_history)

    def record_error(
        self,
        exc: Exception,
        module: str = "general",
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Captures exception details, formats traceback, and adds to history."""
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "module": module,
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": tb_str,
            "task_id": task_id,
            "context": context or {},
        }
        self._history.appendleft(error_record)
        return error_record

    def get_recent(self, limit: int = 50, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns recent error entries, optionally filtered by task_id."""
        errors = list(self._history)
        if task_id:
            errors = [e for e in errors if e.get("task_id") == task_id]
        return errors[:limit]

    def clear(self):
        """Clears stored error entries."""
        self._history.clear()


error_tracker = ErrorTracker()


def setup_logger(name: str = "shorts_generator") -> logging.Logger:
    """Initializes and returns configured application logger."""
    app_logger = logging.getLogger(name)
    if app_logger.handlers:
        return app_logger

    app_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)

    # File Handlers (ensure logs directory exists)
    logs_dir = config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Main App Log
    app_log_file = logs_dir / "app.log"
    file_handler = RotatingFileHandler(
        str(app_log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)

    # Error Only Log
    error_log_file = logs_dir / "error.log"
    err_file_handler = RotatingFileHandler(
        str(error_log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    err_file_handler.setLevel(logging.ERROR)
    err_file_handler.setFormatter(formatter)
    app_logger.addHandler(err_file_handler)

    return app_logger


logger = setup_logger()


def track_error(
    exc: Exception,
    module: str = "pipeline",
    task_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Logs the exception and records it into the error tracker."""
    record = error_tracker.record_error(exc=exc, module=module, task_id=task_id, context=context)
    logger.error(
        f"[{module.upper()}] {exc.__class__.__name__}: {exc} (Task: {task_id or 'none'})",
        exc_info=True,
    )
    return record
