"""Logging configuration for FracFeedExtractor.

Call setup_logging() once at the start of any entry-point script
(__main__ block or CLI main()). All modules then use the standard:

    import logging
    log = logging.getLogger(__name__)

No custom logger module, no special imports elsewhere.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "fracfeed.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB per file
    backup_count: int = 5,
) -> None:
    """Attach a WARNING+ rotating file handler to the root logger.

    Called once at process startup. All subsequent getLogger() calls
    in any module automatically inherit this handler.

    Console output is intentionally left to print() — this only
    persists warnings and errors to disk between runs.

    Args:
        log_dir:      Directory for log files (created if absent).
        log_file:     Log filename inside log_dir.
        max_bytes:    Max size of a single log file before rotation.
        backup_count: Number of rotated backup files to retain.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        Path(log_dir) / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(logging.WARNING)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)
