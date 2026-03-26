"""Logging configuration for the Latence SDK."""

from __future__ import annotations

import logging
import os
from typing import Literal

# Create SDK logger with NullHandler to prevent "No handler found" warnings
logger = logging.getLogger("latence")
logger.addHandler(logging.NullHandler())

# Log levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel | int = "WARNING",
    *,
    fmt: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """
    Configure logging for the Latence SDK.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int
        fmt: Custom log format string
        handler: Custom handler (defaults to StreamHandler)

    Example:
        >>> import latence
        >>> latence.setup_logging("DEBUG")  # See all requests and retries

        >>> # Custom format
        >>> latence.setup_logging("INFO", fmt="%(asctime)s - %(message)s")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)

    # Remove existing non-NullHandler handlers, keep NullHandler
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]

    if handler is None:
        handler = logging.StreamHandler()

    log_fmt = fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(log_fmt))

    logger.addHandler(handler)


def _init_logging() -> None:
    """Initialize logging from environment variable."""
    env_level = os.environ.get("LATENCE_LOG_LEVEL", "").upper()
    if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        setup_logging(env_level)  # type: ignore


# Initialize from environment on import
_init_logging()


def log_request(method: str, path: str, **kwargs) -> None:
    """Log an outgoing request."""
    logger.debug(f"Request: {method} {path}")
    if kwargs and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"  Params: {kwargs}")


def log_response(status_code: int, path: str, credits_used: float | None = None) -> None:
    """Log an incoming response."""
    msg = f"Response: {status_code} {path}"
    if credits_used is not None:
        msg += f" (credits: {credits_used})"
    logger.debug(msg)


def log_retry(attempt: int, delay: float, reason: str) -> None:
    """Log a retry attempt."""
    logger.info(f"Retry {attempt}: waiting {delay:.2f}s ({reason})")


def log_error(error: Exception, context: str = "") -> None:
    """Log an error."""
    msg = f"Error: {error}"
    if context:
        msg = f"{context}: {msg}"
    logger.error(msg)
