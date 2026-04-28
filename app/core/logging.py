"""Logging configuration helpers."""

import logging
from logging.config import dictConfig


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure global logging once at app startup."""
    format_string = (
        "%(asctime)s %(levelname)s %(name)s %(message)s"
        if json_logs
        else "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": format_string,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {
                "level": level,
                "handlers": ["console"],
            },
        }
    )

    logging.getLogger(__name__).debug(
        "Logging configured", extra={"json_logs": json_logs}
    )
