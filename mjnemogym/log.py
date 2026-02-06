# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Centralized logging for all mjnemogym scoring domains.
#
# Usage:
#   from mjnemogym.log import get_logger
#   _logger = get_logger("math")   # creates logger "mjnemogym.math"
#
# Control via environment variables:
#   MJNEMOGYM_DEBUG=1      → DEBUG level (entry/exit, timing, values)
#   default (unset)        → WARNING level (errors, timeouts only)
#
#   MJNEMOGYM_LOG_FILE=path  → Also write all logs to this file
#   default (unset)          → Console only

import logging
import os

_DEBUG = os.environ.get("MJNEMOGYM_DEBUG", "0") == "1"
_LOG_FILE = os.environ.get("MJNEMOGYM_LOG_FILE", "")
_CONFIGURED = set()

_FMT = "[%(asctime)s][%(name)s][%(levelname)s][pid=%(process)d] %(message)s"
_DATEFMT = "%H:%M:%S"

# Shared file handler (created once, reused across all loggers)
_file_handler = None


def _get_file_handler() -> logging.FileHandler:
    """Lazily create a single shared file handler."""
    global _file_handler
    if _file_handler is None and _LOG_FILE:
        _file_handler = logging.FileHandler(_LOG_FILE, mode="a")
        _file_handler.setLevel(logging.DEBUG)  # File always gets everything
        _file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    return _file_handler


def get_logger(domain: str) -> logging.Logger:
    """
    Get a domain-specific logger with consistent formatting.

    Args:
        domain: Short domain name, e.g. "math", "code", "mcqa".
                Creates logger named "mjnemogym.{domain}".

    Returns:
        Configured logging.Logger instance.
    """
    name = f"mjnemogym.{domain}"
    logger = logging.getLogger(name)

    if name not in _CONFIGURED:
        _CONFIGURED.add(name)
        level = logging.DEBUG if _DEBUG else logging.WARNING
        logger.setLevel(level)

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(console)

        # File handler (if MJNEMOGYM_LOG_FILE is set)
        fh = _get_file_handler()
        if fh is not None:
            logger.setLevel(logging.DEBUG)  # Logger must accept DEBUG for file
            logger.addHandler(fh)

        logger.propagate = False

    return logger
