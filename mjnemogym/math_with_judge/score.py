# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Math scoring with fallback chain:
# 1. QY math parser (fast, regex-based extraction + math_verify)
# 2. DAPO (regex normalization + string comparison)
# 3. MathVerify (full symbolic parsing)

import logging
import sys

from mjnemogym.math_with_judge import qy_parser
from mjnemogym.math_with_judge import dapo
from mjnemogym.math_with_judge import math_verify_method

# Configure module-level logger for Ray compatibility
_logger = logging.getLogger("mjnemogym.math_with_judge")
_logger.setLevel(logging.DEBUG)

if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s][mjnemogym.math][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(_handler)
    _logger.propagate = False

logging.getLogger("math_verify").setLevel(logging.CRITICAL)


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Math scoring function with fallback chain.

    Tries methods in order:
    1. QY math parser (fast regex + math_verify)
    2. DAPO (regex normalization + string comparison)
    3. MathVerify (full symbolic parsing)

    Returns on first non-zero result.

    Args:
        model_output: Model-generated answer text (should contain \\boxed{answer})
        extra_info: Dictionary containing expected_answer

    Returns:
        float: 1.0 (correct) or 0.0 (incorrect)
    """
    expected_answer = extra_info.get("expected_answer", "")
    if not expected_answer:
        _logger.debug(f"[mjnemogym.math] expected_answer is empty. extra_info keys: {list(extra_info.keys())}")
        return 0.0

    expected_answer = str(expected_answer)

    # Method 1: QY math parser
    try:
        reward = qy_parser.score_fn(model_output, expected_answer)
        if reward > 0:
            return reward
    except Exception as e:
        _logger.debug(f"QY method failed: {e}")

    # Method 2: DAPO
    try:
        reward = dapo.score_fn(model_output, expected_answer)
        if reward > 0:
            return reward
    except Exception as e:
        _logger.debug(f"DAPO method failed: {e}")

    # Method 3: MathVerify
    try:
        reward = math_verify_method.score_fn(model_output, expected_answer)
        return reward
    except Exception as e:
        _logger.debug(f"MathVerify method failed: {e}")
        return 0.0
