# MathVerify method - Full symbolic parsing using math_verify library
#
# Thread-safe implementation using parse() with parsing_timeout=None
# and custom timeout wrapper.

import concurrent.futures
import logging
from typing import Optional

from math_verify import grader
from math_verify.parser import parse, ExprExtractionConfig, LatexExtractionConfig

# Timeout settings (in seconds)
PARSE_TIMEOUT = 5.0
VERIFY_TIMEOUT = 5.0

_logger = logging.getLogger("mjnemogym.math_with_judge.math_verify_method")


def _run_with_timeout(func, timeout: float, default=None):
    """Run a function with a thread-safe timeout."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        _logger.warning(f"Operation timed out after {timeout}s")
        return default
    except Exception as e:
        _logger.debug(f"Operation failed: {e}")
        return default


def verify_answer(expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
    """
    Verify the correctness of a generated answer using math_verify library.

    Thread-safe implementation using parse() with parsing_timeout=None.

    Args:
        expected_answer: Ground truth answer (without \\boxed{})
        generated_answer: Model-generated answer text (should contain \\boxed{answer})

    Returns:
        (reward, extracted_answer): reward is 0.0 or 1.0, extracted_answer is the parsed answer
    """
    gold_extraction_target = (LatexExtractionConfig(),)
    pred_extraction_target = (
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    )

    try:
        ground_truth_parsable = "\\boxed{" + expected_answer + "}"

        # Use parse() with our own thread-safe timeout wrapper
        extracted_golds = _run_with_timeout(
            lambda: parse(
                ground_truth_parsable,
                gold_extraction_target,
                parsing_timeout=None
            ),
            timeout=PARSE_TIMEOUT,
            default=[]
        )

        extracted_preds = _run_with_timeout(
            lambda: parse(
                generated_answer,
                pred_extraction_target,
                parsing_timeout=None
            ),
            timeout=PARSE_TIMEOUT,
            default=[]
        )

        # Check if we extracted anything
        if not extracted_golds:
            _logger.warning(f"Could not extract gold answer from: {ground_truth_parsable[:100]}")
            return 0.0, None

        if not extracted_preds:
            _logger.debug(f"Could not extract prediction from model output (first 200 chars): {generated_answer[:200]}")
            return 0.0, None

        # Check if any prediction matches any gold
        matched_pred = None
        for pred in extracted_preds:
            for gold in extracted_golds:
                is_match = _run_with_timeout(
                    lambda g=gold, p=pred: grader.verify(g, p, timeout_seconds=None),
                    timeout=VERIFY_TIMEOUT,
                    default=False
                )
                if is_match:
                    matched_pred = pred
                    break
            if matched_pred is not None:
                break

        if matched_pred is not None:
            try:
                extracted_answer = str(matched_pred)
            except Exception:
                extracted_answer = None
            return 1.0, extracted_answer
        else:
            try:
                extracted_answer = str(extracted_preds[0]) if extracted_preds else None
            except Exception:
                extracted_answer = None
            return 0.0, extracted_answer

    except Exception as e:
        _logger.debug(f"verify_answer failed: {type(e).__name__}: {e}")
        return 0.0, None


def score_fn(model_output: str, expected_answer: str) -> float:
    """Score function compatible with fallback chain."""
    reward, _ = verify_answer(expected_answer, model_output)
    return reward
