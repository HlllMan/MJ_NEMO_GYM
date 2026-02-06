# MathVerify method - Full symbolic parsing using math_verify library
#
# Thread-safe implementation using parse() with parsing_timeout=None
# and custom timeout wrapper.

import threading
from typing import Optional

from math_verify import grader
from math_verify.parser import parse, ExprExtractionConfig, LatexExtractionConfig

from mjnemogym.log import get_logger

# Timeout settings (in seconds)
PARSE_TIMEOUT = 5.0
VERIFY_TIMEOUT = 5.0

_logger = get_logger("math.math_verify")


def _run_with_timeout(func, timeout: float, default=None):
    """Run a function with a timeout using a daemon thread.

    Uses daemon=True so the thread is killed immediately when the worker
    process exits, preventing ProcessPoolExecutor shutdown from hanging.
    """
    result_box = [default]
    exc_box = [None]

    def wrapper():
        try:
            result_box[0] = func()
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        _logger.warning(f"Operation timed out after {timeout}s")
        return default
    if exc_box[0] is not None:
        _logger.debug(f"Operation failed: {exc_box[0]}")
        return default
    return result_box[0]


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
