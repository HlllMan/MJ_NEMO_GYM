"""
MJ NeMo Gym - Offline Scoring Functions for 6 Domains

This package provides standalone reward functions that can run without
the NemoGym server infrastructure.

Usage:
    from mjnemogym import score_fn_dict, get_score_fn

    # Get score function by data_source
    score_fn = score_fn_dict["nemogym_math"]
    reward = score_fn(model_output="\\boxed{42}", label="42")

    # Or use the helper function
    reward = get_score_fn("nemogym_math")(model_output="\\boxed{42}", label="42")
"""

import logging
import sys
import traceback

# Configure module-level logger for Ray compatibility
# Force logs to stderr (which Ray captures) with immediate flush
_logger = logging.getLogger("mjnemogym")
_logger.setLevel(logging.DEBUG)

# Only add handler if not already configured (avoid duplicates)
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s][mjnemogym][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(_handler)
    _logger.propagate = False  # Don't propagate to root logger

from mjnemogym.math_with_judge.score import score_fn as math_score_fn
from mjnemogym.code_gen.score import score_fn as code_score_fn
from mjnemogym.mcqa.score import score_fn as mcqa_score_fn
from mjnemogym.instruction_following.score import score_fn as if_score_fn
from mjnemogym.structured_outputs.score import score_fn as so_score_fn
from mjnemogym.workplace_assistant.score import score_fn as wa_score_fn
import functools


def extract_final_answer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "model_output" in kwargs and isinstance(kwargs["model_output"], str):
            kwargs["model_output"] = kwargs["model_output"].split("<|end_of_thought|>")[
                -1
            ]
        elif args and isinstance(args[0], str):
            args_list = list(args)
            args_list[0] = args_list[0].split("<|end_of_thought|>")[-1]
            args = tuple(args_list)

        return func(*args, **kwargs)

    return wrapper


# Map data_source values (from parquet) to score functions
score_fn_dict = {
    "nemogym_math": extract_final_answer(math_score_fn),
    "nemogym_code": extract_final_answer(code_score_fn),
    "nemogym_mcqa": extract_final_answer(mcqa_score_fn),
    "nemogym_if": extract_final_answer(if_score_fn),
    "nemogym_structured": extract_final_answer(so_score_fn),
    "nemogym_workplace": extract_final_answer(wa_score_fn),
}


def get_score_fn(data_source: str):
    """
    Get score function by data_source.

    Args:
        data_source: The data_source value from the parquet file (e.g., "nemogym_math")

    Returns:
        The corresponding score function

    Raises:
        KeyError: If data_source is not recognized
    """
    if data_source not in score_fn_dict:
        raise KeyError(
            f"Unknown data_source: {data_source}. "
            f"Available: {list(score_fn_dict.keys())}"
        )
    return score_fn_dict[data_source]


def verl_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """
    Compute score compatible with verl's dapo.py compute_score interface.

    Drop-in replacement for verl.utils.reward_score.default_compute_score
    for NemoGym data sources.

    Args:
        data_source: The data_source from parquet (e.g., "nemogym_math")
        solution_str: Model's generated response
        ground_truth: Ground truth from reward_model.ground_truth (unused by most domains)
        extra_info: Domain-specific metadata from parquet extra_info field
        **kwargs: Additional arguments (ignored)

    Returns:
        float: Score value (typically 0.0 or 1.0)

    Raises:
        KeyError: If data_source is not a NemoGym domain

    Usage in dapo.py:
        from mjnemogym import verl_compute_score
        # Then pass as compute_score parameter or modify default_compute_score
    """
    if data_source not in score_fn_dict:
        raise KeyError(
            f"Unknown NemoGym data_source: {data_source}. "
            f"Available: {list(score_fn_dict.keys())}"
        )

    score_fn = score_fn_dict[data_source]
    try:
        score = float(score_fn(solution_str, extra_info))
        return score
    except Exception as e:
        # Use module logger + print to stderr for Ray visibility
        error_msg = f"[mjnemogym] Error computing score for {data_source}: {type(e).__name__}: {e}"
        _logger.error(error_msg, exc_info=True)
        # Also print to stderr with flush to ensure Ray captures it
        print(error_msg, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 0.0


__all__ = [
    "math_score_fn",
    "code_score_fn",
    "mcqa_score_fn",
    "if_score_fn",
    "so_score_fn",
    "wa_score_fn",
    "score_fn_dict",
    "get_score_fn",
    "verl_compute_score",
]
