"""
MJ NeMo Gym - Offline Scoring Functions for 9 Domains

This package provides standalone reward functions that can run without
the NemoGym server infrastructure.

Thread-safe timeout mechanism:
    Every call to verl_compute_score() is wrapped in a daemon thread with
    a per-domain timeout. This ensures bounded return time regardless of
    the execution context (main thread, ThreadPoolExecutor, Ray actor).

    ACTIVE PATH in verl (with ppo_megatron_trainer.yaml + use_reward_loop=True):
      ray_trainer.py → AgentLoopManager.generate_sequences()
        → AgentLoopWorker._compute_score()
          → RewardLoopWorker.compute_score.remote(data) [Ray actor]
            → experimental/dapo.py:run_single()
              → loop.run_in_executor(None, verl_compute_score)
                → THIS MODULE: _run_score_with_timeout() → daemon thread → score_fn()

    This timeout is the GLOBAL safety net. Individual domains (math, code)
    also have their own internal timeouts as defense-in-depth.

    No signal.alarm() is used, so it works in any thread.

Usage:
    from mjnemogym import verl_compute_score

    # Dict return (score + metadata):
    result = verl_compute_score(
        data_source="nemogym_math",
        solution_str="\\boxed{42}",
        ground_truth="42",
        extra_info={"expected_answer": "42"},
    )
    # result = {"score": 1.0, "timed_out": False, "elapsed": 0.01, "domain": "nemogym_math"}
"""

import logging
import threading
import time

from mjnemogym.math_with_judge.score import score_fn as math_score_fn
from mjnemogym.code_gen.score import score_fn as code_score_fn
from mjnemogym.mcqa.score import score_fn as mcqa_score_fn
from mjnemogym.instruction_following.score import score_fn as if_score_fn
from mjnemogym.structured_outputs.score import score_fn as so_score_fn
from mjnemogym.workplace_assistant.score import score_fn as wa_score_fn
from mjnemogym.qydomain import (
    typos_score_fn,
    connections_score_fn,
    unscrambling_score_fn,
)
import functools

_logger = logging.getLogger("mjnemogym.timeout")

# ---------------------------------------------------------------------------
# Per-domain timeout configuration (seconds)
#
# Rationale for each value:
#   nemogym_math:   3-method fallback chain. Internal timeouts: qy_parser=10s,
#                   math_verify=5s parse + 5s verify. Worst case all 3 fail:
#                   ~25s. Set 30s for headroom.
#   nemogym_code:   check_correctness runs user code in subprocess.
#                   _parent_timeout is passed down so child computes:
#                     join_timeout = parent_timeout - CLEANUP_MARGIN_S (15s)
#                   INVARIANT: join_timeout < this timeout → cleanup always runs.
#                   Safe to adjust this value — children auto-adapt.
#   nemogym_mcqa:   Pure regex matching. <0.01s typical. 5s is generous.
#   nemogym_if:     Instruction class instantiation + rule checks. <0.1s. 10s.
#   nemogym_structured: JSON schema validation. <0.01s. 5s.
#   nemogym_workplace:  State simulation with pandas DataFrames. <1s. 15s.
#   typos:          String matching. <0.01s. 5s.
#   connections:    Regex + grouping. <0.01s. 5s.
#   unscrambling:   Levenshtein distance on sentences. Can be slow. 10s.
# ---------------------------------------------------------------------------
DOMAIN_TIMEOUTS = {
    "nemogym_math": 30,
    "nemogym_code": 45,
    "nemogym_mcqa": 5,
    "nemogym_if": 10,
    "nemogym_structured": 5,
    "nemogym_workplace": 15,
    "typos": 5,
    "connections": 5,
    "unscrambling": 10,
}

# Absolute maximum for any domain (fallback if domain not in DOMAIN_TIMEOUTS)
DEFAULT_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Abandoned thread monitoring
#
# Daemon threads that exceed their timeout are "abandoned" - the caller
# returns immediately but the thread keeps running until the process exits.
# Track these for diagnostics.
# ---------------------------------------------------------------------------
_abandoned_thread_count = 0
_abandoned_lock = threading.Lock()
_total_timeout_count = 0
_total_error_count = 0


def get_timeout_stats() -> dict:
    """Return timeout monitoring stats. Call from logging/metrics hooks."""
    return {
        "abandoned_threads": _abandoned_thread_count,
        "total_timeouts": _total_timeout_count,
        "total_errors": _total_error_count,
        "active_threads": threading.active_count(),
    }


def _run_score_with_timeout(func, timeout: float, data_source: str, default=0.0):
    """Run a score function with a hard timeout using a daemon thread.

    Thread-safe: no signal.alarm(), works in any thread context.

    Compatible with verl's parallel execution (active path):
      AgentLoopWorker → RewardLoopWorker (Ray actor, async)
        → loop.run_in_executor(None, verl_compute_score)
          → THIS FUNCTION

    The daemon thread is abandoned after timeout. Since daemon=True, it will
    be killed when the process exits (Ray actor cleanup or process pool
    termination). The abandoned thread may continue consuming CPU until then,
    but the caller is unblocked and can proceed.

    Args:
        func: Callable that returns a float score
        timeout: Maximum seconds to wait
        data_source: Domain name for logging
        default: Value to return on timeout (default: 0.0)

    Returns:
        float: Score from func, or default on timeout/error
    """
    global _abandoned_thread_count, _total_timeout_count, _total_error_count

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
        with _abandoned_lock:
            _abandoned_thread_count += 1
            _total_timeout_count += 1
            count = _abandoned_thread_count
        _logger.warning(
            f"[TIMEOUT] {data_source} score_fn exceeded {timeout}s, "
            f"returning {default} (abandoned_threads={count})"
        )
        return default

    if exc_box[0] is not None:
        with _abandoned_lock:
            _total_error_count += 1
        _logger.warning(
            f"[ERROR] {data_source} score_fn raised "
            f"{type(exc_box[0]).__name__}: {exc_box[0]}"
        )
        return default

    return result_box[0]


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
    # NemoGym domains
    "nemogym_math": extract_final_answer(math_score_fn),
    "nemogym_code": extract_final_answer(code_score_fn),
    "nemogym_mcqa": extract_final_answer(mcqa_score_fn),
    "nemogym_if": extract_final_answer(if_score_fn),
    "nemogym_structured": extract_final_answer(so_score_fn),
    "nemogym_workplace": extract_final_answer(wa_score_fn),
    # QY domains - data_source is the task type, extra_info contains "label"
    "typos": extract_final_answer(typos_score_fn),
    "connections": extract_final_answer(connections_score_fn),
    "unscrambling": extract_final_answer(unscrambling_score_fn),
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
) -> dict:
    """
    Compute score compatible with verl's dapo.py compute_score interface.

    Drop-in replacement for verl.utils.reward_score.default_compute_score
    for NemoGym data sources.

    Thread-safe global timeout wrapper:
        Every call is executed inside a daemon thread with a per-domain timeout
        (see DOMAIN_TIMEOUTS). If the score function doesn't return within the
        timeout, score=0.0 is returned and the caller is unblocked.

    Active verl call path (see plans/thread_safe_timeout/call_stack_trace.md):
        ray_trainer.py:1450 → AgentLoopManager.generate_sequences()
          → AgentLoopWorker._compute_score()  [agent_loop.py:701]
            → RewardLoopWorker.compute_score.remote(data)  [agent_loop.py:728]
              → experimental/dapo.py:run_single()
                → loop.run_in_executor(None, THIS_FUNCTION)

    Return type compatibility:
        Returns dict with "score" key. Both DAPORewardManager variants handle
        dict returns correctly:
        - experimental/dapo.py:94: isinstance(result, dict) → score = result["score"]
        - workers/dapo.py:142: isinstance(result, dict) → score = result["score"]

    Args:
        data_source: The data_source from parquet (e.g., "nemogym_math")
        solution_str: Model's generated response
        ground_truth: Ground truth from reward_model.ground_truth
        extra_info: Domain-specific metadata from parquet extra_info field
        **kwargs: Additional arguments (ignored, for forward compatibility)

    Returns:
        dict: {"score": float, "timed_out": bool, "elapsed": float, "domain": str}
              score is typically 0.0 or 1.0. Returns 0.0 on timeout or error.

    Raises:
        KeyError: If data_source is not a NemoGym domain
    """
    if data_source not in score_fn_dict:
        raise KeyError(
            f"Unknown NemoGym data_source: {data_source}. "
            f"Available: {list(score_fn_dict.keys())}"
        )

    if extra_info is None:
        extra_info = {}

    timeout = DOMAIN_TIMEOUTS.get(data_source, DEFAULT_TIMEOUT)
    fn = score_fn_dict[data_source]

    # Pass parent timeout to children so they can compute their own
    # timeouts relative to ours. This ensures the invariant:
    #   child_timeout < parent_timeout  →  cleanup always runs
    # No hardcoded caps needed — children derive from this value.
    extra_info["_parent_timeout"] = timeout

    t0 = time.time()
    score = _run_score_with_timeout(
        func=lambda: float(fn(solution_str, extra_info)),
        timeout=timeout,
        data_source=data_source,
        default=0.0,
    )
    elapsed = time.time() - t0
    timed_out = elapsed >= timeout * 0.99

    if elapsed > timeout * 0.8:
        _logger.info(
            f"[SLOW] {data_source} took {elapsed:.1f}s (timeout={timeout}s)"
        )

    return {
        "score": score,
        "timed_out": timed_out,
        "elapsed": round(elapsed, 3),
        "domain": data_source,
    }


__all__ = [
    "math_score_fn",
    "code_score_fn",
    "mcqa_score_fn",
    "if_score_fn",
    "so_score_fn",
    "wa_score_fn",
    "typos_score_fn",
    "connections_score_fn",
    "unscrambling_score_fn",
    "score_fn_dict",
    "get_score_fn",
    "verl_compute_score",
    "DOMAIN_TIMEOUTS",
    "DEFAULT_TIMEOUT",
    "get_timeout_stats",
]
