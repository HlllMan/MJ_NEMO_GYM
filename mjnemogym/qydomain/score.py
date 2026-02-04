# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone scoring functions for QY language domains:
# - typos: Exact match / substring check
# - connections: Word grouping puzzle
# - unscrambling: Plot sentence ordering

import logging
import re
import sys
import traceback
from typing import Optional

# Configure module-level logger for Ray compatibility
_logger = logging.getLogger("mjnemogym.qydomain")
_logger.setLevel(logging.DEBUG)

# Only add handler if not already configured
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s][mjnemogym.qydomain][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(_handler)
    _logger.propagate = False


# ==========================================
# Shared Helper Functions
# ==========================================


def levenshtein_distance(A, B) -> int:
    """
    Calculates the Levenshtein distance between two sequences A and B using Dynamic Programming.
    Works with both strings and lists.
    """
    N, M = len(A), len(B)
    dp = [[0 for _ in range(M + 1)] for _ in range(N + 1)]

    for j in range(M + 1):
        dp[0][j] = j
    for i in range(N + 1):
        dp[i][0] = i

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Insertion
                    dp[i][j - 1],      # Deletion
                    dp[i - 1][j - 1],  # Replacement
                )

    return dp[N][M]


def extract_answer(llm_answer: str) -> str:
    """Extracts the answer part from a string following the pattern '... --- answer --- ...'."""
    pattern = r".* --- (.*?) --- .*"
    match = re.search(pattern, llm_answer)
    return match.group(1) if match else llm_answer


# ==========================================
# Evaluator 1: Plot Unscrambling
# ==========================================


def extract_plot_summary(text: str) -> str:
    pattern = r"<PLOT_SUMMARY>(.*)</PLOT_SUMMARY>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        pattern = r"<PLOT_SUMMARY>(.*)"
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text


def plot_unscrambling_process_results(
    ground_truth: str, llm_answer: str, debug=False
) -> float:
    """
    Evaluates how well the LLM ordered sentences compared to the ground truth.
    Uses Levenshtein distance on the sentence indices.
    """
    # Extract relevant text
    llm_answer = extract_plot_summary(llm_answer)

    # Split into sentences
    gt_sentences = [s.strip() for s in ground_truth.split(".")]
    ans_sentences = [
        s.strip()
        for s in llm_answer.split(".")
        if s.strip() != "</PLOT_SUMMARY>" and s.strip() != "**End of Plot Summary**"
    ]

    # Filter empty sentences
    gt_sentences = [s for s in gt_sentences if s]
    ans_sentences = [s for s in ans_sentences if s]

    ans_ordering = []

    # Map ground truth sentences to the answer sentences
    for x in gt_sentences:
        if not ans_sentences:
            break

        # Replacement for difflib.get_close_matches:
        # Find the sentence in 'ans_sentences' with the smallest Levenshtein distance to 'x'
        best_match = None
        min_dist = float("inf")

        for candidate in ans_sentences:
            dist = levenshtein_distance(x, candidate)
            # Find the closest match (simulating cutoff=0.0 logic)
            if dist < min_dist:
                min_dist = dist
                best_match = candidate

        if best_match:
            try:
                ans_ordering.append(ans_sentences.index(best_match))
            except ValueError:
                pass

    n_sentences_gt = len(gt_sentences)
    if n_sentences_gt == 0:
        return 0.0

    # Calculate edit distance between the expected index order (0, 1, 2...) and actual found order
    raw_distance = levenshtein_distance(list(range(len(gt_sentences))), ans_ordering)
    score = 1 - (raw_distance / n_sentences_gt)

    if debug and score < 1:
        print(f"[DEBUG-PLOT] INCORRECT Score: {score}")
        print(f"[DEBUG-PLOT] GT Sentences: {gt_sentences}")
        print(f"[DEBUG-PLOT] Ans Sentences: {ans_sentences}")

    return score

# ==========================================
# Evaluator 2: Connections Puzzle
# ==========================================


def last_boxed_only_string(string: str) -> Optional[str]:
    """Parses LaTeX style \\boxed{content}."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> Optional[str]:
    left = "\\boxed{"
    try:
        if s[: len(left)] == left and s[-1] == "}":
            return s[len(left) : -1]
        return None
    except Exception:
        return None


def group_words(words: list) -> list:
    """Groups a list of words into sets of 4."""
    groups = [set()]
    words = [w.strip().lower() for w in words]
    for word in words:
        if len(groups[-1]) == 4:
            groups.append(set())
        groups[-1].add(word)
    return groups


def connections_process_results_old(
    ground_truth: str, llm_answer: str, debug=False
) -> float:
    """Evaluator for older puzzles (looks for bold text)."""
    bold_words = re.findall(r"\*\*(.*?)\*\*", llm_answer.replace("\n", ""))

    if not bold_words:
        if debug:
            print("[DEBUG-CONN-OLD] No bold words found.")
        return 0

    bold_words = [words.split(",") for words in bold_words]
    ground_truth_groups = group_words(ground_truth.split(","))

    max_score = 0
    # Check similarity against extracted bold groups
    for output_groups in list(map(group_words, bold_words)):
        correct_groups = 0
        for ground_truth_group in ground_truth_groups:
            for output_group in output_groups:
                # Check if all words in GT group exist in Output group
                if all([word in output_group for word in ground_truth_group]):
                    correct_groups += 1
                    break

        if len(ground_truth_groups) > 0:
            max_score = max(max_score, correct_groups / len(ground_truth_groups))

    if debug and max_score < 1:
        print(f"[DEBUG-CONN-OLD] Incorrect. Score: {max_score}")
    return max_score


def connections_process_results(
    ground_truth: str, llm_answer: str, debug=False
) -> float:
    """Evaluator for newer puzzles (looks for <solution> tags or boxed text)."""

    # Try to find content inside <solution> tags
    solution_matches = re.findall(r"<solution>(.*?)<\/solution>", llm_answer)
    if not solution_matches:
        solution_matches = re.findall(
            r"<solution>(.*?)<\/solution>", llm_answer.replace("\n", "")
        )
    if not solution_matches:
        # Check for malformed closing tags scenarios
        solution_matches = re.findall(r"</solution>(.*?)<\/solution>", llm_answer)

    ground_truth_words = ground_truth.split(",")

    # Fallback to \boxed format if no xml tags found
    if len(solution_matches) == 0 and "\\boxed" in llm_answer:
        boxed = last_boxed_only_string(llm_answer)
        if boxed:
            no_box = remove_boxed(boxed)
            if no_box:
                # Clean up latex syntax
                clean_text = (
                    no_box.replace("\\text{", "").replace("}", "").replace("\\", "")
                )
                solution_matches = [clean_text]

    # Clean newlines from matches
    solution_matches = [match.replace("\n", "") for match in solution_matches]

    if len(solution_matches) == 0:
        if debug:
            print("[DEBUG-CONN] No solution text found.")
        return 0

    # Handle multiple matches or single match
    if len(solution_matches) > 1:
        if debug:
            print("[DEBUG-CONN] Multiple solution texts found. Combining from last.")
        all_words = []
        num_words = len(ground_truth_words)
        for match in solution_matches:
            all_words.extend(match.split(","))
        solution_words = all_words[-num_words:]
    else:
        solution_words = solution_matches[-1].split(",")

    # Compare Groups
    llm_groups = group_words(solution_words)
    ground_truth_groups = group_words(ground_truth_words)

    correct_groups = 0
    for llm_group in llm_groups:
        if llm_group in ground_truth_groups:
            correct_groups += 1

    if len(ground_truth_groups) == 0:
        return 0.0

    score = correct_groups / len(ground_truth_groups)

    if debug and score < 1:
        print(f"[DEBUG-CONN] Incorrect. Score: {score}")
        print(f"GT Groups: {sorted([sorted(list(g)) for g in ground_truth_groups])}")
        print(f"LLM Groups: {sorted([sorted(list(g)) for g in llm_groups])}")

    return score


def get_connections_puzzle_evaluator(release_date: str):
    """Factory function to get the correct evaluator based on date."""
    if release_date < "2024-11-25":
        return connections_process_results_old
    return connections_process_results


# ==========================================
# Evaluator 3: Typos / Exact Match
# ==========================================


def typos_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    """
    Checks if the ground truth is present in the LLM answer.
    """
    parsed_answer = None

    # Priority 1: Extract from <solution> tags
    solution_matches = re.findall(r"<solution>(.*?)</solution>", llm_answer)
    if len(solution_matches) > 0:
        parsed_answer = solution_matches[-1]
    else:
        # Priority 2: Clean tags and use separator pattern extraction
        parsed_answer = llm_answer.replace("<solution>", "").replace("</solution>", "")
        parsed_answer = extract_answer(parsed_answer)

    # Clean up whitespace/newlines
    parsed_answer = " ".join(list(filter(None, parsed_answer.strip().split("\n"))))

    # Core Logic: Check for substring inclusion
    if int(ground_truth in parsed_answer):
        return 1

    # Simplified Debug Logic (No difflib)
    score = 0
    if debug and score == 0:
        print("[DEBUG-TYPO] INCORRECT")
        print(f"GT  : {ground_truth}")
        print(f"PRED: {parsed_answer}")

    return score


# ==========================================
# Main Score Function
# ==========================================


def language_judge(ground_truth: str, llm_answer: str, task_type: str, release_date: str = "2025-01-01") -> float:
    """
    Routes to the appropriate evaluator based on task_type.

    Args:
        ground_truth: Expected answer/content
        llm_answer: Model's response
        task_type: One of "typos", "connections", "unscrambling"
        release_date: Date string for connections puzzle version selection

    Returns:
        float: Score between 0.0 and 1.0
    """
    if task_type == "typos":
        return typos_process_results(ground_truth, llm_answer)
    elif task_type == "connections":
        evaluator = get_connections_puzzle_evaluator(release_date)
        return evaluator(ground_truth, llm_answer)
    elif task_type == "unscrambling":
        return plot_unscrambling_process_results(ground_truth, llm_answer)
    else:
        warn_msg = f"[mjnemogym.qydomain] Unknown task_type: {task_type}"
        _logger.warning(warn_msg)
        print(warn_msg, file=sys.stderr, flush=True)
        return 0.0


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Standalone QY domain scoring function for verl reward manager.

    Args:
        model_output: Model-generated answer text
        extra_info: Dictionary from parquet containing:
            - ground_truth or expected_answer: Ground truth answer
            - task_type: One of "typos", "connections", "unscrambling"
            - release_date (optional): For connections puzzle version selection

    Returns:
        float: Score between 0.0 and 1.0
    """
    try:
        # Extract task_type
        task_type = extra_info.get("task_type", "")
        if not task_type:
            warn_msg = f"[mjnemogym.qydomain] score_fn: task_type is empty or missing. extra_info keys: {list(extra_info.keys())}"
            _logger.warning(warn_msg)
            print(warn_msg, file=sys.stderr, flush=True)
            return 0.0

        # Extract ground_truth (try multiple keys for compatibility)
        ground_truth = extra_info.get("ground_truth", "") or extra_info.get("expected_answer", "") or extra_info.get("label", "")
        if not ground_truth:
            warn_msg = f"[mjnemogym.qydomain] score_fn: ground_truth is empty or missing. extra_info keys: {list(extra_info.keys())}"
            _logger.warning(warn_msg)
            print(warn_msg, file=sys.stderr, flush=True)
            return 0.0

        # Extract optional release_date for connections
        release_date = extra_info.get("release_date", "2025-01-01")

        # Compute score
        reward = language_judge(ground_truth, model_output, task_type, release_date)
        return float(reward)

    except Exception as e:
        error_msg = f"[mjnemogym.qydomain] score_fn failed: {type(e).__name__}: {e}"
        _logger.error(error_msg, exc_info=True)
        print(error_msg, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 0.0
