# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script to replicate single=1, parallel=0 issue
# Usage:
#   python tests/test_parallel_eval.py --mode single    # Single process evaluation
#   python tests/test_parallel_eval.py --mode parallel  # Parallel evaluation with ProcessPoolExecutor
#   python tests/test_parallel_eval.py --mode both      # Run both and compare

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjnemogym.code_gen.score import score_fn


def load_test_data(jsonl_path: str, limit: int = None) -> List[dict]:
    """Load test data from JSONL file."""
    data = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data


def evaluate_single(item: dict) -> Tuple[int, float, str]:
    """Evaluate a single item. Returns (index, reward, status)."""
    idx = item.get('extra_info', {}).get('index', -1)
    response = item.get('response', '')
    extra_info = item.get('extra_info', {})

    try:
        reward = score_fn(response, extra_info)
        status = 'OK'
    except Exception as e:
        reward = -1.0
        status = f'ERROR: {type(e).__name__}: {e}'

    return (idx, reward, status)


def run_single_process(data: List[dict]) -> List[Tuple[int, float, str]]:
    """Run evaluation sequentially in single process."""
    results = []
    for item in data:
        result = evaluate_single(item)
        results.append(result)
        print(f"[Single] idx={result[0]}, reward={result[1]}, status={result[2]}")
    return results


def run_parallel_process(data: List[dict], num_workers: int = 4) -> List[Tuple[int, float, str]]:
    """Run evaluation in parallel using ProcessPoolExecutor."""
    results = []

    # Force spawn to match what happens in real parallel evaluation
    ctx = multiprocessing.get_context('spawn')

    print(f"[Parallel] Using {num_workers} workers with spawn context")
    print(f"[Parallel] Current start method: {multiprocessing.get_start_method()}")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(evaluate_single, item): item for item in data}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"[Parallel] idx={result[0]}, reward={result[1]}, status={result[2]}")
            except Exception as e:
                item = futures[future]
                idx = item.get('extra_info', {}).get('index', -1)
                print(f"[Parallel] idx={idx} FUTURE ERROR: {type(e).__name__}: {e}")
                results.append((idx, -1.0, f'FUTURE_ERROR: {e}'))

    return results


def compare_results(single_results: List[Tuple], parallel_results: List[Tuple]):
    """Compare single vs parallel results."""
    # Sort by index
    single_dict = {r[0]: r for r in single_results}
    parallel_dict = {r[0]: r for r in parallel_results}

    all_indices = sorted(set(single_dict.keys()) | set(parallel_dict.keys()))

    print("\n" + "=" * 80)
    print("COMPARISON: Single vs Parallel")
    print("=" * 80)

    mismatches = []
    for idx in all_indices:
        s = single_dict.get(idx)
        p = parallel_dict.get(idx)

        if s is None:
            print(f"idx={idx}: MISSING in single")
            continue
        if p is None:
            print(f"idx={idx}: MISSING in parallel")
            continue

        s_reward = s[1]
        p_reward = p[1]

        if s_reward != p_reward:
            mismatches.append((idx, s_reward, p_reward))
            print(f"idx={idx}: MISMATCH - single={s_reward}, parallel={p_reward}")

    print("\n" + "=" * 80)
    print(f"Total samples: {len(all_indices)}")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        print("\nMismatch details (single=1 but parallel=0):")
        for idx, s_reward, p_reward in mismatches:
            if s_reward == 1.0 and p_reward == 0.0:
                print(f"  idx={idx}: single=1.0, parallel=0.0")

    # Summary statistics
    single_sum = sum(r[1] for r in single_results if r[1] >= 0)
    parallel_sum = sum(r[1] for r in parallel_results if r[1] >= 0)
    print(f"\nSingle total reward: {single_sum}")
    print(f"Parallel total reward: {parallel_sum}")


def main():
    parser = argparse.ArgumentParser(description='Test parallel vs single process code evaluation')
    parser.add_argument('--mode', choices=['single', 'parallel', 'both'], default='both',
                        help='Evaluation mode')
    parser.add_argument('--data', default='tests/code_random_128.jsonl',
                        help='Path to test data JSONL file')
    parser.add_argument('--limit', type=int, default=10,
                        help='Limit number of samples to evaluate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--no-subprocess', action='store_true',
                        help='Set CODEGEN_NO_SUBPROCESS=1 to skip subprocess isolation')

    args = parser.parse_args()

    if args.no_subprocess:
        os.environ['CODEGEN_NO_SUBPROCESS'] = '1'
        print(">>> CODEGEN_NO_SUBPROCESS=1 (subprocess isolation disabled)")

    print(f"Loading data from {args.data} (limit={args.limit})")
    data = load_test_data(args.data, limit=args.limit)
    print(f"Loaded {len(data)} samples")

    print(f"\nPython version: {sys.version}")
    print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")
    print(f"Platform: {sys.platform}")
    print()

    if args.mode == 'single':
        print("=" * 80)
        print("SINGLE PROCESS EVALUATION")
        print("=" * 80)
        start = time.time()
        results = run_single_process(data)
        elapsed = time.time() - start
        print(f"\nCompleted in {elapsed:.2f}s")
        total_reward = sum(r[1] for r in results if r[1] >= 0)
        print(f"Total reward: {total_reward}/{len(results)}")

    elif args.mode == 'parallel':
        print("=" * 80)
        print("PARALLEL PROCESS EVALUATION")
        print("=" * 80)
        start = time.time()
        results = run_parallel_process(data, num_workers=args.workers)
        elapsed = time.time() - start
        print(f"\nCompleted in {elapsed:.2f}s")
        total_reward = sum(r[1] for r in results if r[1] >= 0)
        print(f"Total reward: {total_reward}/{len(results)}")

    elif args.mode == 'both':
        print("=" * 80)
        print("SINGLE PROCESS EVALUATION")
        print("=" * 80)
        start = time.time()
        single_results = run_single_process(data)
        elapsed = time.time() - start
        print(f"\nSingle completed in {elapsed:.2f}s")

        print("\n" + "=" * 80)
        print("PARALLEL PROCESS EVALUATION")
        print("=" * 80)
        start = time.time()
        parallel_results = run_parallel_process(data, num_workers=args.workers)
        elapsed = time.time() - start
        print(f"\nParallel completed in {elapsed:.2f}s")

        compare_results(single_results, parallel_results)


if __name__ == '__main__':
    # On macOS, spawn is the default since Python 3.8
    # Explicitly set it to ensure consistent behavior
    multiprocessing.set_start_method('spawn', force=True)
    main()
