# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script to test with fork mode and NO_SUBPROCESS
# This tests different multiprocessing configurations

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
    # Import here to ensure fresh import in each process
    from mjnemogym.code_gen.score import score_fn

    idx = item.get('extra_info', {}).get('index', -1)
    response = item.get('response', '')
    extra_info = item.get('extra_info', {})

    try:
        reward = score_fn(response, extra_info)
        status = 'OK'
    except Exception as e:
        import traceback
        reward = -1.0
        status = f'ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}'

    return (idx, reward, status)


def run_single_process(data: List[dict]) -> List[Tuple[int, float, str]]:
    """Run evaluation sequentially in single process."""
    results = []
    for item in data:
        result = evaluate_single(item)
        results.append(result)
        print(f"[Single] idx={result[0]}, reward={result[1]}")
    return results


def run_parallel_fork(data: List[dict], num_workers: int = 4) -> List[Tuple[int, float, str]]:
    """Run evaluation in parallel using FORK mode."""
    results = []

    # Try to use fork - this may cause issues!
    try:
        ctx = multiprocessing.get_context('fork')
        print(f"[Parallel-Fork] Using FORK context")
    except ValueError as e:
        print(f"[Parallel-Fork] FORK not available: {e}, falling back to spawn")
        ctx = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(evaluate_single, item): item for item in data}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"[Parallel-Fork] idx={result[0]}, reward={result[1]}")
            except Exception as e:
                item = futures[future]
                idx = item.get('extra_info', {}).get('index', -1)
                print(f"[Parallel-Fork] idx={idx} ERROR: {type(e).__name__}: {e}")
                results.append((idx, -1.0, f'FUTURE_ERROR: {e}'))

    return results


def run_parallel_no_subprocess(data: List[dict], num_workers: int = 4) -> List[Tuple[int, float, str]]:
    """Run evaluation with CODEGEN_NO_SUBPROCESS=1 (no subprocess isolation)."""
    # Set the env var
    os.environ['CODEGEN_NO_SUBPROCESS'] = '1'
    print(f"[No-Subprocess] CODEGEN_NO_SUBPROCESS=1 set")

    results = []
    ctx = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(evaluate_single, item): item for item in data}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"[No-Subprocess] idx={result[0]}, reward={result[1]}")
            except Exception as e:
                item = futures[future]
                idx = item.get('extra_info', {}).get('index', -1)
                print(f"[No-Subprocess] idx={idx} ERROR: {type(e).__name__}: {e}")
                results.append((idx, -1.0, f'FUTURE_ERROR: {e}'))

    # Unset
    del os.environ['CODEGEN_NO_SUBPROCESS']
    return results


def compare_results(name1: str, results1: List[Tuple], name2: str, results2: List[Tuple]):
    """Compare two result sets."""
    dict1 = {r[0]: r for r in results1}
    dict2 = {r[0]: r for r in results2}

    all_indices = sorted(set(dict1.keys()) | set(dict2.keys()))

    print(f"\n{'='*80}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}")

    mismatches = []
    for idx in all_indices:
        r1 = dict1.get(idx)
        r2 = dict2.get(idx)

        if r1 is None or r2 is None:
            continue

        if r1[1] != r2[1]:
            mismatches.append((idx, r1[1], r2[1]))

    if mismatches:
        print(f"MISMATCHES: {len(mismatches)}")
        for idx, v1, v2 in mismatches:
            print(f"  idx={idx}: {name1}={v1}, {name2}={v2}")
    else:
        print("No mismatches!")

    sum1 = sum(r[1] for r in results1 if r[1] >= 0)
    sum2 = sum(r[1] for r in results2 if r[1] >= 0)
    print(f"\n{name1} total: {sum1}")
    print(f"{name2} total: {sum2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='tests/code_random_128.jsonl')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test', choices=['fork', 'no-subprocess', 'all'], default='all')

    args = parser.parse_args()

    print(f"Loading data from {args.data} (limit={args.limit})")
    data = load_test_data(args.data, limit=args.limit)
    print(f"Loaded {len(data)} samples")
    print(f"Platform: {sys.platform}")
    print()

    # Always run single as baseline
    print("="*80)
    print("SINGLE PROCESS EVALUATION (baseline)")
    print("="*80)
    single_results = run_single_process(data)
    single_sum = sum(r[1] for r in single_results if r[1] >= 0)
    print(f"Single total: {single_sum}")

    if args.test in ['fork', 'all']:
        print("\n" + "="*80)
        print("PARALLEL FORK MODE")
        print("="*80)
        fork_results = run_parallel_fork(data, args.workers)
        compare_results("Single", single_results, "Fork", fork_results)

    if args.test in ['no-subprocess', 'all']:
        print("\n" + "="*80)
        print("PARALLEL NO-SUBPROCESS MODE")
        print("="*80)
        no_sub_results = run_parallel_no_subprocess(data, args.workers)
        compare_results("Single", single_results, "No-Subprocess", no_sub_results)


if __name__ == '__main__':
    main()
