# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script mimicking verl's parallel reward computation
# Tests verl_compute_score via ProcessPoolExecutor like dapo.py might use

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

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


def evaluate_verl_style(args) -> Tuple[int, float, str]:
    """
    Evaluate a single item using verl_compute_score style.
    This mimics how verl's dapo.py compute_score callback works.
    """
    data_source, solution_str, ground_truth, extra_info, idx = args

    # Import here to simulate fresh import in each worker
    from mjnemogym import verl_compute_score

    try:
        reward = verl_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        status = 'OK'
    except Exception as e:
        import traceback
        reward = -1.0
        status = f'ERROR: {type(e).__name__}: {e}'
        print(f"[ERROR idx={idx}] {status}")
        traceback.print_exc()

    return (idx, reward, status)


def prepare_verl_args(data: List[dict]) -> List[tuple]:
    """Convert test data to verl_compute_score arguments."""
    args_list = []
    for item in data:
        data_source = item.get('data_source', 'nemogym_code')
        solution_str = item.get('response', '')
        ground_truth = item.get('reward_model', {}).get('ground_truth', '')
        extra_info = item.get('extra_info', {})
        idx = extra_info.get('index', -1)

        args_list.append((data_source, solution_str, ground_truth, extra_info, idx))

    return args_list


def run_single_process(args_list: List[tuple]) -> List[Tuple[int, float, str]]:
    """Run evaluation sequentially in single process."""
    results = []
    for args in args_list:
        result = evaluate_verl_style(args)
        results.append(result)
        print(f"[Single] idx={result[0]}, reward={result[1]}")
    return results


def run_parallel_spawn(args_list: List[tuple], num_workers: int = 4) -> List[Tuple[int, float, str]]:
    """Run evaluation in parallel using spawn (macOS default)."""
    results = []

    ctx = multiprocessing.get_context('spawn')
    print(f"[Parallel-Spawn] Using {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(evaluate_verl_style, args): args for args in args_list}

        for future in as_completed(futures):
            try:
                result = future.result(timeout=600)
                results.append(result)
                print(f"[Parallel-Spawn] idx={result[0]}, reward={result[1]}")
            except Exception as e:
                args = futures[future]
                idx = args[4]
                print(f"[Parallel-Spawn] idx={idx} FUTURE ERROR: {type(e).__name__}: {e}")
                results.append((idx, -1.0, f'FUTURE_ERROR: {e}'))

    return results


def run_parallel_fork(args_list: List[tuple], num_workers: int = 4) -> List[Tuple[int, float, str]]:
    """Run evaluation in parallel using fork (Linux default, can be used on macOS)."""
    results = []

    try:
        ctx = multiprocessing.get_context('fork')
        print(f"[Parallel-Fork] Using {num_workers} workers")
    except ValueError:
        print("[Parallel-Fork] fork not available, skipping")
        return results

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(evaluate_verl_style, args): args for args in args_list}

        for future in as_completed(futures):
            try:
                result = future.result(timeout=600)
                results.append(result)
                print(f"[Parallel-Fork] idx={result[0]}, reward={result[1]}")
            except Exception as e:
                args = futures[future]
                idx = args[4]
                print(f"[Parallel-Fork] idx={idx} FUTURE ERROR: {type(e).__name__}: {e}")
                results.append((idx, -1.0, f'FUTURE_ERROR: {e}'))

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
    single_1_parallel_0 = []
    for idx in all_indices:
        r1 = dict1.get(idx)
        r2 = dict2.get(idx)

        if r1 is None or r2 is None:
            continue

        if r1[1] != r2[1]:
            mismatches.append((idx, r1[1], r2[1]))
            if r1[1] == 1.0 and r2[1] == 0.0:
                single_1_parallel_0.append(idx)

    if mismatches:
        print(f"MISMATCHES: {len(mismatches)}")
        for idx, v1, v2 in mismatches:
            marker = " <-- SINGLE=1, PARALLEL=0" if v1 == 1.0 and v2 == 0.0 else ""
            print(f"  idx={idx}: {name1}={v1}, {name2}={v2}{marker}")
    else:
        print("No mismatches!")

    if single_1_parallel_0:
        print(f"\n*** CRITICAL: {len(single_1_parallel_0)} cases where single=1 but parallel=0 ***")

    sum1 = sum(r[1] for r in results1 if r[1] >= 0)
    sum2 = sum(r[1] for r in results2 if r[1] >= 0)
    print(f"\n{name1} total: {sum1}/{len(results1)}")
    print(f"{name2} total: {sum2}/{len(results2)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='tests/code_random_128.jsonl')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mode', choices=['spawn', 'fork', 'both'], default='both')
    parser.add_argument('--no-subprocess', action='store_true',
                        help='Set CODEGEN_NO_SUBPROCESS=1')

    args = parser.parse_args()

    if args.no_subprocess:
        os.environ['CODEGEN_NO_SUBPROCESS'] = '1'
        print(">>> CODEGEN_NO_SUBPROCESS=1")

    print(f"Loading data from {args.data} (limit={args.limit})")
    data = load_test_data(args.data, limit=args.limit)
    print(f"Loaded {len(data)} samples")

    # Convert to verl-style args
    args_list = prepare_verl_args(data)

    print(f"\nPython: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    # Single process baseline
    print("="*80)
    print("SINGLE PROCESS EVALUATION (baseline)")
    print("="*80)
    start = time.time()
    single_results = run_single_process(args_list)
    print(f"Single completed in {time.time() - start:.2f}s")

    if args.mode in ['spawn', 'both']:
        print("\n" + "="*80)
        print("PARALLEL SPAWN MODE")
        print("="*80)
        start = time.time()
        spawn_results = run_parallel_spawn(args_list, args.workers)
        print(f"Spawn completed in {time.time() - start:.2f}s")
        compare_results("Single", single_results, "Spawn", spawn_results)

    if args.mode in ['fork', 'both']:
        print("\n" + "="*80)
        print("PARALLEL FORK MODE")
        print("="*80)
        start = time.time()
        fork_results = run_parallel_fork(args_list, args.workers)
        if fork_results:
            print(f"Fork completed in {time.time() - start:.2f}s")
            compare_results("Single", single_results, "Fork", fork_results)


if __name__ == '__main__':
    main()
