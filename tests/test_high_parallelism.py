# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stress test for high parallelism - mimics 128+ worker scenario

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_test_data(jsonl_path: str, limit: int = None) -> List[dict]:
    data = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data


def evaluate_single(item: dict) -> Tuple[int, float, str]:
    """Evaluate with fresh import to simulate worker isolation."""
    # Fresh import in each worker
    from mjnemogym.code_gen.score import score_fn

    idx = item.get('extra_info', {}).get('index', -1)
    response = item.get('response', '')
    extra_info = item.get('extra_info', {})

    try:
        reward = score_fn(response, extra_info)
        return (idx, reward, 'OK')
    except Exception as e:
        return (idx, -1.0, f'ERROR: {e}')


def run_high_parallelism_test(data: List[dict], num_workers: int,
                               batch_size: int = None, timeout_per_task: int = 300):
    """
    Run with high parallelism, optionally in batches.

    This mimics scenarios where many workers are spawned simultaneously,
    each creating their own Manager and subprocess.
    """
    if batch_size is None:
        batch_size = len(data)

    ctx = multiprocessing.get_context('spawn')
    all_results = []

    print(f"Running with {num_workers} workers, batch_size={batch_size}")
    print(f"Total samples: {len(data)}")

    # Process in batches to simulate continuous load
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch = data[batch_start:batch_end]

        print(f"\nBatch {batch_start//batch_size + 1}: samples {batch_start}-{batch_end-1}")

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(evaluate_single, item): item for item in batch}

            completed = 0
            errors = 0
            for future in as_completed(futures, timeout=timeout_per_task * len(batch)):
                try:
                    result = future.result(timeout=timeout_per_task)
                    all_results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  Completed {completed}/{len(batch)}")
                except TimeoutError:
                    item = futures[future]
                    idx = item.get('extra_info', {}).get('index', -1)
                    print(f"  TIMEOUT: idx={idx}")
                    all_results.append((idx, -1.0, 'TIMEOUT'))
                    errors += 1
                except Exception as e:
                    item = futures[future]
                    idx = item.get('extra_info', {}).get('index', -1)
                    print(f"  ERROR idx={idx}: {e}")
                    all_results.append((idx, -1.0, str(e)))
                    errors += 1

        print(f"  Batch done: {completed} completed, {errors} errors")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='tests/code_random_128.jsonl')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Process in batches (default: all at once)')
    parser.add_argument('--compare', action='store_true',
                        help='Also run single-process for comparison')

    args = parser.parse_args()

    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"CPUs: {multiprocessing.cpu_count()}")
    print(f"Workers: {args.workers}")
    print()

    data = load_test_data(args.data, limit=args.limit)
    print(f"Loaded {len(data)} samples\n")

    # Run single process baseline if requested
    if args.compare:
        print("=" * 80)
        print("SINGLE PROCESS BASELINE")
        print("=" * 80)
        single_results = []
        for item in data:
            result = evaluate_single(item)
            single_results.append(result)
        single_rewards = {r[0]: r[1] for r in single_results}
        single_total = sum(r[1] for r in single_results if r[1] >= 0)
        print(f"Single total reward: {single_total}/{len(single_results)}")
        print()

    # Run high parallelism test
    print("=" * 80)
    print(f"HIGH PARALLELISM TEST ({args.workers} workers)")
    print("=" * 80)

    start = time.time()
    parallel_results = run_high_parallelism_test(
        data, args.workers, args.batch_size
    )
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.2f}s")

    parallel_total = sum(r[1] for r in parallel_results if r[1] >= 0)
    errors = sum(1 for r in parallel_results if r[1] < 0)
    print(f"Parallel total reward: {parallel_total}/{len(parallel_results)}")
    print(f"Errors: {errors}")

    # Compare if baseline was run
    if args.compare:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)

        parallel_rewards = {r[0]: r[1] for r in parallel_results}

        mismatches = []
        single_1_parallel_0 = []
        for idx in single_rewards:
            s = single_rewards.get(idx, -999)
            p = parallel_rewards.get(idx, -999)
            if s != p:
                mismatches.append((idx, s, p))
                if s == 1.0 and p == 0.0:
                    single_1_parallel_0.append(idx)

        if mismatches:
            print(f"\nMISMATCHES FOUND: {len(mismatches)}")
            for idx, s, p in mismatches[:20]:  # Show first 20
                marker = " *** SINGLE=1, PARALLEL=0 ***" if s == 1.0 and p == 0.0 else ""
                print(f"  idx={idx}: single={s}, parallel={p}{marker}")

            if single_1_parallel_0:
                print(f"\n*** CRITICAL: {len(single_1_parallel_0)} cases where single=1 but parallel=0 ***")
                print(f"Indices: {single_1_parallel_0}")
        else:
            print("\nNo mismatches!")

        print(f"\nSingle total: {single_total}")
        print(f"Parallel total: {parallel_total}")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
