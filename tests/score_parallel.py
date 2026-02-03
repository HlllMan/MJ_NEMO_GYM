#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Efficient parallel scoring script for all NemoGym domains
# Supports mixed data sources: nemogym_math, nemogym_code, nemogym_mcqa,
#                              nemogym_if, nemogym_structured, nemogym_workplace
#
# Usage:
#   python tests/score_parallel.py --input data.jsonl --output scored.jsonl --workers 200

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

from tqdm import tqdm

from mjnemogym import score_fn_dict


def score_single(args: Tuple[int, dict]) -> Tuple[int, dict, float]:
    """
    Score a single item based on its data_source.
    Returns (line_index, original_data, reward).
    """
    line_idx, item = args

    data_source = item.get('data_source', '')
    response = item.get('response', '')
    extra_info = item.get('extra_info', {})

    score_fn = score_fn_dict[data_source]
    reward = score_fn(response, extra_info)

    return (line_idx, item, reward)


def main():
    parser = argparse.ArgumentParser(description='Parallel scoring for all NemoGym domains')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of workers (default: CPU count - 16)')

    args = parser.parse_args()

    # Determine worker count
    cpu_count = multiprocessing.cpu_count()
    if args.workers is None:
        args.workers = max(1, cpu_count - 16)

    print(f"=" * 60)
    print(f"Parallel NemoGym Scoring (All Domains)")
    print(f"=" * 60)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"CPUs:    {cpu_count}")
    print(f"Workers: {args.workers}")
    print(f"Domains: {list(score_fn_dict.keys())}")
    print(f"=" * 60)

    # Load all data
    print(f"\nLoading data from {args.input}...")
    data = []
    domain_counts = {}
    with open(args.input, 'r') as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            data.append((line_idx, item))
            # Count domains
            ds = item.get('data_source', 'unknown')
            domain_counts[ds] = domain_counts.get(ds, 0) + 1

    total = len(data)
    print(f"Loaded {total} samples")
    print(f"Domain distribution:")
    for ds, count in sorted(domain_counts.items()):
        print(f"  {ds}: {count}")

    # Prepare output
    results = [None] * total
    total_reward = 0.0
    domain_rewards = {}

    # Use spawn context for clean process isolation
    ctx = multiprocessing.get_context('spawn')

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = {executor.submit(score_single, item): item[0] for item in data}

        with tqdm(total=total, desc="Scoring", unit="sample") as pbar:
            for future in as_completed(futures):
                try:
                    line_idx, item, reward = future.result()
                    item['reward'] = reward
                    results[line_idx] = item
                    total_reward += reward

                    # Track per-domain rewards
                    ds = item.get('data_source', 'unknown')
                    domain_rewards[ds] = domain_rewards.get(ds, 0) + reward

                except Exception as e:
                    line_idx = futures[future]
                    results[line_idx] = data[line_idx][1]
                    results[line_idx]['reward'] = 0.0

                pbar.update(1)
                pbar.set_postfix(reward=f"{total_reward:.0f}", refresh=False)

    elapsed = time.time() - start_time

    # Write results in original order
    print(f"\nWriting results to {args.output}...")
    with open(args.output, 'w') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Total samples:  {total}")
    print(f"Total reward:   {total_reward:.0f}")
    print(f"Pass rate:      {100*total_reward/total:.2f}%")
    print(f"Time:           {elapsed:.1f}s")
    print(f"Throughput:     {total/elapsed:.1f} samples/sec")
    print(f"\nPer-domain results:")
    for ds in sorted(domain_counts.keys()):
        count = domain_counts[ds]
        reward = domain_rewards.get(ds, 0)
        rate = 100 * reward / count if count > 0 else 0
        print(f"  {ds}: {reward:.0f}/{count} ({rate:.1f}%)")
    print(f"\nOutput: {args.output}")
    print(f"=" * 60)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
