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
import os
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

from tqdm import tqdm

from mjnemogym import score_fn_dict

# Debug logging with timestamps
DEBUG_HANG = True  # Set to True to enable hang debugging

def debug_log(msg: str):
    """Thread-safe debug logging with PID and timestamp."""
    if DEBUG_HANG:
        pid = os.getpid()
        tid = threading.current_thread().name
        ts = time.strftime("%H:%M:%S")
        print(f"[DEBUG {ts} pid={pid} {tid}] {msg}", flush=True)


def score_single(args: Tuple[int, dict]) -> Tuple[int, dict, float]:
    """
    Score a single item based on its data_source.
    Returns (line_index, original_data, reward).
    """
    line_idx, item = args

    data_source = item.get("data_source", "")
    response = item.get("response", "")
    extra_info = item.get("extra_info", {})
    idx = extra_info.get('index', line_idx)

    debug_log(f"START scoring line={line_idx} domain={data_source} idx={idx}")
    start_time = time.time()

    score_fn = score_fn_dict[data_source]
    reward = score_fn(response, extra_info)

    elapsed = time.time() - start_time
    debug_log(f"DONE scoring line={line_idx} domain={data_source} idx={idx} reward={reward} elapsed={elapsed:.2f}s")

    print(f"{data_source}_index_{extra_info['index']}_rew_{reward}", flush=True)
    return (line_idx, item, reward)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel scoring for all NemoGym domains"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of workers (default: CPU count - 16)",
    )

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
    with open(args.input, "r") as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            data.append((line_idx, item))
            # Count domains
            ds = item.get("data_source", "unknown")
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
    ctx = multiprocessing.get_context("spawn")

    start_time = time.time()

    # Track pending futures for debugging
    pending_count = total
    last_progress_time = time.time()
    HANG_TIMEOUT = 120  # Log warning if no progress for 2 minutes

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = {executor.submit(score_single, item): item[0] for item in data}
        debug_log(f"Submitted {len(futures)} tasks to executor")

        with tqdm(total=total, desc="Scoring", unit="sample") as pbar:
            for future in as_completed(futures):
                try:
                    line_idx, item, reward = future.result(timeout=300)  # 5min timeout per result
                    item["reward"] = reward
                    results[line_idx] = item
                    total_reward += reward

                    # Track per-domain rewards
                    domain_rewards[item["data_source"]] = (
                        domain_rewards.get(ds, 0) + reward
                    )

                    pending_count -= 1
                    last_progress_time = time.time()

                except TimeoutError:
                    line_idx = futures[future]
                    debug_log(f"TIMEOUT waiting for result line={line_idx}")
                    results[line_idx] = data[line_idx][1]
                    results[line_idx]["reward"] = 0.0
                    pending_count -= 1

                except Exception as e:
                    line_idx = futures[future]
                    debug_log(f"EXCEPTION for line={line_idx}: {type(e).__name__}: {e}")
                    results[line_idx] = data[line_idx][1]
                    results[line_idx]["reward"] = 0.0
                    pending_count -= 1

                pbar.update(1)
                pbar.set_postfix(reward=f"{total_reward:.0f}", pending=pending_count, refresh=False)

                # Check for potential hang
                if pending_count <= 10:
                    # Find which items are still pending
                    pending_lines = [futures[f] for f in futures if not f.done()]
                    pending_domains = []
                    for pl in pending_lines[:5]:  # Show up to 5
                        d = data[pl][1].get("data_source", "?")
                        idx = data[pl][1].get("extra_info", {}).get("index", "?")
                        pending_domains.append(f"line={pl}/domain={d}/idx={idx}")
                    debug_log(f"NEAR_END: {pending_count} items remaining: {pending_domains}")

        debug_log("Exiting executor context manager (waiting for shutdown)")

    elapsed = time.time() - start_time

    # Write results in original order
    print(f"\nWriting results to {args.output}...")
    with open(args.output, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

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


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
