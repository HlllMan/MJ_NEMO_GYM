# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test to specifically check nested multiprocessing issues
# This tests what happens when ProcessPoolExecutor workers spawn additional processes

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_manager_in_worker():
    """Test if Manager works correctly inside a ProcessPoolExecutor worker."""
    import multiprocessing

    pid = os.getpid()
    print(f"[Worker PID={pid}] Testing Manager creation")

    try:
        manager = multiprocessing.Manager()
        result_list = manager.list()
        result_list.append("test_value")
        print(f"[Worker PID={pid}] Manager list works, len={len(result_list)}, value={result_list[0]}")
        return True, "Manager OK"
    except Exception as e:
        print(f"[Worker PID={pid}] Manager FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_nested_process_in_worker():
    """Test if nested Process works correctly inside a ProcessPoolExecutor worker."""
    import multiprocessing

    pid = os.getpid()
    print(f"[Worker PID={pid}] Testing nested Process creation")

    def nested_target(result_list):
        nested_pid = os.getpid()
        print(f"[Nested PID={nested_pid}] Running nested process")
        result_list.append("nested_result")
        print(f"[Nested PID={nested_pid}] Appended to result list")

    try:
        manager = multiprocessing.Manager()
        result_list = manager.list()

        p = multiprocessing.Process(target=nested_target, args=(result_list,))
        p.start()
        p.join(timeout=10)

        if p.is_alive():
            p.kill()
            return False, "Nested process timeout"

        print(f"[Worker PID={pid}] Nested process exitcode={p.exitcode}, result_len={len(result_list)}")

        if len(result_list) == 0:
            return False, f"Result list empty despite exitcode={p.exitcode}"

        return True, f"Nested process OK, result={result_list[0]}"
    except Exception as e:
        print(f"[Worker PID={pid}] Nested process FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_signal_handler_in_worker():
    """Test if signal handlers work in ProcessPoolExecutor workers."""
    import signal
    import threading

    pid = os.getpid()
    tid = threading.current_thread().name
    print(f"[Worker PID={pid} TID={tid}] Testing signal handler")

    alarm_triggered = [False]

    def alarm_handler(signum, frame):
        nonlocal alarm_triggered
        alarm_triggered[0] = True
        print(f"[Worker PID={pid}] SIGALRM triggered!")

    try:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(1)  # 1 second alarm
        time.sleep(2)  # Wait for alarm
        signal.alarm(0)  # Cancel

        if alarm_triggered[0]:
            return True, "Signal handler works"
        else:
            return False, "Signal handler did NOT trigger"
    except Exception as e:
        print(f"[Worker PID={pid}] Signal handler FAILED: {type(e).__name__}: {e}")
        return False, str(e)


def test_check_correctness_in_worker(item: dict):
    """Test check_correctness specifically inside a worker."""
    from mjnemogym.code_gen.score import score_fn

    pid = os.getpid()
    idx = item.get('extra_info', {}).get('index', -1)
    print(f"[Worker PID={pid}] Testing check_correctness for idx={idx}")

    try:
        response = item.get('response', '')
        extra_info = item.get('extra_info', {})
        reward = score_fn(response, extra_info)
        print(f"[Worker PID={pid}] idx={idx} reward={reward}")
        return (idx, reward, "OK")
    except Exception as e:
        print(f"[Worker PID={pid}] idx={idx} FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return (idx, -1.0, str(e))


def run_nested_mp_tests(num_workers: int = 2, start_method: str = 'spawn'):
    """Run nested multiprocessing tests."""
    ctx = multiprocessing.get_context(start_method)
    print(f"\n{'='*80}")
    print(f"TESTING NESTED MULTIPROCESSING ({start_method} mode, {num_workers} workers)")
    print(f"{'='*80}")

    tests = [
        ("Manager in Worker", test_manager_in_worker),
        ("Nested Process in Worker", test_nested_process_in_worker),
        ("Signal Handler in Worker", test_signal_handler_in_worker),
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        for test_name, test_fn in tests:
            print(f"\n--- Test: {test_name} ---")
            future = executor.submit(test_fn)
            try:
                success, msg = future.result(timeout=30)
                results[test_name] = (success, msg)
                print(f"Result: {'PASS' if success else 'FAIL'} - {msg}")
            except Exception as e:
                results[test_name] = (False, str(e))
                print(f"Result: FAIL - {e}")

    return results


def run_check_correctness_tests(data: List[dict], num_workers: int = 2, start_method: str = 'spawn'):
    """Run check_correctness inside workers."""
    ctx = multiprocessing.get_context(start_method)
    print(f"\n{'='*80}")
    print(f"TESTING check_correctness ({start_method} mode, {num_workers} workers)")
    print(f"{'='*80}")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(test_check_correctness_in_worker, item): item for item in data}

        for future in as_completed(futures):
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                item = futures[future]
                idx = item.get('extra_info', {}).get('index', -1)
                print(f"FUTURE ERROR idx={idx}: {e}")
                results.append((idx, -1.0, str(e)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='tests/code_random_128.jsonl')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--method', choices=['spawn', 'fork'], default='spawn')

    args = parser.parse_args()

    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Default start method: {multiprocessing.get_start_method()}")

    # Test basic nested multiprocessing
    nested_results = run_nested_mp_tests(args.workers, args.method)

    # Load data for check_correctness test
    data = []
    with open(args.data, 'r') as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            data.append(json.loads(line))

    print(f"\nLoaded {len(data)} samples")

    # Test check_correctness inside workers
    cc_results = run_check_correctness_tests(data, args.workers, args.method)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print("\nNested MP Tests:")
    for test_name, (success, msg) in nested_results.items():
        print(f"  {test_name}: {'PASS' if success else 'FAIL'} - {msg}")

    print("\ncheck_correctness Tests:")
    total_reward = sum(r[1] for r in cc_results if r[1] >= 0)
    print(f"  Total reward: {total_reward}/{len(cc_results)}")

    # Check for any failures
    failures = [r for r in cc_results if r[1] < 0]
    if failures:
        print(f"  Failures: {len(failures)}")
        for idx, reward, status in failures:
            print(f"    idx={idx}: {status}")


if __name__ == '__main__':
    main()
