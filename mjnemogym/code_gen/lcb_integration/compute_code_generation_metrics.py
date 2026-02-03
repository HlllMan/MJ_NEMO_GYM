# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

import json
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import ray
from tqdm import tqdm

from mjnemogym.code_gen.lcb_integration.pass_k_utils import compute_metrics_from_results
from mjnemogym.code_gen.lcb_integration.testing_util import run_test


sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _temp_run(in_outs, generation, debug, result, metadata_list, timeout):
    import os
    print(f"[DEBUG _temp_run] PID={os.getpid()} Starting run_test", flush=True)
    try:
        res, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
        print(f"[DEBUG _temp_run] PID={os.getpid()} run_test returned res={res}", flush=True)
        result.append(res)
        metadata_list.append(metadata)
        print(f"[DEBUG _temp_run] PID={os.getpid()} Appended to result list, len={len(result)}", flush=True)
    except Exception as e:
        print(f"[DEBUG _temp_run] PID={os.getpid()} EXCEPTION in _temp_run: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


# Using SPREAD scheduling so that Ray assigns tasks to as many distinct nodes as possible.
@ray.remote(scheduling_strategy="SPREAD")
def check_correctness_remote(sample, generation, timeout, debug=True):
    """Ray wrapper of check_correctness for remote execution."""
    return check_correctness(sample, generation, timeout, debug)


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    import os as _os
    _pid = _os.getpid()
    _start_method = multiprocessing.get_start_method()
    print(f"[DEBUG check_correctness] PID={_pid} start_method={_start_method}", flush=True)

    # Parse JSON once at the beginning to avoid multiple parsing
    try:
        in_outs = json.loads(sample["input_output"])
    except (ValueError, MemoryError) as e:
        print(f"[DEBUG check_correctness] PID={_pid} JSON parse failed: {e}", flush=True)
        return [-1], None

    num_tests = len(in_outs.get("inputs", []))
    print(f"[DEBUG check_correctness] PID={_pid} num_tests={num_tests}", flush=True)

    if _os.environ.get("CODEGEN_NO_SUBPROCESS", "0") == "1":
        print(f"[DEBUG check_correctness] PID={_pid} NO_SUBPROCESS mode", flush=True)
        try:
            result, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
            print(f"[DEBUG check_correctness] PID={_pid} NO_SUBPROCESS result={result}", flush=True)
            return result, metadata
        except Exception as e:
            print(f"[DEBUG check_correctness] PID={_pid} NO_SUBPROCESS exception: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return [-1 for _ in range(len(in_outs["inputs"]))], None

    # Original subprocess mode (for isolation/extra timeout safety)
    print(f"[DEBUG check_correctness] PID={_pid} SUBPROCESS mode - creating Manager", flush=True)
    try:
        manager = multiprocessing.Manager()
        result = manager.list()
        metadata_list = manager.list()
        print(f"[DEBUG check_correctness] PID={_pid} Manager created, creating Process", flush=True)
    except Exception as e:
        print(f"[DEBUG check_correctness] PID={_pid} Manager creation FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return [-1 for _ in range(num_tests)], None

    p = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, debug, result, metadata_list, timeout),
    )

    global_timeout = (timeout + 1) * len(in_outs["inputs"]) + 5
    print(f"[DEBUG check_correctness] PID={_pid} Starting subprocess, global_timeout={global_timeout}", flush=True)

    p.start()
    print(f"[DEBUG check_correctness] PID={_pid} Subprocess started as PID={p.pid}", flush=True)

    p.join(timeout=global_timeout)

    if p.is_alive():
        print(f"[DEBUG check_correctness] PID={_pid} Subprocess TIMEOUT - killing", flush=True)
        p.kill()
        p.join()  # Wait for kill to complete

    print(f"[DEBUG check_correctness] PID={_pid} Subprocess exitcode={p.exitcode}, result_len={len(result)}", flush=True)

    if not result:
        # consider that all tests failed
        print(f"[DEBUG check_correctness] PID={_pid} EMPTY RESULT - subprocess likely crashed or timed out", flush=True)
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        metadata_list = [None]
        if debug:
            print("global timeout")
    else:
        print(f"[DEBUG check_correctness] PID={_pid} Got result: {result[0]}", flush=True)

    return result[0], metadata_list[0]


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            assert isinstance(curr_res, list), curr_res
            assert isinstance(curr_metadata, dict), curr_metadata
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 30 + "\n\n")
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
    """

    # generations are code generations in the same order of the dataset

    inputs = [
        [(generations_list[index], samples_list[index], debug, timeout), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(max_workers=1 if debug else num_process_evaluate) as executor:
            futures = {executor.submit(evaluate_generations_by_problem, arg): index for arg, index in inputs}

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)} {results=}"
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    print(f"Evaluating {len(samples_linear)}...")

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(generations_list[0]), f"{len(final_metadata[i])=}"

    return [metrics, results, final_metadata]


if __name__ == "__main__":
    # print(
    #     check_correctness(
    #         {
    #             "input_output": json.dumps(
    #                 {
    #                     "inputs": [
    #                         json.dumps([1] * 100000)
    #                         + "\n"
    #                         + json.dumps([100000, -100000] * (100000 // 2))
    #                     ],
    #                     "outputs": [json.dumps([100000, 0] * (100000 // 2))],
    #                     "fn_name": "mostFrequentIDs",
    #                 }
    #             )
    #         },
    #         "class Solution:\n    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:\n        from collections import defaultdict\n        \n        # Count of each ID\n        count = defaultdict(int)\n        # How many IDs exist for a given frequency\n        freq_of_count = defaultdict(int)\n        \n        max_freq = 0\n        ans = []\n        \n        for i in range(len(nums)):\n            x = nums[i]\n            change = freq[i]\n            \n            old_freq = count[x]\n            new_freq = old_freq + change\n            \n            # If there was an old frequency, decrease its usage\n            if old_freq > 0:\n                freq_of_count[old_freq] -= 1\n                if freq_of_count[old_freq] == 0:\n                    del freq_of_count[old_freq]\n            \n            # Update with the new frequency\n            count[x] = new_freq\n            freq_of_count[new_freq] += 1\n            \n            # Update max_freq if needed\n            if new_freq > max_freq:\n                max_freq = new_freq\n            \n            # If the collection at max_freq is empty, reduce max_freq until we find a non-empty bin\n            while max_freq > 0 and max_freq not in freq_of_count:\n                max_freq -= 1\n            \n            # If the collection is empty, max_freq will be 0\n            ans.append(max_freq)\n        \n        return ans",
    #         6,
    #         debug=True,
    #     )
    # )

    print(
        check_correctness(
            {
                "input_output": json.dumps(
                    {
                        "inputs": ")))))",
                        "outputs": "0",
                    },
                )
            },
            "\nMOD = 998244353\n\nS = input().strip()\nn = len(S)\n\nif n % 2 != 0:\n    print(0)\n    exit()\n\n# Initialize DP table\ndp = [[0] * (n + 2) for _ in range(n + 1)]\ndp[0][0] = 1\n\nfor i in range(1, n + 1):\n    c = S[i-1]\n    for b in range(n + 1):\n        if dp[i-1][b] == 0:\n            continue\n        if c == '(':\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        elif c == ')':\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        else:  # '?'\n            # Replace with '('\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n            # Replace with ')'\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n\nprint(dp[n][0] % MOD)\n",
            6,
            debug=True,
        )
    )
