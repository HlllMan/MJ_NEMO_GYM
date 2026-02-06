# mjnemogym - Offline Scoring Functions for NemoGym

Standalone reward functions for 9 scoring domains, designed for parallel offline evaluation. Extracted from NemoGym server infrastructure to run without FastAPI/Ray dependencies.

## Quick Start

```bash
# Install
pip install -e .

# Score a JSONL file (quiet mode - warnings/errors only)
python tests/score_parallel.py -i input.jsonl -o output.jsonl

# Score with debug logging (all domain entry/exit/timing)
MJNEMOGYM_DEBUG=1 python tests/score_parallel.py -i input.jsonl -o output.jsonl

# Control worker count (default: CPU_count - 16)
python tests/score_parallel.py -i input.jsonl -o output.jsonl --workers 32
```

## Input Format

Each line in the input JSONL must have:

```json
{
  "data_source": "nemogym_math",
  "response": "The answer is \\boxed{42}",
  "extra_info": {
    "index": 123,
    "expected_answer": "42"
  }
}
```

The `extra_info` fields vary by domain (see [Domain Reference](#domain-reference) below).

## Domains

| `data_source` | Domain | Scoring Method |
|---|---|---|
| `nemogym_math` | Math | 3-method fallback: QY parser -> DAPO -> MathVerify |
| `nemogym_code` | Code Generation | Unit test execution in isolated subprocess |
| `nemogym_mcqa` | Multiple Choice QA | Regex-based answer letter extraction |
| `nemogym_if` | Instruction Following | Rule-based instruction verification |
| `nemogym_structured` | Structured Outputs | JSON schema validation |
| `nemogym_workplace` | Workplace Assistant | Tool call state comparison |
| `typos` | QY Typos | Exact substring match in `<solution>` tags |
| `connections` | QY Connections | Group-word matching |
| `unscrambling` | QY Unscrambling | Sentence order via Levenshtein distance |

## Programmatic Usage

```python
from mjnemogym import score_fn_dict, get_score_fn, verl_compute_score

# Option 1: Direct dict lookup
score_fn = score_fn_dict["nemogym_math"]
reward = score_fn("\\boxed{42}", {"expected_answer": "42", "index": 0})

# Option 2: Helper function
score_fn = get_score_fn("nemogym_math")

# Option 3: verl-compatible interface
reward = verl_compute_score(
    data_source="nemogym_math",
    solution_str="\\boxed{42}",
    ground_truth="",
    extra_info={"expected_answer": "42"},
)
```

## Logging

Controlled by two environment variables:

| Variable | Effect | Default |
|---|---|---|
| `MJNEMOGYM_DEBUG=1` | Console shows DEBUG level (entry/exit, timing, rewards) | Off (WARNING only) |
| `MJNEMOGYM_LOG_FILE=path` | Write **all** logs (DEBUG) to file, regardless of `MJNEMOGYM_DEBUG` | Off (no file) |

```bash
# Quiet console, no file (default)
python tests/score_parallel.py -i input.jsonl -o output.jsonl

# Verbose console
MJNEMOGYM_DEBUG=1 python tests/score_parallel.py -i input.jsonl -o output.jsonl

# Quiet console + full debug log to file (recommended for production)
MJNEMOGYM_LOG_FILE=scoring.log python tests/score_parallel.py -i input.jsonl -o output.jsonl

# Verbose console + file (maximum visibility)
MJNEMOGYM_DEBUG=1 MJNEMOGYM_LOG_FILE=scoring.log python tests/score_parallel.py -i input.jsonl -o output.jsonl
```

The file handler **always captures DEBUG level** even when `MJNEMOGYM_DEBUG` is not set. This means you get full diagnostic logs in the file while keeping the console clean for progress bars and summary output.

Log format:
```
[HH:MM:SS][mjnemogym.{domain}][LEVEL][pid=PID] message
```

Logger names by domain:
- `mjnemogym.parallel` - score_parallel.py orchestrator
- `mjnemogym.math` / `mjnemogym.math.qy_parser` / `mjnemogym.math.math_verify` - math scoring chain
- `mjnemogym.code` / `mjnemogym.code.exec` - code generation + subprocess execution
- `mjnemogym.mcqa`, `mjnemogym.if`, `mjnemogym.structured`, `mjnemogym.workplace`, `mjnemogym.qydomain`

To use in your own code:
```python
from mjnemogym.log import get_logger
_logger = get_logger("my_domain")  # creates mjnemogym.my_domain
```

**Note on file logging with multiprocessing:** Each worker process writes to the same log file. Logging's `FileHandler` uses OS-level file locks, so writes are safe but lines from different workers may interleave. Use the `[pid=...]` field to filter per-worker.

---

## Bugs Fixed

### 1. `_run_with_timeout` Hang (CRITICAL) — Two Layers

**Files:** `math_with_judge/qy_parser.py`, `math_with_judge/math_verify_method.py`

**Problem:** The `math_verify` library's `parse()` and `verify()` functions are called with `parsing_timeout=None` and `timeout_seconds=None` (no internal timeout). When they hit pathological input, they run forever. Python cannot kill a thread — a timeout only stops the **caller** from waiting, the stuck thread lives on.

The original code had two separate hang points:

**Hang Point A — `executor.shutdown(wait=True)` in `with`-block:**
```python
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(func)
    return future.result(timeout=5)  # timeout fires correctly
# BUT: with-block __exit__ calls executor.shutdown(wait=True)
# which blocks forever waiting for the stuck thread!
```

**Hang Point B — Process exit joins non-daemon threads:**
Even after fixing Hang Point A with `shutdown(wait=False)`, the stuck thread continues running. `ThreadPoolExecutor` creates **non-daemon threads**. When the worker process exits (after all scoring is done), Python's interpreter shutdown calls `threading._shutdown()` which joins all non-daemon threads — blocking forever. This causes `ProcessPoolExecutor.__exit__()` to hang because the worker process never terminates.

```
All futures done → ProcessPoolExecutor.shutdown(wait=True)
  → signals workers to exit → worker tries to exit
    → Python threading._shutdown() → joins non-daemon thread → HANG
```

This hang is invisible in logs because it happens inside Python's C-level interpreter teardown.

**Fix:** Replace `ThreadPoolExecutor` entirely with a raw `threading.Thread(daemon=True)`:

```python
def _run_with_timeout(func, timeout, default=None):
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
        return default  # thread abandoned, killed on process exit
    return result_box[0]
```

Daemon threads are killed immediately when the process exits — no join, no hang.

### 2. BrokenPipeError in Code Generation

**File:** `code_gen/lcb_integration/compute_code_generation_metrics.py`

**Problem:** `check_correctness()` creates a `multiprocessing.Manager()` with shared `manager.list()` proxies. If `manager.shutdown()` is called before reading results from the proxy objects, accessing them raises `BrokenPipeError` because the Manager server process is already gone.

**Fix:** Copy results from manager proxies into local Python objects **before** shutting down:

```python
# Copy BEFORE shutdown
final_result = list(result[0])
final_metadata = dict(metadata_list[0]) if metadata_list[0] else None

# Then shutdown safely
manager.shutdown()
return final_result, final_metadata
```

### 3. `ds` Variable Bug in score_parallel.py

**Problem:** Line tracking per-domain rewards used a stale `ds` variable from the data-loading loop instead of the current item's `data_source`:

```python
# Bug: ds is leftover from the loading loop
domain_rewards[item["data_source"]] = domain_rewards.get(ds, 0) + reward
```

**Fix:**
```python
domain_rewards[item["data_source"]] = domain_rewards.get(item["data_source"], 0) + reward
```

---

## Safeguards

### 1. 300-second Timeout on Future Results

`score_parallel.py` calls `future.result(timeout=300)`. If any worker hangs for more than 5 minutes, it gets a `TimeoutError`, is assigned `reward=0.0`, and processing continues. Without this, a single hung worker blocks the entire pipeline forever.

### 2. Pending Task Tracking

When fewer than 10 items remain, the orchestrator logs exactly which items are still pending:

```
[mjnemogym.parallel][DEBUG] NEAR_END: 2 remaining: ['line=6127/nemogym_math/idx=11006', 'line=29385/nemogym_code/idx=13486']
```

This immediately tells you **which domain and sample** is stuck.

### 3. Manager Lifecycle Cleanup

Code generation scoring creates a `multiprocessing.Manager()` per item. Each Manager spawns a server process. After results are copied, `manager.shutdown()` is called (wrapped in try/except) to prevent zombie server processes from accumulating.

### 4. Subprocess Kill with Verification

When code execution exceeds its timeout, the subprocess is killed with verification:

```python
p.join(timeout=join_timeout)
if p.is_alive():
    p.kill()
    p.join(timeout=5)  # Verify kill completed
```

### 5. All Domain Score Functions Wrapped with Exception Handling

Every domain's `score_fn` is wrapped with try/except that:
- Logs the exception at WARNING level (always visible)
- Returns `reward=0.0` instead of crashing the worker
- Includes the sample `index` in the log message for tracing

---

## Gotchas

### 1. `nemogym_code` Spawns Nested Processes

Each code scoring creates a `multiprocessing.Manager()` + `multiprocessing.Process()` **inside** the ProcessPoolExecutor worker. With N workers, you get up to N simultaneous Manager server processes + N child processes. On a 200-worker setup, this means up to 600 processes (200 workers + 200 managers + 200 children).

**Recommendation:** Keep `--workers` reasonable for code-heavy workloads. 32-64 workers is usually sufficient.

### 2. Code Execution Timeout Scales with Test Count

The code domain timeout is: `(timeout_secs + 1) * num_test_inputs + 5`. For a sample with 50 test inputs and 10s timeout, join_timeout = **555 seconds** (9+ minutes). This is per-item, not global.

### 3. Math `parse()`/`verify()` with `timeout=None` — Daemon Threads by Design

The `math_verify` library functions are intentionally called with `parsing_timeout=None` and `timeout_seconds=None` because the library's internal timeout mechanism is not thread-safe. Our external `_run_with_timeout` wrapper provides the timeout instead (10s for QY parser, 5s for MathVerify parse/verify).

When a timeout fires, the stuck thread **continues running** — Python has no way to kill a thread. The thread is marked `daemon=True` so it is killed when the worker process exits. This means:
- Memory used by the stuck thread is not freed until the process ends
- On pathological inputs, you may accumulate stuck daemon threads within a worker
- This is harmless in practice because workers process items serially and exit after the job completes

### 4. `<|end_of_thought|>` Stripping

All score functions are wrapped with `extract_final_answer()` in `__init__.py`, which strips everything before the last `<|end_of_thought|>` token. If your model output contains this token, only the text after it is scored.

### 5. Spawn Context Required

The script uses `multiprocessing.set_start_method("spawn")`. This is required because:
- `fork` can deadlock with the thread pools used in math scoring
- `spawn` ensures clean process state without inherited locks

This means every worker process re-imports all modules on startup, which adds ~2-3 seconds of overhead at the beginning.

### 6. NLTK Data Dependency

The instruction-following domain requires NLTK's `punkt_tab` tokenizer. Set the `NLTK_DATA` environment variable if running in an environment without internet access:

```bash
export NLTK_DATA=/path/to/nltk_data
```

### 7. `signal.alarm()` Only Works on Linux

The code execution domain uses `signal.SIGALRM` for per-test-case timeouts inside subprocesses. This is Linux-only. On macOS, these signals may not fire correctly, potentially causing test execution to hang. The outer `p.join(timeout=...)` provides a fallback safety net.

### 8. Memory Usage

All input data is loaded into memory before processing. For large JSONL files (30k+ samples with long responses), expect several GB of memory usage. The output is also buffered in a pre-allocated results array before writing.

### 9. `reliability_guard()` Side Effects in Code Domain

The code execution subprocess calls `reliability_guard()` which disables destructive functions (`os.kill`, `os.remove`, `subprocess.Popen`, etc.) at the module level. This only affects the child subprocess, not the main process, because spawn context creates a fresh process.

---

## Domain Reference

### `nemogym_math`

```json
{"extra_info": {"expected_answer": "42", "index": 0}}
```

Fallback chain: QY Parser (10s timeout) -> DAPO (no timeout, pure regex) -> MathVerify (5s parse + 5s verify). Returns 1.0 on first match, 0.0 if all fail. Extracts from `\boxed{}`.

### `nemogym_code`

```json
{
  "extra_info": {
    "verifier_metadata": {
      "unit_tests": {
        "inputs": ["[1, 2, 3]\n3"],
        "outputs": ["6"],
        "fn_name": "sum_array"
      }
    },
    "timeout_secs": 10,
    "index": 0
  }
}
```

Extracts code from markdown fences. Runs unit tests in isolated subprocess. 1.0 if all pass, 0.0 otherwise. `fn_name: null` triggers stdio-based testing.

### `nemogym_mcqa`

```json
{
  "extra_info": {
    "expected_answer": "B",
    "options": [{"A": "Paris"}, {"B": "London"}],
    "grading_mode": "strict_single_letter_boxed",
    "template_metadata": {"output_regex": "Answer:\\s*([A-D])"},
    "index": 0
  }
}
```

### `nemogym_if`

```json
{
  "extra_info": {
    "instruction_id_list": ["length_constraints:number_words", "detectable_format:title"],
    "kwargs": [{"num_words": 100, "relation": "at least"}, {}],
    "grading_mode": "binary",
    "index": 0
  }
}
```

### `nemogym_structured`

```json
{
  "extra_info": {
    "schema_str": "{\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}}}",
    "schema_type": "json",
    "index": 0
  }
}
```

### `nemogym_workplace`

```json
{
  "extra_info": {
    "response_output": [{"type": "function_call", "name": "create_event", "arguments": "{}"}],
    "ground_truth": [{"name": "create_event", "arguments": "{}"}],
    "index": 0
  }
}
```

### `typos` / `connections` / `unscrambling`

```json
{"extra_info": {"label": "correct answer text", "index": 0}}
```

---

## Project Structure

```
mjnemogym/
  __init__.py                  # score_fn_dict registry, verl_compute_score API
  log.py                       # Centralized logging (MJNEMOGYM_DEBUG env var)
  math_with_judge/
    score.py                   # 3-method fallback chain
    qy_parser.py               # Fast regex + math_verify
    dapo.py                    # Regex normalization + string comparison
    math_verify_method.py      # Full symbolic parsing
  code_gen/
    score.py                   # Code extraction + test dispatch
    lcb_integration/
      compute_code_generation_metrics.py   # Subprocess + Manager
      testing_util.py                      # Test execution + signal handling
      extraction_utils.py                  # Code fence extraction
  mcqa/score.py                # Answer letter parsing
  instruction_following/
    score.py                   # Instruction verification
    verifiable_instructions/   # Instruction library
  structured_outputs/score.py  # JSON schema validation
  workplace_assistant/
    score.py                   # Tool call state comparison
    utils.py                   # Tool environment setup
  qydomain/score.py            # Typos, Connections, Unscrambling

tests/
  score_parallel.py            # Main parallel scoring script
  test_score_fns.py            # Integration tests (all domains)
  test_math_score.py           # Math-specific tests
  test_qydomain_score.py       # QY domain tests
```
