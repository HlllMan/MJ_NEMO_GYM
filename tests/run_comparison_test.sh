#!/bin/bash
# Run single vs parallel comparison test with full logging
#
# Usage:
#   ./tests/run_comparison_test.sh [LIMIT] [WORKERS]
#
# Example:
#   ./tests/run_comparison_test.sh 50 128    # 50 samples, 128 workers
#   ./tests/run_comparison_test.sh 100 64    # 100 samples, 64 workers

LIMIT=${1:-50}
WORKERS=${2:-128}
DATA="tests/code_random_128.jsonl"

echo "=============================================="
echo "Parallel Code Evaluation Comparison Test"
echo "=============================================="
echo "Samples: $LIMIT"
echo "Workers: $WORKERS"
echo "Data: $DATA"
echo "Python: $(python3 --version)"
echo "Platform: $(uname -a)"
echo "CPUs: $(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count())')"
echo "Date: $(date)"
echo "=============================================="
echo ""

# Run single process test
echo ">>> Running SINGLE process test..."
echo ">>> Output: single.log"
python3 tests/test_parallel_eval.py \
    --mode single \
    --limit $LIMIT \
    --workers $WORKERS \
    --data $DATA \
    2>&1 | tee single.log

echo ""
echo ">>> Single test completed. Log saved to single.log"
echo ""

# Run parallel process test
echo ">>> Running PARALLEL process test with $WORKERS workers..."
echo ">>> Output: parallel.log"
python3 tests/test_parallel_eval.py \
    --mode parallel \
    --limit $LIMIT \
    --workers $WORKERS \
    --data $DATA \
    2>&1 | tee parallel.log

echo ""
echo ">>> Parallel test completed. Log saved to parallel.log"
echo ""

# Extract and compare results
echo "=============================================="
echo "COMPARISON SUMMARY"
echo "=============================================="

echo ""
echo "Single results:"
grep -E "^\[Single\]" single.log | sort -t= -k2 -n > single_results.txt
cat single_results.txt | head -20
SINGLE_TOTAL=$(grep "Total reward:" single.log | tail -1)
echo "..."
echo "$SINGLE_TOTAL"

echo ""
echo "Parallel results:"
grep -E "^\[Parallel\]" parallel.log | sort -t= -k2 -n > parallel_results.txt
cat parallel_results.txt | head -20
PARALLEL_TOTAL=$(grep "Total reward:" parallel.log | tail -1)
echo "..."
echo "$PARALLEL_TOTAL"

echo ""
echo "=============================================="
echo "MISMATCH DETECTION"
echo "=============================================="

# Create comparison
python3 << 'EOF'
import re

def parse_results(filename):
    results = {}
    with open(filename) as f:
        for line in f:
            match = re.search(r'idx=(\d+), reward=([\d.]+)', line)
            if match:
                idx = int(match.group(1))
                reward = float(match.group(2))
                results[idx] = reward
    return results

single = parse_results('single_results.txt')
parallel = parse_results('parallel_results.txt')

all_idx = sorted(set(single.keys()) | set(parallel.keys()))

mismatches = []
single_1_parallel_0 = []

for idx in all_idx:
    s = single.get(idx, -999)
    p = parallel.get(idx, -999)
    if s != p:
        mismatches.append((idx, s, p))
        if s == 1.0 and p == 0.0:
            single_1_parallel_0.append(idx)

print(f"Total samples compared: {len(all_idx)}")
print(f"Total mismatches: {len(mismatches)}")
print(f"Single=1 but Parallel=0: {len(single_1_parallel_0)}")

if mismatches:
    print("\nMismatch details:")
    for idx, s, p in mismatches[:50]:
        marker = " *** BUG: single=1, parallel=0 ***" if s == 1.0 and p == 0.0 else ""
        print(f"  idx={idx}: single={s}, parallel={p}{marker}")

    if single_1_parallel_0:
        print(f"\n*** CRITICAL BUG DETECTED ***")
        print(f"Indices where single=1 but parallel=0: {single_1_parallel_0}")
else:
    print("\nNo mismatches found - single and parallel produce identical results!")

# Also check for errors in parallel log
print("\n" + "="*50)
print("ERROR ANALYSIS (from parallel.log)")
print("="*50)

import subprocess
result = subprocess.run(['grep', '-c', 'EMPTY RESULT', 'parallel.log'], capture_output=True, text=True)
empty_count = int(result.stdout.strip()) if result.returncode == 0 else 0
print(f"EMPTY RESULT (subprocess crash): {empty_count}")

result = subprocess.run(['grep', '-c', 'EXCEPTION', 'parallel.log'], capture_output=True, text=True)
exception_count = int(result.stdout.strip()) if result.returncode == 0 else 0
print(f"EXCEPTION messages: {exception_count}")

result = subprocess.run(['grep', '-c', 'TIMEOUT', 'parallel.log'], capture_output=True, text=True)
timeout_count = int(result.stdout.strip()) if result.returncode == 0 else 0
print(f"TIMEOUT messages: {timeout_count}")

EOF

echo ""
echo "=============================================="
echo "Log files saved:"
echo "  - single.log (full single-process output)"
echo "  - parallel.log (full parallel output)"
echo "  - single_results.txt (extracted results)"
echo "  - parallel_results.txt (extracted results)"
echo "=============================================="
