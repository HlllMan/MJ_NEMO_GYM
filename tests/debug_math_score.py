#!/usr/bin/env python3
"""
Debug script to diagnose why math scoring returns 0.0 in distributed environment
but works correctly in isolation.
"""

import json
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test data from the failed rollout
TEST_DATA = {
    "output": """
We are given: "A ball was floating in a lake when the lake froze. The ball was removed (without breaking the ice), leaving a hole 24 cm across as the top and 8 cm deep. What was the radius of the ball (in centimeters)?" Options: (A) 8, (B) 12, (C) 13, (D) 8√3, (E) 6√6.

Thus answer: \\boxed{13}.
<|end_of_thought|>
The hole left by the removed ball is a spherical cap with a circular top of diameter 24 cm (radius $r = 12$ cm) and depth $h = 8$ cm. For a spherical cap, the radius $R$ of the sphere is given by the formula:

$
R = \\frac{r^2 + h^2}{2h}.
$

Substituting $r = 12$ and $h = 8$:

$
R = \\frac{12^2 + 8^2}{2 \\cdot 8} = \\frac{144 + 64}{16} = \\frac{208}{16} = 13.
$

Thus, the radius of the ball is 13 cm.

\\boxed{13}""",
    "extra_info": {
        "index": 56620,
        "expected_answer": "13",
        "question": "A ball was floating in a lake..."
    },
    "data_source": "nemogym_math"
}


def test_1_direct_score_fn():
    """Test 1: Direct call to math score_fn (bypasses decorators)"""
    print("\n" + "="*60)
    print("TEST 1: Direct call to math_with_judge.score.score_fn")
    print("="*60)

    from mjnemogym.math_with_judge.score import score_fn

    model_output = TEST_DATA["output"]
    extra_info = TEST_DATA["extra_info"]

    print(f"extra_info type: {type(extra_info)}")
    print(f"expected_answer: {extra_info.get('expected_answer')}")
    boxed_check = '\\boxed{13}' in model_output
    print(f"model_output contains \\boxed{{13}}: {boxed_check}")

    try:
        score = score_fn(model_output, extra_info)
        print(f"Score: {score}")
        return score
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_2_with_extract_final_answer():
    """Test 2: Call through extract_final_answer decorator"""
    print("\n" + "="*60)
    print("TEST 2: Through extract_final_answer decorator")
    print("="*60)

    from mjnemogym import score_fn_dict

    model_output = TEST_DATA["output"]
    extra_info = TEST_DATA["extra_info"]
    data_source = TEST_DATA["data_source"]

    # This is what verl_compute_score does
    score_fn = score_fn_dict[data_source]

    # Show what extract_final_answer does
    parts = model_output.split("<|end_of_thought|>")
    print(f"Split on <|end_of_thought|>: {len(parts)} parts")
    print(f"Last part preview: {parts[-1][:200]}...")
    boxed_in_last = '\\boxed{13}' in parts[-1]
    print(f"Last part contains \\boxed{{13}}: {boxed_in_last}")

    try:
        score = score_fn(model_output, extra_info)
        print(f"Score: {score}")
        return score
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_3_verl_compute_score():
    """Test 3: Call through verl_compute_score (full path)"""
    print("\n" + "="*60)
    print("TEST 3: Through verl_compute_score (full path)")
    print("="*60)

    from mjnemogym import verl_compute_score

    model_output = TEST_DATA["output"]
    extra_info = TEST_DATA["extra_info"]
    data_source = TEST_DATA["data_source"]
    ground_truth = extra_info.get("expected_answer", "")

    try:
        score = verl_compute_score(
            data_source=data_source,
            solution_str=model_output,
            ground_truth=ground_truth,
            extra_info=extra_info
        )
        print(f"Score: {score}")
        return score
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_4_json_serialized_extra_info():
    """Test 4: What if extra_info comes as JSON string (Ray serialization)"""
    print("\n" + "="*60)
    print("TEST 4: extra_info as JSON string (simulating Ray serialization)")
    print("="*60)

    from mjnemogym import verl_compute_score

    model_output = TEST_DATA["output"]
    # Simulate what might happen in Ray - extra_info as JSON string
    extra_info_json = json.dumps(TEST_DATA["extra_info"])
    data_source = TEST_DATA["data_source"]

    print(f"extra_info type: {type(extra_info_json)}")
    print(f"extra_info value: {extra_info_json[:100]}...")

    # Try with JSON string
    try:
        score = verl_compute_score(
            data_source=data_source,
            solution_str=model_output,
            ground_truth="13",
            extra_info=extra_info_json  # This is a STRING, not dict!
        )
        print(f"Score with JSON string: {score}")
    except Exception as e:
        print(f"EXCEPTION with JSON string: {type(e).__name__}: {e}")

    # What does score_fn do with a string?
    from mjnemogym.math_with_judge.score import score_fn
    try:
        # If extra_info is a string, .get() will fail!
        result = extra_info_json.get("expected_answer", "")
        print(f"string.get() result: {result}")
    except AttributeError as e:
        print(f"AttributeError (expected): {e}")
        print(">>> THIS IS THE BUG! If extra_info is serialized as JSON string,")
        print(">>> string.get('expected_answer') raises AttributeError!")


def test_5_empty_extra_info():
    """Test 5: What if extra_info is empty or None"""
    print("\n" + "="*60)
    print("TEST 5: Empty/None extra_info")
    print("="*60)

    from mjnemogym.math_with_judge.score import score_fn

    model_output = TEST_DATA["output"]

    # Test with empty dict
    score = score_fn(model_output, {})
    print(f"Score with empty dict: {score}")

    # Test with None (should fail)
    try:
        score = score_fn(model_output, None)
        print(f"Score with None: {score}")
    except Exception as e:
        print(f"EXCEPTION with None: {type(e).__name__}: {e}")


def test_6_check_verifier_singleton():
    """Test 6: Check if verifier singleton works correctly"""
    print("\n" + "="*60)
    print("TEST 6: Verifier singleton behavior")
    print("="*60)

    from mjnemogym.math_with_judge.score import _get_verifier, _global_verifier

    print(f"_global_verifier before: {_global_verifier}")

    v1 = _get_verifier()
    print(f"Verifier 1: {v1}")

    v2 = _get_verifier()
    print(f"Verifier 2: {v2}")

    print(f"Same instance: {v1 is v2}")

    # Test verification
    reward, extracted = v1.verify_answer("13", "The answer is \\boxed{13}")
    print(f"Direct verify_answer result: reward={reward}, extracted={extracted}")


def test_7_from_file():
    """Test 7: Load actual failed data from file"""
    print("\n" + "="*60)
    print("TEST 7: Load from actual failed rollout file")
    print("="*60)

    jsonl_path = "tests/debug_wandb_gens/merged_output_bruteforce.jsonl"

    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    from mjnemogym import verl_compute_score

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} entries")

    mismatches = 0
    for i, line in enumerate(lines[:10]):  # Check first 10
        if not line.strip():
            continue

        data = json.loads(line)
        model_output = data['output']
        extra_info = data['extra_info']
        recorded_score = data.get('score', -1)
        data_source = data.get('data_source', 'nemogym_math')

        # Check extra_info type
        if isinstance(extra_info, str):
            print(f"  Entry {i}: extra_info is STRING! Parsing...")
            extra_info = json.loads(extra_info)

        try:
            calculated_score = verl_compute_score(
                data_source=data_source,
                solution_str=model_output,
                ground_truth=extra_info.get('expected_answer', ''),
                extra_info=extra_info
            )
        except Exception as e:
            calculated_score = f"ERROR: {e}"

        if calculated_score != recorded_score:
            mismatches += 1
            print(f"  Entry {i}: MISMATCH - recorded={recorded_score}, calculated={calculated_score}")
            print(f"    expected_answer: {extra_info.get('expected_answer')}")
        else:
            print(f"  Entry {i}: OK (score={recorded_score})")

    print(f"\nTotal mismatches in first 10: {mismatches}")


if __name__ == "__main__":
    print("="*60)
    print("MATH SCORE DEBUGGING")
    print("="*60)

    test_1_direct_score_fn()
    test_2_with_extract_final_answer()
    test_3_verl_compute_score()
    test_4_json_serialized_extra_info()
    test_5_empty_extra_info()
    test_6_check_verifier_singleton()
    test_7_from_file()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Potential causes of 0.0 scores in distributed environment:

1. extra_info is JSON string instead of dict (Ray serialization)
   - string.get('expected_answer') raises AttributeError
   - Exception caught in verl_compute_score, returns 0.0

2. expected_answer missing from extra_info
   - Returns 0.0 immediately

3. Verifier singleton issues in Ray workers
   - Each worker creates its own singleton

4. math_verify library not available in Ray worker environment
   - Import fails, exception caught, returns 0.0

Check the logs for error messages from verl_compute_score!
""")
