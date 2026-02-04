#!/usr/bin/env python3
"""
Test script for math scoring with fallback chain.

Tests:
1. Sequential scoring with various input types
2. Thread safety with concurrent execution
3. Individual method testing
"""

import concurrent.futures
import sys
import time

from mjnemogym import verl_compute_score
from mjnemogym.math_with_judge import qy_parser, dapo, math_verify_method


def test_sequential():
    """Test sequential scoring with various input types."""
    print('=== Sequential Tests ===\n', flush=True)

    test_cases = [
        # (model_output, expected_answer, description)
        ('\\boxed{42}', '42', 'Simple number'),
        ('\\boxed{-5}', '-5', 'Negative number'),
        ('\\boxed{3.14}', '3.14', 'Decimal'),
        ('\\boxed{\\frac{1}{2}}', '0.5', 'Fraction vs decimal'),
        ('\\boxed{\\frac{3}{4}}', '0.75', 'Fraction 3/4'),
        ('Answer: 42', '42', 'Answer: pattern'),
        ('The answer is 42', '42', 'Natural language (may fail)'),
        ('\\boxed{x^2 + 2x + 1}', '(x+1)^2', 'Equivalent algebraic expr'),
        ('\\boxed{1,000}', '1000', 'Number with comma'),
        ('\\boxed{1000}', '1,000', 'Reversed comma case'),
        ('\\boxed{\\sqrt{4}}', '2', 'Square root'),
        ('\\boxed{2^3}', '8', 'Exponent'),
        ('', '42', 'Empty output'),
        ('\\boxed{}', '42', 'Empty boxed'),
        ('no answer here', '42', 'No boxed answer'),
    ]

    passed = 0
    failed = 0

    for output, expected, desc in test_cases:
        result = verl_compute_score('nemogym_math', output, '', {'expected_answer': expected})
        # Some cases are expected to fail
        expected_to_pass = desc not in ['Natural language (may fail)', 'Empty output', 'Empty boxed', 'No boxed answer']

        if expected_to_pass:
            status = '✓' if result == 1.0 else '✗'
            if result == 1.0:
                passed += 1
            else:
                failed += 1
        else:
            status = '○'  # Expected to fail

        print(f'  {status} {desc}: {result}', flush=True)

    print(f'\nPassed: {passed}, Failed: {failed}', flush=True)
    return failed == 0


def test_threaded():
    """Test thread safety with concurrent execution."""
    print('\n=== Thread Safety Tests ===\n', flush=True)

    num_workers = 8
    num_tasks = 32

    def test_math(i):
        return verl_compute_score('nemogym_math', f'\\boxed{{{i}}}', '', {'expected_answer': str(i)})

    print(f'Running {num_tasks} tasks with {num_workers} workers...', flush=True)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(test_math, i) for i in range(num_tasks)]
        results = [f.result(timeout=60) for f in futures]
    elapsed = time.time() - start

    all_correct = all(r == 1.0 for r in results)
    print(f'  Results: {sum(r == 1.0 for r in results)}/{len(results)} correct', flush=True)
    print(f'  Time: {elapsed:.2f}s', flush=True)
    print(f'  Status: {"✓ PASSED" if all_correct else "✗ FAILED"}', flush=True)

    return all_correct


def test_individual_methods():
    """Test each scoring method individually."""
    print('\n=== Individual Method Tests ===\n', flush=True)

    test_cases = [
        ('\\boxed{42}', '42'),
        ('\\boxed{\\frac{1}{2}}', '0.5'),
        ('Answer: 100', '100'),
    ]

    methods = [
        ('QY Parser', qy_parser.score_fn),
        ('DAPO', dapo.score_fn),
        ('MathVerify', math_verify_method.score_fn),
    ]

    for method_name, method_fn in methods:
        print(f'{method_name}:', flush=True)
        for output, expected in test_cases:
            try:
                result = method_fn(output, expected)
                print(f'  {output[:30]:30} -> {result}', flush=True)
            except Exception as e:
                print(f'  {output[:30]:30} -> ERROR: {e}', flush=True)


def test_edge_cases():
    """Test edge cases and error handling."""
    print('\n=== Edge Case Tests ===\n', flush=True)

    # Test with None/empty extra_info
    print('Empty extra_info:', flush=True)
    result = verl_compute_score('nemogym_math', '\\boxed{42}', '', {})
    print(f'  Result: {result} (expected 0.0)', flush=True)

    # Test with wrong type
    print('String extra_info (should handle gracefully):', flush=True)
    try:
        result = verl_compute_score('nemogym_math', '\\boxed{42}', '', '{"expected_answer": "42"}')
        print(f'  Result: {result}', flush=True)
    except Exception as e:
        print(f'  Exception: {e}', flush=True)

    # Test with very long input
    print('Long input:', flush=True)
    long_output = 'x' * 10000 + '\\boxed{42}'
    result = verl_compute_score('nemogym_math', long_output, '', {'expected_answer': '42'})
    print(f'  Result: {result}', flush=True)


def main():
    print('=' * 60, flush=True)
    print('MATH SCORE TEST SUITE', flush=True)
    print('=' * 60, flush=True)

    all_passed = True

    all_passed &= test_sequential()
    all_passed &= test_threaded()
    test_individual_methods()
    test_edge_cases()

    print('\n' + '=' * 60, flush=True)
    if all_passed:
        print('ALL TESTS PASSED', flush=True)
        sys.exit(0)
    else:
        print('SOME TESTS FAILED', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
