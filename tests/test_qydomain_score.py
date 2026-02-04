#!/usr/bin/env python3
"""
Test script for QY domain scoring (typos, connections, unscrambling).

Tests:
1. Typos evaluator - exact match / substring check
2. Connections evaluator - word grouping puzzle
3. Unscrambling evaluator - plot sentence ordering
4. Thread safety
"""

import concurrent.futures
import sys
import time

from mjnemogym import verl_compute_score
from mjnemogym.qydomain.score import score_fn, language_judge


def test_typos():
    """Test typos evaluator."""
    print('=== Typos Evaluator Tests ===\n', flush=True)

    test_cases = [
        # (model_output, ground_truth, expected_score, description)
        ('<solution>extraordinary</solution>', 'extraordinary', 1.0, 'Exact match in solution tag'),
        ('<solution>extraordinry</solution>', 'extraordinary', 0.0, 'Typo in answer'),
        ('The answer is <solution>hello</solution>.', 'hello', 1.0, 'Answer with surrounding text'),
        ('--- hello ---', 'hello', 1.0, 'Answer in separator pattern'),
        ('The correct spelling is extraordinary.', 'extraordinary', 1.0, 'Substring match'),
        ('no match here', 'extraordinary', 0.0, 'No match'),
        ('', 'test', 0.0, 'Empty output'),
    ]

    passed = 0
    for output, gt, expected, desc in test_cases:
        result = language_judge(gt, output, 'typos')
        status = '✓' if result == expected else '✗'
        if result == expected:
            passed += 1
        print(f'  {status} {desc}: {result} (expected {expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_connections():
    """Test connections evaluator."""
    print('\n=== Connections Evaluator Tests ===\n', flush=True)

    test_cases = [
        # (model_output, ground_truth, expected_score, description)
        (
            '<solution>Apple, Banana, Pear, Grape, Red, Blue, Green, Yellow</solution>',
            'Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow',
            1.0,
            'Perfect match - 2 groups of 4'
        ),
        (
            '<solution>Apple, Banana, Pear, Orange, Red, Blue, Green, Yellow</solution>',
            'Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow',
            0.5,
            'One group wrong (Orange vs Grape)'
        ),
        (
            '<solution>a,b,c,d</solution>',
            'a,b,c,d',
            1.0,
            'Single group of 4'
        ),
        (
            '\\boxed{\\text{cat, dog, bird, fish}}',
            'cat,dog,bird,fish',
            1.0,
            'Boxed format'
        ),
        (
            'no solution here',
            'a,b,c,d',
            0.0,
            'No solution found'
        ),
    ]

    passed = 0
    for output, gt, expected, desc in test_cases:
        result = language_judge(gt, output, 'connections')
        status = '✓' if result == expected else '✗'
        if result == expected:
            passed += 1
        print(f'  {status} {desc}: {result} (expected {expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_unscrambling():
    """Test plot unscrambling evaluator."""
    print('\n=== Unscrambling Evaluator Tests ===\n', flush=True)

    test_cases = [
        # (model_output, ground_truth, expected_score, description)
        (
            '<PLOT_SUMMARY>The hero wakes up. He fights the dragon. He wins the gold.</PLOT_SUMMARY>',
            'The hero wakes up. He fights the dragon. He wins the gold.',
            1.0,
            'Perfect order'
        ),
        (
            '<PLOT_SUMMARY>The hero wakes up. He wins the gold. He fights the dragon.</PLOT_SUMMARY>',
            'The hero wakes up. He fights the dragon. He wins the gold.',
            0.33,  # Approximate - depends on Levenshtein
            'Swapped last two sentences'
        ),
        (
            '<PLOT_SUMMARY>A. B. C.</PLOT_SUMMARY>',
            'A. B. C.',
            1.0,
            'Short sentences'
        ),
        (
            'no plot summary here',
            'A. B. C.',
            0.33,  # Raw text doesn't match well
            'No PLOT_SUMMARY tag (partial match)'
        ),
    ]

    passed = 0
    for output, gt, expected, desc in test_cases:
        result = language_judge(gt, output, 'unscrambling')
        # Use approximate matching for unscrambling (Levenshtein-based)
        is_close = abs(result - expected) < 0.1
        status = '✓' if is_close else '✗'
        if is_close:
            passed += 1
        print(f'  {status} {desc}: {result:.2f} (expected ~{expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_via_verl_compute_score():
    """Test via verl_compute_score interface."""
    print('\n=== verl_compute_score Interface Tests ===\n', flush=True)

    test_cases = [
        ('qy_typos', '<solution>hello</solution>', {'task_type': 'typos', 'ground_truth': 'hello'}, 1.0),
        ('qy_connections', '<solution>a,b,c,d</solution>', {'task_type': 'connections', 'ground_truth': 'a,b,c,d'}, 1.0),
        ('qy_unscrambling', '<PLOT_SUMMARY>A. B.</PLOT_SUMMARY>', {'task_type': 'unscrambling', 'ground_truth': 'A. B.'}, 1.0),
    ]

    passed = 0
    for data_source, output, extra_info, expected in test_cases:
        result = verl_compute_score(data_source, output, '', extra_info)
        status = '✓' if result == expected else '✗'
        if result == expected:
            passed += 1
        print(f'  {status} {data_source}: {result} (expected {expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_threaded():
    """Test thread safety with concurrent execution."""
    print('\n=== Thread Safety Tests ===\n', flush=True)

    num_workers = 8
    num_tasks = 32

    def test_qy(i):
        return verl_compute_score(
            'qy_typos',
            f'<solution>word{i}</solution>',
            '',
            {'task_type': 'typos', 'ground_truth': f'word{i}'}
        )

    print(f'Running {num_tasks} tasks with {num_workers} workers...', flush=True)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(test_qy, i) for i in range(num_tasks)]
        results = [f.result(timeout=60) for f in futures]
    elapsed = time.time() - start

    all_correct = all(r == 1.0 for r in results)
    print(f'  Results: {sum(r == 1.0 for r in results)}/{len(results)} correct', flush=True)
    print(f'  Time: {elapsed:.2f}s', flush=True)
    print(f'  Status: {"✓ PASSED" if all_correct else "✗ FAILED"}', flush=True)

    return all_correct


def test_edge_cases():
    """Test edge cases and error handling."""
    print('\n=== Edge Case Tests ===\n', flush=True)

    # Missing task_type
    print('Missing task_type:', flush=True)
    result = score_fn('test', {'ground_truth': 'test'})
    print(f'  Result: {result} (expected 0.0)', flush=True)

    # Missing ground_truth
    print('Missing ground_truth:', flush=True)
    result = score_fn('test', {'task_type': 'typos'})
    print(f'  Result: {result} (expected 0.0)', flush=True)

    # Unknown task_type
    print('Unknown task_type:', flush=True)
    result = score_fn('test', {'task_type': 'unknown', 'ground_truth': 'test'})
    print(f'  Result: {result} (expected 0.0)', flush=True)

    # Empty model output
    print('Empty model output:', flush=True)
    result = score_fn('', {'task_type': 'typos', 'ground_truth': 'test'})
    print(f'  Result: {result} (expected 0.0)', flush=True)


def test_connections_old_vs_new():
    """Test connections evaluator version selection based on release_date."""
    print('\n=== Connections Version Tests ===\n', flush=True)

    # Old format uses bold text **...**
    old_output = '**Apple, Banana, Pear, Grape**'
    gt = 'Apple,Banana,Pear,Grape'

    # Test with old date (before 2024-11-25)
    result_old = score_fn(old_output, {
        'task_type': 'connections',
        'ground_truth': gt,
        'release_date': '2024-01-01'
    })
    print(f'  Old format with old date: {result_old}', flush=True)

    # Test with new date (after 2024-11-25)
    result_new = score_fn(old_output, {
        'task_type': 'connections',
        'ground_truth': gt,
        'release_date': '2025-01-01'
    })
    print(f'  Old format with new date: {result_new}', flush=True)


def main():
    print('=' * 60, flush=True)
    print('QY DOMAIN SCORE TEST SUITE', flush=True)
    print('=' * 60, flush=True)

    all_passed = True

    all_passed &= test_typos()
    all_passed &= test_connections()
    all_passed &= test_unscrambling()
    all_passed &= test_via_verl_compute_score()
    all_passed &= test_threaded()
    test_edge_cases()
    test_connections_old_vs_new()

    print('\n' + '=' * 60, flush=True)
    if all_passed:
        print('ALL TESTS PASSED', flush=True)
        sys.exit(0)
    else:
        print('SOME TESTS FAILED', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
