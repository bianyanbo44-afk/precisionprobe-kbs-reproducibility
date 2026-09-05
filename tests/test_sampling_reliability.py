import numpy as np

from scripts.analyze_sampling_reliability import (
    statistics,
    summarize_greedy_reproducibility,
)


def test_reliability_contrast_separates_within_from_cross_precision_ordering():
    ascending = np.arange(8, dtype=float)
    descending = ascending[::-1]
    result = statistics(
        {
            "q4_a": ascending,
            "q4_b": ascending,
            "q8_a": descending,
            "q8_b": descending,
        },
        np.asarray([f"task-{index}" for index in range(len(ascending))]),
    )
    assert result["rho_within_q4"] == 1.0
    assert result["rho_within_q8"] == 1.0
    assert result["rho_cross_a"] == -1.0
    assert result["rho_cross_b"] == -1.0
    assert result["rho_contrast_within_minus_cross"] == 2.0
    assert result["drift_contrast_cross_minus_within"] > 0


def test_reliability_drift_is_row_order_invariant_with_score_ties():
    task_ids = np.asarray(["task-d", "task-b", "task-a", "task-c"])
    arrays = {
        "q4_a": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "q4_b": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "q8_a": np.asarray([0.0, 1.0, 0.0, 0.0]),
        "q8_b": np.asarray([0.0, 1.0, 0.0, 0.0]),
    }
    first = statistics(arrays, task_ids)

    permutation = np.asarray([2, 0, 3, 1])
    second = statistics(
        {name: values[permutation] for name, values in arrays.items()},
        task_ids[permutation],
    )
    assert first == second


def test_greedy_reproducibility_requires_complete_matching_task_sets():
    complete = summarize_greedy_reproducibility(
        {"a": "hash-a", "b": "hash-b"},
        {"a": "hash-a", "b": "hash-b"},
        ["a", "b"],
    )
    assert complete["complete_match"] is True

    mismatched = summarize_greedy_reproducibility(
        {"a": "hash-a", "b": "hash-b"},
        {"a": "hash-a", "b": "changed", "c": "unexpected"},
        ["a", "b"],
    )
    assert mismatched["complete_match"] is False
    assert mismatched["mismatched_task_ids"] == ["b", "c"]
    assert mismatched["unexpected_task_ids"] == ["c"]
