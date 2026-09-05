import numpy as np
import pytest

from precisionprobe.ranking import (
    low_risk_membership,
    random_tie_drift_sensitivity,
    tie_cutoff_diagnostics,
)


def selected_ids(scores, task_ids, count):
    membership = low_risk_membership(scores, task_ids, count)
    return {task_id for task_id, selected in zip(task_ids, membership) if selected}


def test_fixed_tie_rule_is_task_id_based_and_row_order_invariant():
    scores = np.zeros(4)
    task_ids = np.array(["task-d", "task-b", "task-a", "task-c"])
    assert selected_ids(scores, task_ids, 2) == {"task-a", "task-b"}

    permutation = np.array([2, 0, 3, 1])
    assert selected_ids(scores[permutation], task_ids[permutation], 2) == {
        "task-a",
        "task-b",
    }


def test_shared_random_priority_does_not_invent_drift_for_identical_scores():
    scores = np.zeros(8)
    task_ids = np.array([f"task-{index}" for index in range(8)])
    result = random_tie_drift_sensitivity(
        scores,
        scores.copy(),
        task_ids,
        count=4,
        repeats=100,
        seed=17,
    )
    assert result["fixed_id_drift"] == 0.0
    assert result["random_drift_min"] == 0.0
    assert result["random_drift_max"] == 0.0


def test_bootstrap_duplicate_task_ids_use_draw_position_only_as_tertiary_key():
    scores = np.zeros(4)
    sampled_task_ids = np.array(["task-b", "task-a", "task-a", "task-c"])
    membership = low_risk_membership(scores, sampled_task_ids, 2)
    assert membership.tolist() == [False, True, True, False]


def test_random_tie_sensitivity_is_reproducible_and_reports_cutoff_ties():
    q4 = np.array([0.0, 0.0, 0.0, 1.0])
    q8 = np.array([0.0, 0.0, 1.0, 0.0])
    task_ids = np.array(["a", "b", "c", "d"])
    first = random_tie_drift_sensitivity(q4, q8, task_ids, repeats=50, seed=9)
    second = random_tie_drift_sensitivity(q4, q8, task_ids, repeats=50, seed=9)
    assert first == second
    assert first["q4_cutoff"] == {
        "cutoff_score": 0.0,
        "strictly_below_cutoff": 0,
        "tied_at_cutoff": 3,
        "selected_from_cutoff_tie": 2,
    }


def test_ranking_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        low_risk_membership([0.0, 1.0], ["only-one-id"], 1)


def test_zero_selection_has_explicit_cutoff_diagnostics():
    assert tie_cutoff_diagnostics([0.1, 0.2], 0)["cutoff_score"] is None
