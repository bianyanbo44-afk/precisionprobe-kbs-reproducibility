import pytest

from precisionprobe.scoring import (
    Observation,
    ast_distance,
    behavioral_distance,
    dominant_semantic_distance_entropy,
    exact_disagreement_score,
    exact_dominant_semantic_distance_entropy,
    exact_semantic_distance_entropy,
    semantic_distance_entropy,
    xpbd_score,
)


def test_behavioral_distance_is_bounded_and_exact_on_equal_values():
    left = Observation("ok", [1, 2, 3])
    right = Observation("ok", [1, 2, 3])
    assert behavioral_distance(left, right) == 0.0
    assert 0.0 <= behavioral_distance(left, Observation("ok", [1, 9, 3])) <= 1.0


def test_status_mismatch_is_maximal_for_success_vs_failure():
    assert behavioral_distance(Observation("ok", 1), Observation("exception", exception="ValueError")) == 1.0


def test_xpbd_penalizes_missing_observation():
    score = xpbd_score([{"status": "ok", "value": 1}], [])
    assert score == 1.0


def test_ast_distance_ignores_formatting():
    assert ast_distance("def f(x):\n return x+1\n", "def f(x):\n    return x + 1\n") == 0.0


def test_semantic_distance_aggregates_equally_weighted_samples():
    same = [{"status": "ok", "value": 1}]
    different = [{"status": "ok", "value": 2}]
    executions = [same, same, different]
    assert semantic_distance_entropy(executions) == pytest.approx(1 / 9)
    assert dominant_semantic_distance_entropy(same, executions) == pytest.approx(1 / 6)
    assert exact_disagreement_score(same, different) == 1.0
    assert exact_semantic_distance_entropy(executions) == pytest.approx(2 / 9)
    assert exact_dominant_semantic_distance_entropy(same, executions) == pytest.approx(1 / 3)
