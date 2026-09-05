from scripts.compare_score_stability import compare


def test_stability_comparison_ignores_only_execution_timing():
    left = {"a": {"task_id": "a", "q4_dsde": 0.2, "q4_execution_seconds": 1.0}}
    right = {"a": {"task_id": "a", "q4_dsde": 0.2, "q4_execution_seconds": 9.0}}
    result = compare(left, right, expected_tasks=1)
    assert result["status"] == "PASS"
    assert result["left_complete"] is True
    assert result["right_complete"] is True


def test_stability_comparison_rejects_identical_partial_files():
    partial = {"a": {"task_id": "a", "q4_dsde": 0.2}}
    result = compare(partial, partial, expected_tasks=2)
    assert result["status"] == "FAIL"
    assert result["left_complete"] is False
    assert result["right_complete"] is False


def test_stability_comparison_rejects_semantic_change():
    left = {"a": {"task_id": "a", "q4_dsde": 0.2}}
    right = {"a": {"task_id": "a", "q4_dsde": 0.3}}
    result = compare(left, right, expected_tasks=1)
    assert result["status"] == "FAIL"
    assert result["semantic_mismatch_task_ids"] == ["a"]
