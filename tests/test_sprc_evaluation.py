import pandas as pd

from scripts.evaluate_sprc import always_accept, hash_random_matched


def test_always_accept_reports_full_empirical_risk():
    frame = pd.DataFrame(
        {
            "task_id": ["a", "b", "c", "d"],
            "q4_error": [0, 1, 0, 1],
        }
    )
    result = always_accept(frame, "q4")
    assert result == {
        "accepted": 4,
        "coverage": 1.0,
        "errors": 2,
        "empirical_risk": 0.5,
    }


def test_hash_random_baseline_is_deterministic_and_label_independent():
    frame = pd.DataFrame(
        {
            "task_id": ["a", "b", "c", "d", "e"],
            "q8_error": [0, 1, 0, 1, 1],
        }
    )
    first = hash_random_matched(frame, "q8", 3)
    second = hash_random_matched(frame.sample(frac=1.0, random_state=7), "q8", 3)
    assert first == second
    assert first["accepted"] == 3
    assert first["coverage"] == 0.6


def test_hash_random_baseline_handles_zero_coverage():
    frame = pd.DataFrame({"task_id": ["a"], "q4_error": [1]})
    result = hash_random_matched(frame, "q4", 0)
    assert result["accepted"] == 0
    assert result["empirical_risk"] is None
