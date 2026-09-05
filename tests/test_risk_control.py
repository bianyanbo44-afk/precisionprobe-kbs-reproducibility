import numpy as np

from precisionprobe.risk_control import (
    apply_threshold,
    select_fixed_sequence_precision_threshold,
    select_precision_envelope_threshold,
    select_risk_controlling_threshold,
)


def test_no_feasible_threshold_is_explicit():
    result = select_risk_controlling_threshold(
        np.linspace(0, 1, 20),
        np.ones(20),
        alpha=0.2,
        delta=0.1,
        min_accepted=5,
    )
    assert result.threshold is None
    assert result.status == "no_feasible_threshold"
    assert not apply_threshold([0.1, 0.2], result.threshold).any()


def test_easy_low_score_region_can_be_selected():
    scores = np.linspace(0, 1, 200)
    errors = (scores > 0.60).astype(int)
    result = select_risk_controlling_threshold(
        scores,
        errors,
        alpha=0.20,
        delta=0.10,
        grid=np.linspace(0, 1, 21),
        min_accepted=30,
    )
    assert result.status == "selected"
    assert result.threshold is not None
    assert result.upper_bound <= 0.20


def test_precision_envelope_controls_each_supplied_precision():
    scores = np.linspace(0, 1, 400)
    result = select_precision_envelope_threshold(
        {"q4": scores, "q8": scores},
        {
            "q4": (scores > 0.45).astype(int),
            "q8": (scores > 0.70).astype(int),
        },
        alpha=0.20,
        delta=0.10,
        grid=np.linspace(0, 1, 21),
        min_accepted=40,
    )
    assert result.status == "selected"
    assert result.threshold is not None
    assert set(result.per_precision) == {"q4", "q8"}
    assert all(row["upper_bound"] <= 0.20 for row in result.per_precision.values())


def test_fixed_sequence_selects_last_consecutive_passing_threshold():
    scores = np.linspace(0.0, 1.0, 80)
    labels = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
    selection = select_fixed_sequence_precision_threshold(
        {"q4": scores, "q8": scores},
        {"q4": labels, "q8": labels},
        alpha=0.30,
        delta=0.10,
        grid=[0.25, 0.50, 0.75, 1.0],
        min_accepted=20,
    )
    assert selection.status == "selected"
    assert selection.threshold == 0.50
    assert selection.stopped_at == 0.75
    assert [row["passed"] for row in selection.tested_thresholds] == [True, True, False]
