import numpy as np

from precisionprobe.precision_calibration import (
    apply_split_precision_quantile_policy,
    apply_precision_quantile_policy,
    empirical_percentiles,
    fit_split_precision_quantile_policy,
    fit_precision_quantile_policy,
)


def test_empirical_percentiles_are_invariant_to_increasing_affine_transform():
    calibration = np.array([0.0, 0.1, 0.1, 0.4, 0.9])
    values = np.array([-0.1, 0.1, 0.2, 1.0])
    expected = empirical_percentiles(calibration, values)
    transformed = empirical_percentiles(7.0 * calibration + 3.0, 7.0 * values + 3.0)
    np.testing.assert_allclose(transformed, expected)


def test_precision_quantile_policy_handles_precision_scale_mismatch():
    low_risk = np.linspace(0.0, 0.39, 40)
    high_risk = np.linspace(0.60, 1.0, 40)
    base = np.concatenate([low_risk, high_risk])
    errors = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
    scores = {"q4": base, "q8": 100.0 + 50.0 * base}
    labels = {"q4": errors, "q8": errors}
    policy = fit_precision_quantile_policy(
        scores,
        labels,
        alpha=0.30,
        delta=0.10,
        grid=np.linspace(0.0, 1.0, 41),
        min_accepted=20,
    )
    assert policy.selection.status == "selected"
    selected = apply_precision_quantile_policy(policy, scores, scores)
    assert selected["q4"].sum() == selected["q8"].sum()
    assert selected["q4"].sum() >= 20


def test_empirical_percentiles_reject_nonfinite_scores():
    with np.testing.assert_raises(ValueError):
        empirical_percentiles([0.0, np.nan], [0.1])


def test_split_precision_quantile_policy_uses_independent_reference_scale():
    reference = np.linspace(0.0, 1.0, 60)
    calibration = np.linspace(0.0, 1.0, 60)
    errors = np.concatenate([np.zeros(45, dtype=int), np.ones(15, dtype=int)])
    policy = fit_split_precision_quantile_policy(
        {"q4": reference, "q8": 100.0 + 20.0 * reference},
        {"q4": calibration, "q8": 100.0 + 20.0 * calibration},
        {"q4": errors, "q8": errors},
        alpha=0.30,
        delta=0.10,
        grid=np.linspace(0.0, 1.0, 41),
        min_accepted=20,
    )
    assert policy.selection.status == "selected"
    selected = apply_split_precision_quantile_policy(
        policy,
        {"q4": reference, "q8": 100.0 + 20.0 * reference},
        {"q4": calibration, "q8": 100.0 + 20.0 * calibration},
    )
    assert selected["q4"].sum() == selected["q8"].sum()
    assert selected["q4"].sum() >= 20
