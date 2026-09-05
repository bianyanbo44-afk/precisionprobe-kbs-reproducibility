from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from precisionprobe.risk_control import (
    PrecisionEnvelopeSelection,
    apply_threshold,
    select_precision_envelope_threshold,
)


def empirical_percentiles(
    calibration_scores: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Map scores to right-continuous empirical percentiles.

    Lower values remain lower risk. The map depends only on calibration scores,
    not labels, and is invariant to strictly increasing score transforms.
    """

    calibration = np.asarray(calibration_scores, dtype=float)
    target = np.asarray(values, dtype=float)
    if calibration.ndim != 1 or target.ndim != 1:
        raise ValueError("calibration_scores and values must be one-dimensional")
    if calibration.size == 0:
        raise ValueError("calibration_scores must not be empty")
    if not np.isfinite(calibration).all() or not np.isfinite(target).all():
        raise ValueError("scores must be finite")
    ordered = np.sort(calibration)
    return np.searchsorted(ordered, target, side="right") / ordered.size


@dataclass(frozen=True)
class PrecisionQuantilePolicy:
    threshold: float | None
    score_cutoffs: dict[str, float | None]
    calibration_sizes: dict[str, int]
    selection: PrecisionEnvelopeSelection

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["selection"] = self.selection.to_dict()
        return payload


@dataclass(frozen=True)
class SplitPrecisionQuantilePolicy:
    """Quantile policy with independent score-reference and risk-calibration sets."""

    threshold: float | None
    reference_score_cutoffs: dict[str, float | None]
    reference_sizes: dict[str, int]
    risk_calibration_sizes: dict[str, int]
    selection: PrecisionEnvelopeSelection

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["selection"] = self.selection.to_dict()
        return payload


def fit_precision_quantile_policy(
    scores: Mapping[str, Sequence[float] | np.ndarray],
    errors: Mapping[str, Sequence[int | bool] | np.ndarray],
    *,
    alpha: float,
    delta: float,
    grid: Sequence[float] | np.ndarray,
    min_accepted: int,
) -> PrecisionQuantilePolicy:
    """Fit one percentile threshold with simultaneous per-precision risk bounds."""

    names = sorted(scores)
    if not names or set(names) != set(errors):
        raise ValueError("scores and errors must have the same non-empty keys")
    transformed = {
        name: empirical_percentiles(scores[name], scores[name]) for name in names
    }
    selection = select_precision_envelope_threshold(
        transformed,
        errors,
        alpha=alpha,
        delta=delta,
        grid=grid,
        min_accepted=min_accepted,
    )
    cutoffs: dict[str, float | None] = {}
    for name in names:
        values = np.asarray(scores[name], dtype=float)
        accepted = apply_threshold(transformed[name], selection.threshold)
        cutoffs[name] = float(values[accepted].max()) if accepted.any() else None
    return PrecisionQuantilePolicy(
        threshold=selection.threshold,
        score_cutoffs=cutoffs,
        calibration_sizes={name: len(scores[name]) for name in names},
        selection=selection,
    )


def apply_precision_quantile_policy(
    policy: PrecisionQuantilePolicy,
    calibration_scores: Mapping[str, Sequence[float] | np.ndarray],
    values: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply a fitted common percentile threshold to new per-precision scores."""

    if set(calibration_scores) != set(values) or set(values) != set(policy.score_cutoffs):
        raise ValueError("policy, calibration_scores, and values must have identical keys")
    return {
        name: apply_threshold(
            empirical_percentiles(calibration_scores[name], values[name]),
            policy.threshold,
        )
        for name in sorted(values)
    }


def fit_split_precision_quantile_policy(
    reference_scores: Mapping[str, Sequence[float] | np.ndarray],
    risk_calibration_scores: Mapping[str, Sequence[float] | np.ndarray],
    risk_calibration_errors: Mapping[
        str, Sequence[int | bool] | np.ndarray
    ],
    *,
    alpha: float,
    delta: float,
    grid: Sequence[float] | np.ndarray,
    min_accepted: int,
) -> SplitPrecisionQuantilePolicy:
    """Fit a common percentile policy using independent reference and risk sets.

    Conditional on the reference set, the empirical-CDF acceptance rules are
    fixed before the independent risk-calibration labels are examined. This
    separation is what permits the finite-grid binomial bounds to retain their
    usual simultaneous interpretation.
    """

    names = sorted(reference_scores)
    if (
        not names
        or set(names) != set(risk_calibration_scores)
        or set(names) != set(risk_calibration_errors)
    ):
        raise ValueError(
            "reference scores, risk-calibration scores, and errors must have "
            "the same non-empty keys"
        )
    transformed: dict[str, np.ndarray] = {}
    for name in names:
        scores = np.asarray(risk_calibration_scores[name], dtype=float)
        errors = np.asarray(risk_calibration_errors[name], dtype=int)
        if scores.ndim != 1 or scores.shape != errors.shape:
            raise ValueError(f"invalid risk-calibration arrays for {name}")
        transformed[name] = empirical_percentiles(reference_scores[name], scores)

    selection = select_precision_envelope_threshold(
        transformed,
        risk_calibration_errors,
        alpha=alpha,
        delta=delta,
        grid=grid,
        min_accepted=min_accepted,
    )
    cutoffs: dict[str, float | None] = {}
    for name in names:
        reference = np.asarray(reference_scores[name], dtype=float)
        accepted = apply_threshold(
            empirical_percentiles(reference, reference), selection.threshold
        )
        cutoffs[name] = float(reference[accepted].max()) if accepted.any() else None
    return SplitPrecisionQuantilePolicy(
        threshold=selection.threshold,
        reference_score_cutoffs=cutoffs,
        reference_sizes={name: len(reference_scores[name]) for name in names},
        risk_calibration_sizes={
            name: len(risk_calibration_scores[name]) for name in names
        },
        selection=selection,
    )


def apply_split_precision_quantile_policy(
    policy: SplitPrecisionQuantilePolicy,
    reference_scores: Mapping[str, Sequence[float] | np.ndarray],
    values: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply a split-sample policy using only its score-reference set."""

    if (
        set(reference_scores) != set(values)
        or set(values) != set(policy.reference_score_cutoffs)
    ):
        raise ValueError("policy, reference_scores, and values must have identical keys")
    return {
        name: apply_threshold(
            empirical_percentiles(reference_scores[name], values[name]),
            policy.threshold,
        )
        for name in sorted(values)
    }
