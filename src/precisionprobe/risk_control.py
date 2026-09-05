from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float | None
    accepted: int
    errors: int
    empirical_risk: float | None
    upper_bound: float
    alpha: float
    delta: float
    grid_size: int
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionEnvelopeSelection:
    threshold: float | None
    per_precision: dict[str, dict[str, float | int]]
    alpha: float
    delta: float
    grid_size: int
    precision_count: int
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FixedSequencePrecisionSelection:
    threshold: float | None
    per_precision: dict[str, dict[str, float | int]]
    tested_thresholds: list[dict]
    stopped_at: float | None
    alpha: float
    delta: float
    grid_size: int
    precision_count: int
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def clopper_pearson_upper(errors: int, accepted: int, delta: float) -> float:
    """One-sided exact binomial upper confidence bound."""

    if accepted <= 0:
        return 1.0
    if errors >= accepted:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, accepted - errors))


def select_risk_controlling_threshold(
    scores: Sequence[float] | np.ndarray,
    errors: Sequence[int | bool] | np.ndarray,
    *,
    alpha: float,
    delta: float = 0.10,
    grid: Iterable[float] | None = None,
    min_accepted: int = 5,
) -> ThresholdSelection:
    """Select the largest fixed-grid threshold with a simultaneous binomial UCB <= alpha.

    The grid must be fixed independently of calibration labels. A Bonferroni correction
    makes the confidence statements simultaneous over the tested thresholds. This is a
    conservative finite-grid policy, not a claim of distribution-shift validity.
    """

    score_array = np.asarray(scores, dtype=float)
    error_array = np.asarray(errors, dtype=int)
    if score_array.shape != error_array.shape:
        raise ValueError("scores and errors must have identical shapes")
    if score_array.ndim != 1:
        raise ValueError("scores and errors must be one-dimensional")
    if not (0.0 < alpha < 1.0 and 0.0 < delta < 1.0):
        raise ValueError("alpha and delta must be in (0, 1)")
    if np.any((error_array != 0) & (error_array != 1)):
        raise ValueError("errors must be binary")

    threshold_grid = np.asarray(
        list(grid) if grid is not None else np.linspace(0.0, 1.0, 101),
        dtype=float,
    )
    if threshold_grid.ndim != 1 or threshold_grid.size == 0:
        raise ValueError("grid must contain at least one threshold")
    corrected_delta = delta / threshold_grid.size
    feasible: list[tuple[int, float, int, int, float]] = []

    for threshold in threshold_grid:
        mask = score_array <= threshold
        accepted = int(mask.sum())
        if accepted < min_accepted:
            continue
        failures = int(error_array[mask].sum())
        upper = clopper_pearson_upper(failures, accepted, corrected_delta)
        if upper <= alpha:
            feasible.append((accepted, float(threshold), failures, accepted, upper))

    if not feasible:
        return ThresholdSelection(
            threshold=None,
            accepted=0,
            errors=0,
            empirical_risk=None,
            upper_bound=1.0,
            alpha=alpha,
            delta=delta,
            grid_size=int(threshold_grid.size),
            status="no_feasible_threshold",
        )

    _, threshold, failures, accepted, upper = max(feasible, key=lambda row: (row[0], row[1]))
    return ThresholdSelection(
        threshold=threshold,
        accepted=accepted,
        errors=failures,
        empirical_risk=failures / accepted,
        upper_bound=upper,
        alpha=alpha,
        delta=delta,
        grid_size=int(threshold_grid.size),
        status="selected",
    )


def apply_threshold(scores: Sequence[float], threshold: float | None) -> np.ndarray:
    if threshold is None:
        return np.zeros(len(scores), dtype=bool)
    return np.asarray(scores, dtype=float) <= threshold


def select_precision_envelope_threshold(
    scores: Mapping[str, Sequence[float] | np.ndarray],
    errors: Mapping[str, Sequence[int | bool] | np.ndarray],
    *,
    alpha: float,
    delta: float = 0.10,
    grid: Iterable[float] | None = None,
    min_accepted: int = 5,
) -> PrecisionEnvelopeSelection:
    """Choose one threshold whose binomial risk bound holds at every precision.

    The precision family and threshold grid must be fixed without consulting
    calibration labels. Bonferroni correction is applied jointly across the
    precision-by-threshold family. The claim is simultaneous over the supplied
    finite deployment family under the same IID assumptions as the individual
    binomial bounds; it is not a distribution-shift guarantee.
    """

    names = sorted(scores)
    if not names or set(names) != set(errors):
        raise ValueError("scores and errors must have the same non-empty precision keys")
    if not (0.0 < alpha < 1.0 and 0.0 < delta < 1.0):
        raise ValueError("alpha and delta must be in (0, 1)")
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        score_array = np.asarray(scores[name], dtype=float)
        error_array = np.asarray(errors[name], dtype=int)
        if score_array.ndim != 1 or score_array.shape != error_array.shape:
            raise ValueError(f"invalid score/error shapes for precision {name}")
        if np.any((error_array != 0) & (error_array != 1)):
            raise ValueError(f"errors must be binary for precision {name}")
        arrays[name] = (score_array, error_array)

    threshold_grid = np.asarray(
        list(grid) if grid is not None else np.linspace(0.0, 1.0, 101), dtype=float
    )
    if threshold_grid.ndim != 1 or threshold_grid.size == 0:
        raise ValueError("grid must contain at least one threshold")
    corrected_delta = delta / (len(names) * threshold_grid.size)
    feasible: list[tuple[int, float, dict[str, dict[str, float | int]]]] = []
    for threshold in threshold_grid:
        details: dict[str, dict[str, float | int]] = {}
        valid = True
        minimum_coverage_count = np.iinfo(np.int64).max
        for name in names:
            score_array, error_array = arrays[name]
            mask = score_array <= threshold
            accepted = int(mask.sum())
            if accepted < min_accepted:
                valid = False
                break
            failures = int(error_array[mask].sum())
            upper = clopper_pearson_upper(failures, accepted, corrected_delta)
            details[name] = {
                "accepted": accepted,
                "errors": failures,
                "empirical_risk": failures / accepted,
                "upper_bound": upper,
                "coverage": accepted / len(score_array),
            }
            minimum_coverage_count = min(minimum_coverage_count, accepted)
            if upper > alpha:
                valid = False
                break
        if valid:
            feasible.append((int(minimum_coverage_count), float(threshold), details))

    if not feasible:
        return PrecisionEnvelopeSelection(
            threshold=None,
            per_precision={},
            alpha=alpha,
            delta=delta,
            grid_size=int(threshold_grid.size),
            precision_count=len(names),
            status="no_feasible_threshold",
        )
    _, threshold, details = max(feasible, key=lambda item: (item[0], item[1]))
    return PrecisionEnvelopeSelection(
        threshold=threshold,
        per_precision=details,
        alpha=alpha,
        delta=delta,
        grid_size=int(threshold_grid.size),
        precision_count=len(names),
        status="selected",
    )


def select_fixed_sequence_precision_threshold(
    scores: Mapping[str, Sequence[float] | np.ndarray],
    errors: Mapping[str, Sequence[int | bool] | np.ndarray],
    *,
    alpha: float,
    delta: float = 0.10,
    grid: Iterable[float] | None = None,
    min_accepted: int = 5,
) -> FixedSequencePrecisionSelection:
    """Select a common threshold by conservative-to-aggressive fixed sequence.

    Every precision-specific upper bound must pass at a threshold. Testing then
    stops at the first failure, which controls selection of an unsafe rule at
    level ``delta`` when the sequence is fixed before labels are inspected.
    """

    names = sorted(scores)
    if not names or set(names) != set(errors):
        raise ValueError("scores and errors must have the same non-empty precision keys")
    if not (0.0 < alpha < 1.0 and 0.0 < delta < 1.0):
        raise ValueError("alpha and delta must be in (0, 1)")
    if min_accepted <= 0:
        raise ValueError("min_accepted must be positive")

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        score_array = np.asarray(scores[name], dtype=float)
        error_array = np.asarray(errors[name], dtype=int)
        if score_array.ndim != 1 or score_array.shape != error_array.shape:
            raise ValueError(f"invalid score/error shapes for precision {name}")
        if not np.isfinite(score_array).all():
            raise ValueError(f"scores must be finite for precision {name}")
        if np.any((error_array != 0) & (error_array != 1)):
            raise ValueError(f"errors must be binary for precision {name}")
        arrays[name] = (score_array, error_array)

    threshold_grid = np.asarray(
        list(grid) if grid is not None else np.linspace(0.0, 1.0, 101), dtype=float
    )
    if threshold_grid.ndim != 1 or threshold_grid.size == 0:
        raise ValueError("grid must contain at least one threshold")
    if not np.isfinite(threshold_grid).all() or np.any(np.diff(threshold_grid) < 0):
        raise ValueError("grid must be finite and ordered from low to high")

    selected_threshold: float | None = None
    selected_details: dict[str, dict[str, float | int]] = {}
    tested: list[dict] = []
    stopped_at: float | None = None
    eligible_count = 0
    for threshold in threshold_grid:
        details: dict[str, dict[str, float | int]] = {}
        for name in names:
            score_array, error_array = arrays[name]
            mask = score_array <= threshold
            accepted = int(mask.sum())
            failures = int(error_array[mask].sum()) if accepted else 0
            details[name] = {
                "accepted": accepted,
                "errors": failures,
                "empirical_risk": failures / accepted if accepted else 0.0,
                "upper_bound": clopper_pearson_upper(failures, accepted, delta),
                "coverage": accepted / len(score_array),
            }
        if any(details[name]["accepted"] < min_accepted for name in names):
            continue

        eligible_count += 1
        passed = all(details[name]["upper_bound"] <= alpha for name in names)
        tested.append(
            {
                "threshold": float(threshold),
                "passed": passed,
                "per_precision": details,
            }
        )
        if not passed:
            stopped_at = float(threshold)
            break
        selected_threshold = float(threshold)
        selected_details = details

    if selected_threshold is None:
        return FixedSequencePrecisionSelection(
            threshold=None,
            per_precision={},
            tested_thresholds=tested,
            stopped_at=stopped_at,
            alpha=alpha,
            delta=delta,
            grid_size=int(threshold_grid.size),
            precision_count=len(names),
            status=(
                "no_eligible_threshold"
                if eligible_count == 0
                else "no_feasible_threshold"
            ),
        )
    return FixedSequencePrecisionSelection(
        threshold=selected_threshold,
        per_precision=selected_details,
        tested_thresholds=tested,
        stopped_at=stopped_at,
        alpha=alpha,
        delta=delta,
        grid_size=int(threshold_grid.size),
        precision_count=len(names),
        status="selected",
    )
