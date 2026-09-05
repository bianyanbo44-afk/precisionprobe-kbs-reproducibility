from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def low_risk_membership(
    scores: Sequence[float] | np.ndarray,
    task_ids: Sequence[Any] | np.ndarray,
    count: int | None = None,
    *,
    priorities: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Select the lowest scores with a reproducible secondary ordering.

    The confirmatory rule uses ascending task ID to break score ties. Bootstrap
    samples can contain repeated task IDs, so their draw position is a tertiary
    key. A caller may instead supply shared random priorities for a tie-rule
    sensitivity analysis; the same priorities must then be used for both
    precisions.
    """

    score_array = np.asarray(scores, dtype=float)
    id_array = np.asarray([str(task_id) for task_id in task_ids])
    if score_array.ndim != 1 or id_array.ndim != 1 or score_array.shape != id_array.shape:
        raise ValueError("scores and task_ids must be one-dimensional with identical shapes")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")

    selected_count = len(score_array) // 2 if count is None else int(count)
    if not 0 <= selected_count <= len(score_array):
        raise ValueError("count must be between zero and the number of scores")

    positions = np.arange(len(score_array))
    if priorities is None:
        order = np.lexsort((positions, id_array, score_array))
    else:
        priority_array = np.asarray(priorities, dtype=float)
        if priority_array.ndim != 1 or priority_array.shape != score_array.shape:
            raise ValueError("priorities must have the same one-dimensional shape as scores")
        if not np.isfinite(priority_array).all():
            raise ValueError("priorities must be finite")
        order = np.lexsort((positions, priority_array, score_array))

    membership = np.zeros(len(score_array), dtype=bool)
    membership[order[:selected_count]] = True
    return membership


def tie_cutoff_diagnostics(
    scores: Sequence[float] | np.ndarray,
    count: int,
) -> dict[str, int | float | None]:
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or not np.isfinite(score_array).all():
        raise ValueError("scores must be a finite one-dimensional array")
    selected_count = int(count)
    if not 0 <= selected_count <= len(score_array):
        raise ValueError("count must be between zero and the number of scores")
    if selected_count == 0:
        return {
            "cutoff_score": None,
            "strictly_below_cutoff": 0,
            "tied_at_cutoff": 0,
            "selected_from_cutoff_tie": 0,
        }

    cutoff = float(np.sort(score_array)[selected_count - 1])
    below = int(np.sum(score_array < cutoff))
    tied = int(np.sum(score_array == cutoff))
    return {
        "cutoff_score": cutoff,
        "strictly_below_cutoff": below,
        "tied_at_cutoff": tied,
        "selected_from_cutoff_tie": selected_count - below,
    }


def random_tie_drift_sensitivity(
    q4_scores: Sequence[float] | np.ndarray,
    q8_scores: Sequence[float] | np.ndarray,
    task_ids: Sequence[Any] | np.ndarray,
    *,
    count: int | None = None,
    repeats: int = 4000,
    seed: int = 20260811,
) -> dict[str, Any]:
    """Compare fixed-ID drift with shared random tie priorities.

    One random priority is drawn per task and reused at Q4 and Q8. This changes
    only ordering within equal-score blocks and does not create artificial drift
    by breaking an identical tie differently at the two precisions.
    """

    q4 = np.asarray(q4_scores, dtype=float)
    q8 = np.asarray(q8_scores, dtype=float)
    ids = np.asarray([str(task_id) for task_id in task_ids])
    if q4.shape != q8.shape or q4.shape != ids.shape or q4.ndim != 1:
        raise ValueError("q4_scores, q8_scores, and task_ids must have identical 1D shapes")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    selected_count = len(q4) // 2 if count is None else int(count)
    fixed_q4 = low_risk_membership(q4, ids, selected_count)
    fixed_q8 = low_risk_membership(q8, ids, selected_count)
    fixed_drift = float(np.mean(fixed_q4 != fixed_q8)) if len(q4) else 0.0

    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        shared_priorities = rng.random(len(q4))
        random_q4 = low_risk_membership(
            q4, ids, selected_count, priorities=shared_priorities
        )
        random_q8 = low_risk_membership(
            q8, ids, selected_count, priorities=shared_priorities
        )
        draws[index] = float(np.mean(random_q4 != random_q8)) if len(q4) else 0.0

    quantiles = np.quantile(draws, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {
        "n": len(q4),
        "selected_count": selected_count,
        "coverage": selected_count / len(q4) if len(q4) else 0.0,
        "fixed_tie_rule": "score_then_task_id_then_draw_position",
        "fixed_id_drift": fixed_drift,
        "random_tie_rule": "score_then_shared_random_task_priority_then_draw_position",
        "random_repeats": repeats,
        "random_seed": seed,
        "random_drift_mean": float(draws.mean()),
        "random_drift_sd": float(draws.std(ddof=1)) if repeats > 1 else 0.0,
        "random_drift_min": float(draws.min()),
        "random_drift_q025": float(quantiles[0]),
        "random_drift_q25": float(quantiles[1]),
        "random_drift_median": float(quantiles[2]),
        "random_drift_q75": float(quantiles[3]),
        "random_drift_q975": float(quantiles[4]),
        "random_drift_max": float(draws.max()),
        "random_fraction_at_least_0_20": float(np.mean(draws >= 0.20)),
        "q4_cutoff": tie_cutoff_diagnostics(q4, selected_count),
        "q8_cutoff": tie_cutoff_diagnostics(q8, selected_count),
    }
