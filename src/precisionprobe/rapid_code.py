from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from precisionprobe.scoring import dominant_semantic_distance_entropy


PRECISIONS = ("q4", "q8")
CANDIDATE_BUDGETS = (1, 2, 3, 4, 5)
PROBE_BUDGETS = (1, 2, 4, 8)
SCORE_KINDS = ("behavior", "token")


@dataclass(frozen=True, order=True)
class StateSpec:
    precision: str
    candidates: int
    probes: int

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.candidates not in CANDIDATE_BUDGETS:
            raise ValueError(f"unsupported candidate budget: {self.candidates}")
        if self.probes not in PROBE_BUDGETS:
            raise ValueError(f"unsupported probe budget: {self.probes}")

    @property
    def key(self) -> str:
        return f"{self.precision}_c{self.candidates}_p{self.probes}"


@dataclass(frozen=True)
class RouteTemplate:
    name: str
    stages: tuple[StateSpec, ...]
    score_kinds: tuple[str, ...] = ()
    selection_rule: str = "min_acquired"

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("route template must contain at least one stage")
        if self.score_kinds and len(self.score_kinds) != len(self.stages):
            raise ValueError("score_kinds must be empty or match the number of stages")
        if any(score not in SCORE_KINDS for score in self.score_kinds):
            raise ValueError(f"unsupported score kind in {self.score_kinds}")
        if self.selection_rule not in {"min_acquired", "current"}:
            raise ValueError(f"unsupported selection rule: {self.selection_rule}")
        latest: dict[str, StateSpec] = {}
        for state in self.stages:
            previous = latest.get(state.precision)
            if previous is not None and (
                state.candidates < previous.candidates or state.probes < previous.probes
            ):
                raise ValueError(
                    f"non-monotone state sequence for {state.precision}: "
                    f"{previous.key} -> {state.key}"
                )
            latest[state.precision] = state


def all_states() -> tuple[StateSpec, ...]:
    return tuple(
        StateSpec(precision, candidates, probes)
        for precision in PRECISIONS
        for candidates in CANDIDATE_BUDGETS
        for probes in PROBE_BUDGETS
    )


def default_templates() -> tuple[RouteTemplate, ...]:
    q4 = lambda c, p: StateSpec("q4", c, p)
    q8 = lambda c, p: StateSpec("q8", c, p)
    return (
        RouteTemplate("fixed_q4_full", (q4(5, 8),)),
        RouteTemplate("fixed_q8_full", (q8(5, 8),)),
        RouteTemplate("q4_candidate_ladder", (q4(2, 4), q4(3, 4), q4(5, 8))),
        RouteTemplate("q4_probe_ladder", (q4(3, 2), q4(3, 4), q4(5, 8))),
        RouteTemplate("q4_fast_to_full", (q4(3, 2), q4(5, 8))),
        RouteTemplate("q8_fast_to_full", (q8(3, 2), q8(5, 8))),
        RouteTemplate("q4_fast_to_q8_full", (q4(3, 2), q8(5, 8))),
        RouteTemplate(
            "dual_fast_then_q4_full",
            (q4(3, 2), q8(3, 2), q4(5, 8)),
        ),
        RouteTemplate(
            "dual_fast_then_q8_full",
            (q4(3, 2), q8(3, 2), q8(5, 8)),
        ),
        RouteTemplate(
            "joint_evidence_ladder",
            (
                q4(3, 2),
                q4(4, 4),
                q8(3, 2),
                q8(4, 4),
                q4(5, 8),
                q8(5, 8),
            ),
        ),
        RouteTemplate(
            "q4_token_behavior_ladder",
            (q4(1, 1), q4(2, 2), q4(3, 4), q4(5, 8)),
            score_kinds=("token", "behavior", "behavior", "behavior"),
            selection_rule="current",
        ),
        RouteTemplate(
            "q4_token_candidate_ladder",
            (q4(1, 1), q4(2, 4), q4(3, 4), q4(5, 8)),
            score_kinds=("token", "behavior", "behavior", "behavior"),
            selection_rule="current",
        ),
        RouteTemplate(
            "q4_token_fast_to_full",
            (q4(1, 1), q4(2, 4), q4(5, 8)),
            score_kinds=("token", "behavior", "behavior"),
            selection_rule="current",
        ),
        RouteTemplate(
            "q4_token_dual_precision",
            (q4(1, 1), q8(1, 1), q4(2, 4), q4(5, 8)),
            score_kinds=("token", "token", "behavior", "behavior"),
            selection_rule="current",
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def deterministic_reference_mask(
    task_ids: Sequence[str], *, panel: str, salt: str, modulus: int = 3
) -> np.ndarray:
    return deterministic_buckets(
        task_ids, panel=panel, salt=salt, modulus=modulus
    ) == 0


def deterministic_buckets(
    task_ids: Sequence[str], *, panel: str, salt: str, modulus: int = 3
) -> np.ndarray:
    if modulus < 2:
        raise ValueError("modulus must be at least two")
    buckets = []
    for task_id in task_ids:
        payload = f"{salt}|{panel}|{task_id}".encode("utf-8")
        bucket = int(hashlib.sha256(payload).hexdigest()[:16], 16) % modulus
        buckets.append(bucket)
    return np.asarray(buckets, dtype=int)


def _coerce_sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("expected a list or a serialized list")


def state_measurements(row: Mapping[str, Any], state: StateSpec) -> dict[str, float]:
    observations = _coerce_sequence(row[f"{state.precision}_observations"])
    generation_seconds = [
        float(value) for value in _coerce_sequence(row[f"{state.precision}_generation_seconds"])
    ]
    if len(observations) < state.candidates or len(generation_seconds) < state.candidates:
        raise ValueError(f"insufficient candidates for {state.key}")
    available_probes = min(len(candidate) for candidate in observations[: state.candidates])
    realized_probes = min(state.probes, available_probes)
    if realized_probes <= 0:
        raise ValueError(f"no executable probes for {state.key}")

    prefix = [
        candidate[:realized_probes] for candidate in observations[: state.candidates]
    ]
    dsde = dominant_semantic_distance_entropy(prefix[0], prefix)
    execution_failure_fraction = float(
        np.mean(
            [
                observation.get("status") != "ok"
                for candidate in prefix
                for observation in candidate
            ]
        )
    )
    measured_generation = float(sum(generation_seconds[: state.candidates]))

    full_execution = float(row[f"{state.precision}_execution_seconds"])
    full_candidates = len(observations)
    full_probes = min(len(candidate) for candidate in observations)
    execution_fraction = (state.candidates * realized_probes) / (
        full_candidates * full_probes
    )
    estimated_execution = full_execution * execution_fraction
    return {
        "dsde": float(dsde),
        "execution_failure_fraction": execution_failure_fraction,
        "generation_seconds": measured_generation,
        "estimated_execution_seconds": float(estimated_execution),
        "estimated_total_seconds": float(measured_generation + estimated_execution),
        "work_units": float(
            state.candidates + state.candidates * realized_probes
        ),
        "realized_probes": float(realized_probes),
    }


def build_panel_state_table(
    *, panel: str, role: str, run_dir: Path, states: Iterable[StateSpec] | None = None
) -> pd.DataFrame:
    score_rows = read_jsonl(run_dir / "scores.jsonl")
    scores_by_task = {str(row["task_id"]): row for row in score_rows}
    confidence_rows = read_jsonl(run_dir / "token_confidence.jsonl")
    confidence_by_key = {
        (str(row["task_id"]), str(row["precision"])): row
        for row in confidence_rows
        if row.get("status") == "matched"
    }
    labels = pd.read_csv(run_dir / "joined.csv", usecols=["task_id", "q4_error", "q8_error"])
    if labels["task_id"].duplicated().any():
        raise ValueError(f"duplicate task labels in {run_dir}")
    if set(labels["task_id"]) != set(scores_by_task):
        raise ValueError(f"score/label task mismatch in {run_dir}")

    records: list[dict[str, Any]] = []
    requested_states = tuple(states or all_states())
    for label_row in labels.itertuples(index=False):
        source = scores_by_task[str(label_row.task_id)]
        for state in requested_states:
            values = state_measurements(source, state)
            confidence = confidence_by_key.get((str(label_row.task_id), state.precision))
            if confidence is None:
                raise ValueError(
                    f"missing matched token confidence for {label_row.task_id}/{state.precision}"
                )
            records.append(
                {
                    "panel": panel,
                    "role": role,
                    "task_id": str(label_row.task_id),
                    "state": state.key,
                    "precision": state.precision,
                    "candidates": state.candidates,
                    "probes": state.probes,
                    "error": int(getattr(label_row, f"{state.precision}_error")),
                    "token_uncertainty": float(confidence["mean_nll"]),
                    **values,
                }
            )
    return pd.DataFrame.from_records(records)


def add_reference_percentiles(
    table: pd.DataFrame, *, reference_task_ids: set[str]
) -> pd.DataFrame:
    result = table.copy()
    result["behavior_percentile"] = np.nan
    result["token_percentile"] = np.nan
    for state, indices in result.groupby("state").groups.items():
        state_rows = result.loc[indices]
        reference_rows = state_rows[state_rows["task_id"].isin(reference_task_ids)]
        if reference_rows.empty:
            raise ValueError(f"empty reference set for state {state}")
        for source, destination in (
            ("dsde", "behavior_percentile"),
            ("token_uncertainty", "token_percentile"),
        ):
            ordered = np.sort(reference_rows[source].to_numpy(dtype=float))
            values = state_rows[source].to_numpy(dtype=float)
            result.loc[indices, destination] = (
                np.searchsorted(ordered, values, side="right") / ordered.size
            )
    result["percentile"] = result["behavior_percentile"]
    return result


def _state_lookup(task_rows: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if task_rows["state"].duplicated().any():
        raise ValueError("duplicate state rows for a task")
    return {str(row.state): row._asdict() for row in task_rows.itertuples(index=False)}


def simulate_route(
    task_rows: pd.DataFrame, template: RouteTemplate, *, threshold: float
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    lookup = _state_lookup(task_rows)
    acquired: dict[str, Mapping[str, Any]] = {}
    available: list[Mapping[str, Any]] = []
    available_scores: list[tuple[Mapping[str, Any], str, float]] = []
    selected: Mapping[str, Any] | None = None
    selected_score_kind = "behavior"
    selected_score = float("nan")
    accepted = False
    stage_index = 0

    score_kinds = template.score_kinds or ("behavior",) * len(template.stages)
    for stage_index, (state, score_kind) in enumerate(
        zip(template.stages, score_kinds), start=1
    ):
        current = lookup[state.key]
        acquired[state.precision] = current
        available.append(current)
        score_column = (
            "behavior_percentile" if score_kind == "behavior" else "token_percentile"
        )
        if score_kind == "behavior":
            current_score = float(current.get(score_column, current["percentile"]))
        else:
            current_score = float(current[score_column])
        available_scores.append((current, score_kind, current_score))
        if template.selection_rule == "current":
            selected, selected_score_kind, selected_score = (
                current,
                score_kind,
                current_score,
            )
        else:
            selected, selected_score_kind, selected_score = min(
                available_scores,
                key=lambda item: (
                    item[2],
                    float(item[0]["estimated_total_seconds"]),
                    str(item[0]["precision"]),
                ),
            )
        if selected_score <= threshold:
            accepted = True
            break

    assert selected is not None
    total_cost = float(
        sum(float(row["estimated_total_seconds"]) for row in acquired.values())
    )
    generation_cost = float(
        sum(float(row["generation_seconds"]) for row in acquired.values())
    )
    execution_cost = float(
        sum(float(row["estimated_execution_seconds"]) for row in acquired.values())
    )
    work_units = float(sum(float(row["work_units"]) for row in acquired.values()))
    full_q4 = lookup[StateSpec("q4", 5, 8).key]
    full_q4_cost = float(full_q4["estimated_total_seconds"])
    routed_error = int(selected["error"])
    return {
        "panel": str(selected["panel"]),
        "role": str(selected["role"]),
        "task_id": str(selected["task_id"]),
        "policy": template.name,
        "threshold": float(threshold),
        "accepted": int(accepted),
        "error_if_accepted": int(routed_error if accepted else 0),
        "routed_error": routed_error,
        "selected_precision": str(selected["precision"]),
        "selected_state": str(selected["state"]),
        "selected_score_kind": selected_score_kind,
        "selected_percentile": selected_score,
        "selected_dsde": float(selected["dsde"]),
        "stable_wrong_if_accepted": int(
            accepted and routed_error == 1 and float(selected["dsde"]) == 0.0
        ),
        "stages_used": int(stage_index),
        "generation_seconds": generation_cost,
        "estimated_execution_seconds": execution_cost,
        "estimated_total_seconds": total_cost,
        "work_units": work_units,
        "cost_ratio_to_full_q4": total_cost / full_q4_cost,
    }


def evaluate_template(
    table: pd.DataFrame,
    template: RouteTemplate,
    *,
    threshold: float,
    evaluation_task_ids: set[str],
) -> pd.DataFrame:
    subset = table[table["task_id"].isin(evaluation_task_ids)]
    records = [
        simulate_route(group, template, threshold=threshold)
        for _, group in subset.groupby("task_id", sort=True)
    ]
    return pd.DataFrame.from_records(records)


def summarize_routes(routes: pd.DataFrame) -> dict[str, float | int]:
    accepted = int(routes["accepted"].sum())
    errors = int(routes["error_if_accepted"].sum())
    return {
        "tasks": int(len(routes)),
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(routes) if len(routes) else 0.0,
        "empirical_risk": errors / accepted if accepted else 1.0,
        "mean_generation_seconds": float(routes["generation_seconds"].mean()),
        "mean_estimated_execution_seconds": float(
            routes["estimated_execution_seconds"].mean()
        ),
        "mean_estimated_total_seconds": float(
            routes["estimated_total_seconds"].mean()
        ),
        "mean_work_units": float(routes["work_units"].mean()),
        "mean_cost_ratio_to_full_q4": float(
            routes["cost_ratio_to_full_q4"].mean()
        ),
        "accepted_stable_wrong": int(routes["stable_wrong_if_accepted"].sum()),
    }
