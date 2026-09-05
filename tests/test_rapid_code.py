from __future__ import annotations

import pandas as pd

from precisionprobe.rapid_code import (
    RouteTemplate,
    StateSpec,
    deterministic_buckets,
    deterministic_reference_mask,
    simulate_route,
    state_measurements,
)


def test_reference_split_is_deterministic_and_nontrivial():
    task_ids = [f"task-{index}" for index in range(30)]
    first = deterministic_reference_mask(task_ids, panel="panel", salt="salt")
    second = deterministic_reference_mask(task_ids, panel="panel", salt="salt")
    assert first.tolist() == second.tolist()
    assert 0 < int(first.sum()) < len(task_ids)


def test_three_way_buckets_are_deterministic_and_exhaustive():
    task_ids = [f"task-{index}" for index in range(90)]
    buckets = deterministic_buckets(task_ids, panel="panel", salt="salt")
    assert set(buckets.tolist()) == {0, 1, 2}
    assert len(buckets) == len(task_ids)


def test_state_measurements_use_exact_generation_prefix_and_scaled_execution():
    row = {
        "q4_observations": [
            [{"status": "ok", "value": 1}] * 8,
            [{"status": "ok", "value": 2}] * 8,
            [{"status": "ok", "value": 1}] * 8,
            [{"status": "ok", "value": 1}] * 8,
            [{"status": "ok", "value": 1}] * 8,
        ],
        "q4_generation_seconds": [1, 2, 3, 4, 5],
        "q4_execution_seconds": 8.0,
    }
    result = state_measurements(row, StateSpec("q4", 3, 4))
    assert result["generation_seconds"] == 6.0
    assert result["estimated_execution_seconds"] == 8.0 * 12 / 40
    assert result["estimated_total_seconds"] == 8.4
    assert result["realized_probes"] == 4.0


def test_one_candidate_state_supports_low_cost_confidence_stage():
    row = {
        "q4_observations": [
            [{"status": "ok", "value": 1}] * 4 for _ in range(5)
        ],
        "q4_generation_seconds": [1, 2, 3, 4, 5],
        "q4_execution_seconds": 5.0,
    }
    result = state_measurements(row, StateSpec("q4", 1, 1))
    assert result["dsde"] == 0.0
    assert result["generation_seconds"] == 1.0
    assert result["estimated_execution_seconds"] == 5.0 / 20.0


def test_state_measurements_cap_requested_probes_at_available_count():
    row = {
        "q4_observations": [
            [{"status": "ok", "value": 1}] * 2 for _ in range(5)
        ],
        "q4_generation_seconds": [1, 1, 1, 1, 1],
        "q4_execution_seconds": 2.0,
    }
    result = state_measurements(row, StateSpec("q4", 2, 8))
    assert result["realized_probes"] == 2.0
    assert result["estimated_execution_seconds"] == 2.0 * 4 / 10


def _route_rows() -> pd.DataFrame:
    records = []
    for state, percentile, cost, error, dsde in (
        ("q4_c3_p2", 0.80, 3.0, 1, 0.4),
        ("q8_c3_p2", 0.20, 4.0, 0, 0.1),
        ("q4_c5_p8", 0.30, 8.0, 1, 0.2),
    ):
        precision = state[:2]
        records.append(
            {
                "panel": "p",
                "role": "development",
                "task_id": "t",
                "state": state,
                "precision": precision,
                "error": error,
                "percentile": percentile,
                "dsde": dsde,
                "generation_seconds": cost * 0.8,
                "estimated_execution_seconds": cost * 0.2,
                "estimated_total_seconds": cost,
                "work_units": cost * 10,
            }
        )
    records.append(
        {
            "panel": "p",
            "role": "development",
            "task_id": "t",
            "state": "q4_c5_p8",
            "precision": "q4",
            "error": 1,
            "percentile": 0.30,
            "dsde": 0.2,
            "generation_seconds": 6.4,
            "estimated_execution_seconds": 1.6,
            "estimated_total_seconds": 8.0,
            "work_units": 80.0,
        }
    )
    return pd.DataFrame(records).drop_duplicates(subset=["state"], keep="last")


def test_route_switches_precision_and_adds_cross_precision_cost():
    template = RouteTemplate(
        "switch", (StateSpec("q4", 3, 2), StateSpec("q8", 3, 2))
    )
    result = simulate_route(_route_rows(), template, threshold=0.25)
    assert result["accepted"] == 1
    assert result["selected_precision"] == "q8"
    assert result["estimated_total_seconds"] == 7.0
    assert result["error_if_accepted"] == 0


def test_nested_same_precision_cost_is_not_double_counted():
    template = RouteTemplate(
        "nested", (StateSpec("q4", 3, 2), StateSpec("q4", 5, 8))
    )
    result = simulate_route(_route_rows(), template, threshold=0.35)
    assert result["accepted"] == 1
    assert result["estimated_total_seconds"] == 8.0


def test_current_stage_can_switch_from_token_to_behavior_score():
    rows = _route_rows()
    rows.loc[rows["state"] == "q4_c3_p2", "token_percentile"] = 0.8
    rows.loc[rows["state"] == "q4_c5_p8", "behavior_percentile"] = 0.3
    template = RouteTemplate(
        "token_then_behavior",
        (StateSpec("q4", 3, 2), StateSpec("q4", 5, 8)),
        score_kinds=("token", "behavior"),
        selection_rule="current",
    )
    result = simulate_route(rows, template, threshold=0.35)
    assert result["accepted"] == 1
    assert result["selected_state"] == "q4_c5_p8"
    assert result["selected_score_kind"] == "behavior"
