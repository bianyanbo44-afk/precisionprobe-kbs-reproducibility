from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from precisionprobe.ranking import low_risk_membership


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed_scores(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    keyed = {row["task_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate task_id in {path}")
    return keyed


def greedy_hashes(run_dir: Path, precision: str) -> dict[str, str]:
    return {
        row["task_id"]: row["solution_sha256"]
        for row in read_jsonl(run_dir / "generations" / f"{precision}.jsonl")
        if row["kind"] == "greedy" and int(row["seed"]) == 0
    }


def summarize_greedy_reproducibility(
    primary: dict[str, str],
    retest: dict[str, str],
    expected_task_ids: list[str],
) -> dict[str, Any]:
    expected = set(expected_task_ids)
    observed = set(primary) | set(retest)
    mismatched = sorted(task for task in observed if primary.get(task) != retest.get(task))
    missing_primary = sorted(expected - set(primary))
    missing_retest = sorted(expected - set(retest))
    unexpected = sorted(observed - expected)
    complete_match = not (mismatched or missing_primary or missing_retest or unexpected)
    return {
        "expected_tasks": len(expected),
        "primary_tasks": len(primary),
        "retest_tasks": len(retest),
        "matching": sum(
            task in primary and task in retest and primary[task] == retest[task]
            for task in expected
        ),
        "missing_primary_task_ids": missing_primary,
        "missing_retest_task_ids": missing_retest,
        "unexpected_task_ids": unexpected,
        "mismatched_task_ids": mismatched,
        "complete_match": complete_match,
    }


def statistics(
    arrays: dict[str, np.ndarray],
    task_ids: np.ndarray,
) -> dict[str, float]:
    if any(len(values) != len(task_ids) for values in arrays.values()):
        raise ValueError("all score arrays must align with task_ids")
    rho_within_q4 = float(spearmanr(arrays["q4_a"], arrays["q4_b"]).statistic)
    rho_within_q8 = float(spearmanr(arrays["q8_a"], arrays["q8_b"]).statistic)
    rho_cross_a = float(spearmanr(arrays["q4_a"], arrays["q8_a"]).statistic)
    rho_cross_b = float(spearmanr(arrays["q4_b"], arrays["q8_b"]).statistic)
    masks = {
        name: low_risk_membership(values, task_ids)
        for name, values in arrays.items()
    }
    drift_within_q4 = float(np.mean(masks["q4_a"] != masks["q4_b"]))
    drift_within_q8 = float(np.mean(masks["q8_a"] != masks["q8_b"]))
    drift_cross_a = float(np.mean(masks["q4_a"] != masks["q8_a"]))
    drift_cross_b = float(np.mean(masks["q4_b"] != masks["q8_b"]))
    return {
        "rho_within_q4": rho_within_q4,
        "rho_within_q8": rho_within_q8,
        "rho_cross_a": rho_cross_a,
        "rho_cross_b": rho_cross_b,
        "rho_contrast_within_minus_cross": (
            (rho_within_q4 + rho_within_q8) - (rho_cross_a + rho_cross_b)
        )
        / 2,
        "drift_within_q4": drift_within_q4,
        "drift_within_q8": drift_within_q8,
        "drift_cross_a": drift_cross_a,
        "drift_cross_b": drift_cross_b,
        "drift_contrast_cross_minus_within": (
            (drift_cross_a + drift_cross_b) - (drift_within_q4 + drift_within_q8)
        )
        / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-run", required=True)
    parser.add_argument("--retest-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    primary = root / args.primary_run
    retest = root / args.retest_run
    left = keyed_scores(primary / "scores.jsonl")
    right = keyed_scores(retest / "scores.jsonl")
    if set(left) != set(right):
        raise ValueError("primary and retest task sets differ")
    task_ids = sorted(left)
    arrays = {
        "q4_a": np.asarray([left[task]["q4_dsde"] for task in task_ids], dtype=float),
        "q8_a": np.asarray([left[task]["q8_dsde"] for task in task_ids], dtype=float),
        "q4_b": np.asarray([right[task]["q4_dsde"] for task in task_ids], dtype=float),
        "q8_b": np.asarray([right[task]["q8_dsde"] for task in task_ids], dtype=float),
    }
    task_id_array = np.asarray(task_ids)
    observed = statistics(arrays, task_id_array)
    rng = np.random.default_rng(20260812)
    draws: list[dict[str, float]] = []
    for _ in range(10000):
        indices = rng.integers(0, len(task_ids), size=len(task_ids))
        draw = statistics(
            {name: values[indices] for name, values in arrays.items()},
            task_id_array[indices],
        )
        if np.isfinite(list(draw.values())).all():
            draws.append(draw)

    intervals = {
        key: [
            float(np.quantile([draw[key] for draw in draws], 0.025)),
            float(np.quantile([draw[key] for draw in draws], 0.975)),
        ]
        for key in observed
    }
    greedy = {}
    for precision in ("q4", "q8"):
        first = greedy_hashes(primary, precision)
        second = greedy_hashes(retest, precision)
        greedy[precision] = summarize_greedy_reproducibility(
            first,
            second,
            task_ids,
        )
    attribution_eligible = all(item["complete_match"] for item in greedy.values())
    rho_interval_support = intervals["rho_contrast_within_minus_cross"][0] > 0
    drift_interval_support = intervals["drift_contrast_cross_minus_within"][0] > 0
    payload = {
        "n": len(task_ids),
        "primary_run": args.primary_run,
        "retest_run": args.retest_run,
        "greedy_generation_reproducibility": greedy,
        "attribution_eligible": attribution_eligible,
        "observed": observed,
        "ci95": intervals,
        "bootstrap_repeats": len(draws),
        "decision_drift_tie_rule": "score_then_task_id_then_draw_position",
        "rho_interval_support": rho_interval_support,
        "drift_interval_support": drift_interval_support,
        "rho_excess_instability_supported": attribution_eligible
        and rho_interval_support,
        "drift_excess_instability_supported": attribution_eligible
        and drift_interval_support,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
