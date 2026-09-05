from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from precisionprobe.evalplus_labels import evalplus_error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def status_name(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, tuple) and value:
        return str(value[0])
    return str(value)


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def stratified_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    repeats: int = 4000,
    seed: int = 20260811,
) -> dict[str, Any]:
    point = auc(labels, scores)
    if point is None:
        return {"auc": None, "ci95": [None, None], "valid_repeats": 0}
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(repeats):
        indices = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        values.append(roc_auc_score(labels[indices], scores[indices]))
    low, high = np.quantile(values, [0.025, 0.975])
    return {"auc": point, "ci95": [float(low), float(high)], "valid_repeats": len(values)}


def paired_auc_permutation(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    repeats: int = 5000,
    seed: int = 20260811,
) -> dict[str, Any]:
    left_auc = auc(labels, left)
    right_auc = auc(labels, right)
    if left_auc is None or right_auc is None:
        return {"difference": None, "p_two_sided": None}
    observed = left_auc - right_auc
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(repeats):
        swap = rng.integers(0, 2, size=len(labels), dtype=bool)
        perm_left = np.where(swap, right, left)
        perm_right = np.where(swap, left, right)
        difference = roc_auc_score(labels, perm_left) - roc_auc_score(labels, perm_right)
        if abs(difference) >= abs(observed) - 1e-12:
            extreme += 1
    return {"difference": float(observed), "p_two_sided": (extreme + 1) / (repeats + 1)}


def empirical_frontier(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for threshold in sorted(set(float(value) for value in scores)):
        accepted = scores <= threshold
        count = int(accepted.sum())
        if count == 0:
            continue
        rows.append(
            {
                "threshold": threshold,
                "coverage": count / len(scores),
                "selective_error": float(labels[accepted].mean()),
            }
        )
    return rows


def metrics_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["error"].to_numpy(dtype=int)
    metric_columns = [
        "xpbd_cross_q4_q8",
        "xpbd_same_q4",
        "ast_cross_q4_q8",
        "ast_same_q4",
        "q4_a_probe_failure_fraction",
    ]
    metrics = {
        column: stratified_bootstrap_auc(labels, frame[column].to_numpy(dtype=float))
        for column in metric_columns
    }
    metrics["cross_vs_same_q4"] = paired_auc_permutation(
        labels,
        frame["xpbd_cross_q4_q8"].to_numpy(dtype=float),
        frame["xpbd_same_q4"].to_numpy(dtype=float),
    )
    metrics["cross_frontier"] = empirical_frontier(
        labels, frame["xpbd_cross_q4_q8"].to_numpy(dtype=float)
    )
    metrics["same_q4_frontier"] = empirical_frontier(
        labels, frame["xpbd_same_q4"].to_numpy(dtype=float)
    )
    metrics["n"] = len(frame)
    metrics["errors"] = int(labels.sum())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/pilot_qwen15b")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--evaluation", default="evalplus_results.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / args.run_dir
    config = yaml.safe_load((project_root / args.config).read_text(encoding="utf-8"))
    scores = pd.DataFrame(read_jsonl(run_dir / "scores.jsonl"))
    evaluation = json.loads((run_dir / args.evaluation).read_text(encoding="utf-8"))
    target_results: dict[str, int] = {}
    for row in evaluation["results"]:
        if row["sample_id"].endswith("|q4_a"):
            target_results[row["task_id"]] = evalplus_error(row, "plus")
    scores["error"] = scores["task_id"].map(target_results)
    if scores["error"].isna().any():
        missing = scores.loc[scores["error"].isna(), "task_id"].tolist()
        raise KeyError(f"missing q4_a EvalPlus labels: {missing}")
    scores["error"] = scores["error"].astype(int)

    cross_behavior_variation = float((scores["xpbd_cross_q4_q8"] > 0).mean())
    cross_code_variation = float(scores["q4_cross_code_changed"].mean())
    same_behavior_variation = float((scores["xpbd_same_q4"] > 0).mean())
    full_metrics = metrics_for_frame(scores)
    clean = scores[scores["probe_source"] == "prompt_only"].copy()
    clean_metrics = metrics_for_frame(clean) if len(clean) >= 6 else {"n": len(clean), "status": "too_small"}

    cross_auc = full_metrics["xpbd_cross_q4_q8"]["auc"]
    same_auc = full_metrics["xpbd_same_q4"]["auc"]
    gain = None if cross_auc is None or same_auc is None else cross_auc - same_auc
    kill_reasons = []
    if max(cross_behavior_variation, cross_code_variation) < float(
        config["kill_gates"]["minimum_any_variation_fraction"]
    ):
        kill_reasons.append("cross_precision_variation_below_gate")
    if cross_auc is None or cross_auc <= 0.50:
        kill_reasons.append("cross_precision_auc_not_above_chance")
    if gain is None or gain < float(config["kill_gates"]["minimum_auc_gain"]):
        kill_reasons.append("cross_precision_does_not_beat_matched_q4_resampling")

    latency_cross = (
        scores["generation_elapsed"].map(lambda value: value["q4_a"] + value["q8_a"]).mean()
    )
    latency_same = (
        scores["generation_elapsed"].map(lambda value: value["q4_a"] + value["q4_b"]).mean()
    )
    payload = {
        "decision": "KILL_OR_PIVOT" if kill_reasons else "SCALE",
        "kill_reasons": kill_reasons,
        "variation": {
            "cross_behavior_fraction": cross_behavior_variation,
            "cross_code_fraction": cross_code_variation,
            "same_q4_behavior_fraction": same_behavior_variation,
        },
        "latency_seconds_mean": {
            "cross_q4_q8": float(latency_cross),
            "same_q4": float(latency_same),
            "ratio": float(latency_cross / latency_same) if latency_same else None,
        },
        "all_tasks": full_metrics,
        "prompt_only_subset": clean_metrics,
    }
    (run_dir / "pilot_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scores.to_csv(run_dir / "pilot_joined.csv", index=False)

    lines = [
        "# PrecisionProbe Pilot Analysis",
        "",
        f"**Decision:** {payload['decision']}",
        "",
        f"- Tasks: {full_metrics['n']} ({full_metrics['errors']} incorrect Q4 target programs)",
        f"- Cross-precision behavioral variation: {cross_behavior_variation:.1%}",
        f"- Cross-precision code variation: {cross_code_variation:.1%}",
        f"- Same-Q4 behavioral variation: {same_behavior_variation:.1%}",
        f"- XPBD cross AUROC: {cross_auc if cross_auc is not None else 'NA'}",
        f"- XPBD same-Q4 AUROC: {same_auc if same_auc is not None else 'NA'}",
        f"- Paired AUROC difference: {gain if gain is not None else 'NA'}",
        f"- Cross/same generation-latency ratio: {payload['latency_seconds_mean']['ratio']:.3f}",
        "",
        "## Gate failures",
        "",
    ]
    lines.extend([f"- {reason}" for reason in kill_reasons] or ["- None"])
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "This pilot is a mechanism screen, not the paper's confirmatory result. Bootstrap intervals are descriptive at this sample size. The predeclared decision is based on variation, point discrimination, and the matched-cost comparator; full risk-control claims require the larger held-out study.",
            "",
        ]
    )
    (run_dir / "pilot_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
