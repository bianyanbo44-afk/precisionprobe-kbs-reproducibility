from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_status(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "empty"
    return str(value)


def evalplus_error(row: dict[str, Any]) -> int:
    # Official EvalPlus correctness requires both base and plus suites.
    return int(
        not (
            first_status(row["base"]) == "pass"
            and first_status(row["plus"]) == "pass"
        )
    )


def bootstrap_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    repeats: int = 4000,
    seed: int = 20260905,
) -> dict[str, Any]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"value": None, "ci95": [None, None], "valid_repeats": 0}
    point = float(metric(labels, scores))
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values: list[float] = []
    for _ in range(repeats):
        indices = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        values.append(float(metric(labels[indices], scores[indices])))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "value": point,
        "ci95": [float(low), float(high)],
        "valid_repeats": len(values),
    }


def paired_bootstrap_difference(
    left_labels: np.ndarray,
    left_scores: np.ndarray,
    right_labels: np.ndarray,
    right_scores: np.ndarray,
    *,
    repeats: int = 4000,
    seed: int = 20260905,
) -> dict[str, Any]:
    if len(np.unique(left_labels)) < 2 or len(np.unique(right_labels)) < 2:
        return {"difference": None, "ci95": [None, None], "valid_repeats": 0}
    left = float(roc_auc_score(left_labels, left_scores))
    right = float(roc_auc_score(right_labels, right_scores))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(left_labels), size=len(left_labels))
        if len(np.unique(left_labels[indices])) < 2 or len(np.unique(right_labels[indices])) < 2:
            continue
        values.append(
            float(
                roc_auc_score(left_labels[indices], left_scores[indices])
                - roc_auc_score(right_labels[indices], right_scores[indices])
            )
        )
    if not values:
        return {"difference": left - right, "ci95": [None, None], "valid_repeats": 0}
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "difference": left - right,
        "ci95": [float(low), float(high)],
        "valid_repeats": len(values),
    }


def risk_coverage(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    order = np.argsort(scores, kind="mergesort")
    ordered = labels[order]
    cumulative_errors = np.cumsum(ordered)
    n = len(labels)
    rows = []
    for accepted in range(1, n + 1):
        rows.append(
            {
                "accepted": accepted,
                "coverage": accepted / n,
                "risk": float(cumulative_errors[accepted - 1] / accepted),
                "threshold": float(scores[order[accepted - 1]]),
            }
        )
    return pd.DataFrame(rows)


def operating_points(curve: pd.DataFrame, targets: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in targets:
        feasible = curve[curve["risk"] <= target]
        if feasible.empty:
            result[str(target)] = None
            continue
        row = feasible.sort_values(["coverage", "risk"], ascending=[False, True]).iloc[0]
        result[str(target)] = {
            "target_risk": target,
            "coverage": float(row["coverage"]),
            "empirical_risk": float(row["risk"]),
            "threshold": float(row["threshold"]),
            "accepted": int(row["accepted"]),
        }
    return result


def finite(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Qwen3 multi-precision smoke/confirmation runs.")
    parser.add_argument("--run-dir", default="runs/rapid_qwen3_smoke")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    output_dir = ROOT / (args.output_dir or str(Path(args.run_dir) / "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(read_jsonl(run_dir / "scores.jsonl"))
    evaluation = read_jsonl(run_dir / "evalplus_results.json") if False else json.loads(
        (run_dir / "evalplus_results.json").read_text(encoding="utf-8")
    )
    label_rows = {
        (row["task_id"], row["sample_id"].split("|")[-1].split("_")[0]): evalplus_error(row)
        for row in evaluation["results"]
    }
    confidence = {
        (row["task_id"], row["precision"]): row
        for row in read_jsonl(run_dir / "token_confidence.jsonl")
        if row.get("status") == "matched"
    }
    for precision in ("q2", "q4", "q8"):
        frame[f"{precision}_error"] = [
            label_rows[(task, precision)] for task in frame["task_id"]
        ]
        frame[f"{precision}_mean_nll"] = [
            confidence.get((task, precision), {}).get("mean_nll", np.nan)
            for task in frame["task_id"]
        ]

    metric_names = ["dsde", "sde", "exact_dsde", "exact_sde", "mean_nll"]
    metrics: dict[str, Any] = {}
    curves: list[pd.DataFrame] = []
    for precision in ("q2", "q4", "q8"):
        labels = frame[f"{precision}_error"].to_numpy(dtype=int)
        metrics[precision] = {
            "n": int(len(labels)),
            "errors": int(labels.sum()),
            "error_rate": float(labels.mean()),
            "scores": {},
        }
        for metric in metric_names:
            column = f"{precision}_{metric}" if metric != "mean_nll" else f"{precision}_mean_nll"
            scores = frame[column].to_numpy(dtype=float)
            valid = np.isfinite(scores)
            metric_labels = labels[valid]
            metric_scores = scores[valid]
            entry = {
                "n": int(valid.sum()),
                "excluded_missing": int((~valid).sum()),
                "nonzero_rate": float((metric_scores > 0).mean()) if len(metric_scores) else None,
                "median": float(np.median(metric_scores)) if len(metric_scores) else None,
                "q25": float(np.quantile(metric_scores, 0.25)) if len(metric_scores) else None,
                "q75": float(np.quantile(metric_scores, 0.75)) if len(metric_scores) else None,
                "auroc": bootstrap_metric(metric_labels, metric_scores, roc_auc_score),
                "auprc": bootstrap_metric(metric_labels, metric_scores, average_precision_score),
            }
            curve = risk_coverage(metric_labels, metric_scores) if len(metric_scores) else pd.DataFrame()
            entry["risk_coverage"] = operating_points(curve) if len(curve) else {}
            metrics[precision]["scores"][metric] = entry
            curve.insert(0, "metric", metric)
            curve.insert(0, "precision", precision)
            curves.append(curve)

    pairwise: dict[str, Any] = {}
    for metric in metric_names:
        pairwise[metric] = {}
        for left, right in (("q2", "q4"), ("q2", "q8"), ("q4", "q8")):
            left_scores = frame[f"{left}_{metric}" if metric != "mean_nll" else f"{left}_mean_nll"].to_numpy(dtype=float)
            right_scores = frame[f"{right}_{metric}" if metric != "mean_nll" else f"{right}_mean_nll"].to_numpy(dtype=float)
            valid = np.isfinite(left_scores) & np.isfinite(right_scores)
            if int(valid.sum()) < 2:
                pairwise[metric][f"{left}_vs_{right}"] = {"n": int(valid.sum()), "spearman_rho": None, "p_value": None, "mean_abs_difference": None}
                continue
            corr = spearmanr(left_scores[valid], right_scores[valid])
            pairwise[metric][f"{left}_vs_{right}"] = {
                "n": int(valid.sum()),
                "spearman_rho": finite(float(corr.statistic)),
                "p_value": finite(float(corr.pvalue)),
                "mean_abs_difference": float(np.mean(np.abs(left_scores[valid] - right_scores[valid]))),
            }
    contrasts = {}
    for left, right in (("q2", "q4"), ("q2", "q8"), ("q4", "q8")):
        contrasts[f"{left}_minus_{right}"] = {}
        for metric in metric_names:
            column = f"{metric}" if metric != "mean_nll" else "mean_nll"
            left_scores = frame[f"{left}_{column}" if metric != "mean_nll" else f"{left}_mean_nll"].to_numpy(dtype=float)
            right_scores = frame[f"{right}_{column}" if metric != "mean_nll" else f"{right}_mean_nll"].to_numpy(dtype=float)
            valid = np.isfinite(left_scores) & np.isfinite(right_scores)
            contrasts[f"{left}_minus_{right}"][metric] = {
                "n": int(valid.sum()),
                **paired_bootstrap_difference(
                    frame.loc[valid, f"{left}_error"].to_numpy(dtype=int),
                    left_scores[valid],
                    frame.loc[valid, f"{right}_error"].to_numpy(dtype=int),
                    right_scores[valid],
                ),
            }

    static_audit = None
    static_path = run_dir / "generation_static_audit.json"
    if static_path.exists():
        static_audit = json.loads(static_path.read_text(encoding="utf-8"))
    payload = {
        "analysis_status": "SMOKE_FEASIBILITY_ONLY" if len(frame) < 60 else "CONFIRMATION_ANALYSIS",
        "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "n_tasks": int(len(frame)),
        "precisions": ["q2", "q4", "q8"],
        "label_definition": "EvalPlus official plus error: base and plus suites must both pass",
        "metrics": metrics,
        "pairwise_rank_and_scale": pairwise,
        "paired_auc_contrasts": contrasts,
        "label_disagreement": {
            f"{left}_vs_{right}": float(
                (frame[f"{left}_error"] != frame[f"{right}_error"]).mean()
            )
            for left, right in (("q2", "q4"), ("q2", "q8"), ("q4", "q8"))
        },
        "static_audit": static_audit,
        "token_confidence_coverage": {
            precision: int(frame[f"{precision}_mean_nll"].notna().sum())
            for precision in ("q2", "q4", "q8")
        },
        "interpretation_boundary": (
            "This run is a feasibility check and cannot establish a stable generalization claim."
            if len(frame) < 60
            else "This run is a confirmation panel; claims remain bounded to the frozen benchmark, model, and probe protocol."
        ),
    }
    frame.to_csv(output_dir / "joined_metrics.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output_dir / "risk_coverage.csv", index=False)
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=finite),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=finite))


if __name__ == "__main__":
    main()
