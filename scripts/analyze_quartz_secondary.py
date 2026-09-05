from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

try:
    from analyze_pilot import stratified_bootstrap_auc
except ModuleNotFoundError:
    from scripts.analyze_pilot import stratified_bootstrap_auc
from precisionprobe.evalplus_labels import evalplus_error
from precisionprobe.ranking import low_risk_membership, random_tie_drift_sensitivity
from precisionprobe.scoring import (
    dominant_semantic_distance_entropy,
    exact_disagreement_score,
    exact_dominant_semantic_distance_entropy,
    exact_semantic_distance_entropy,
    semantic_distance_entropy,
    xpbd_score,
)


PRECISIONS = ("q4", "q8")
METRICS = ("dsde", "exact_dsde", "sde", "exact_sde", "ast_mean")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_labels(evaluation: dict[str, Any]) -> dict[tuple[str, str, str], int]:
    labels: dict[tuple[str, str, str], int] = {}
    for row in evaluation["results"]:
        suffix = row["sample_id"].split("|")[-1]
        if suffix not in {"q4_greedy", "q8_greedy"}:
            continue
        precision = suffix.split("_")[0]
        for suite in ("base", "augmented", "plus"):
            labels[(row["task_id"], precision, suite)] = evalplus_error(row, suite)
    return labels


def aurc(errors: np.ndarray, scores: np.ndarray, task_ids: np.ndarray) -> float:
    positions = np.arange(len(scores))
    order = np.lexsort((positions, task_ids.astype(str), scores))
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return float(cumulative_risk.mean())


def metric_aurocs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for suite in ("plus", "augmented", "base"):
        for precision in PRECISIONS:
            labels = frame[f"{precision}_{suite}_error"].to_numpy(dtype=int)
            for metric in METRICS:
                scores = frame[f"{precision}_{metric}"].to_numpy(dtype=float)
                estimate = stratified_bootstrap_auc(labels, scores)
                records.append(
                    {
                        "suite": suite,
                        "precision": precision,
                        "metric": metric,
                        "n": len(frame),
                        "errors": int(labels.sum()),
                        "auc": estimate["auc"],
                        "ci95_low": estimate["ci95"][0],
                        "ci95_high": estimate["ci95"][1],
                    }
                )
    return records


def risk_coverage(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_ids = frame["task_id"].astype(str).to_numpy()
    for precision in PRECISIONS:
        errors = frame[f"{precision}_plus_error"].to_numpy(dtype=int)
        scores = frame[f"{precision}_dsde"].to_numpy(dtype=float)
        for coverage in np.linspace(0.1, 1.0, 10):
            count = max(1, int(round(coverage * len(frame))))
            selected = low_risk_membership(scores, task_ids, count)
            records.append(
                {
                    "precision": precision,
                    "coverage": count / len(frame),
                    "accepted": count,
                    "selective_error": float(errors[selected].mean()),
                    "selective_accuracy": float(1 - errors[selected].mean()),
                    "tie_rule": "score_then_task_id_then_draw_position",
                }
            )
    return records


def portability_by_coverage(frame: pd.DataFrame) -> list[dict[str, Any]]:
    q4 = frame["q4_dsde"].to_numpy(dtype=float)
    q8 = frame["q8_dsde"].to_numpy(dtype=float)
    task_ids = frame["task_id"].astype(str).to_numpy()
    records: list[dict[str, Any]] = []
    for coverage in np.linspace(0.1, 0.9, 9):
        count = max(1, int(round(coverage * len(frame))))
        left = low_risk_membership(q4, task_ids, count)
        right = low_risk_membership(q8, task_ids, count)
        intersection = int(np.sum(left & right))
        union = int(np.sum(left | right))
        records.append(
            {
                "coverage": count / len(frame),
                "set_size": count,
                "intersection": intersection,
                "jaccard": intersection / union,
                "membership_drift": float(np.mean(left != right)),
                "tie_rule": "score_then_task_id_then_draw_position",
            }
        )
    return records


def budget_sensitivity(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate_budget in (2, 3, 4, 5):
        for probe_budget in (1, 2, 4, 8):
            for precision in PRECISIONS:
                labels = frame[f"{precision}_plus_error"].to_numpy(dtype=int)
                executions = frame[f"{precision}_observations"].tolist()
                reduced = [
                    [candidate[:probe_budget] for candidate in task[:candidate_budget]]
                    for task in executions
                ]
                scores = {
                    "dsde": [dominant_semantic_distance_entropy(task[0], task) for task in reduced],
                    "sde": [semantic_distance_entropy(task) for task in reduced],
                    "exact_dsde": [
                        exact_dominant_semantic_distance_entropy(task[0], task) for task in reduced
                    ],
                    "exact_sde": [exact_semantic_distance_entropy(task) for task in reduced],
                }
                for metric, values in scores.items():
                    records.append(
                        {
                            "precision": precision,
                            "candidate_budget": candidate_budget,
                            "probe_budget": probe_budget,
                            "metric": metric,
                            "auc": float(roc_auc_score(labels, values)),
                            "nonzero": float(np.mean(np.asarray(values) > 0)),
                        }
                    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / args.run_dir
    output_dir = root / args.output_dir if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(read_jsonl(run_dir / "scores.jsonl"))
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / args.evaluation).read_text(encoding="utf-8"))
    labels = load_labels(evaluation)
    for precision in PRECISIONS:
        for suite in ("base", "augmented", "plus"):
            frame[f"{precision}_{suite}_error"] = frame["task_id"].map(
                lambda task: labels[(task, precision, suite)]
            )
        if f"{precision}_exact_dsde" not in frame:
            frame[f"{precision}_exact_dsde"] = frame[f"{precision}_observations"].map(
                lambda task: exact_dominant_semantic_distance_entropy(task[0], task)
            )
        if f"{precision}_exact_sde" not in frame:
            frame[f"{precision}_exact_sde"] = frame[f"{precision}_observations"].map(
                exact_semantic_distance_entropy
            )

    frame["cross_xpbd"] = frame.apply(
        lambda row: xpbd_score(row["q4_observations"][0], row["q8_observations"][0]), axis=1
    )
    frame["cross_exact_disagreement"] = frame.apply(
        lambda row: exact_disagreement_score(
            row["q4_observations"][0], row["q8_observations"][0]
        ),
        axis=1,
    )
    frame["absolute_dsde_change"] = (frame["q4_dsde"] - frame["q8_dsde"]).abs()
    frame["plus_label_disagreement"] = frame["q4_plus_error"] != frame["q8_plus_error"]
    frame["augmented_label_disagreement"] = (
        frame["q4_augmented_error"] != frame["q8_augmented_error"]
    )

    metric_records = metric_aurocs(frame)
    risk_records = risk_coverage(frame)
    portability_records = portability_by_coverage(frame)
    budget_records = budget_sensitivity(frame)
    pd.DataFrame(metric_records).to_csv(output_dir / "metric_aurocs.csv", index=False)
    pd.DataFrame(risk_records).to_csv(output_dir / "risk_coverage.csv", index=False)
    pd.DataFrame(portability_records).to_csv(
        output_dir / "portability_by_coverage.csv", index=False
    )
    pd.DataFrame(budget_records).to_csv(output_dir / "budget_sensitivity.csv", index=False)

    transfer = {}
    for score_precision in PRECISIONS:
        for label_precision in PRECISIONS:
            transfer[f"{score_precision}_score_{label_precision}_label"] = float(
                roc_auc_score(
                    frame[f"{label_precision}_plus_error"], frame[f"{score_precision}_dsde"]
                )
            )
    source_summary = {}
    for source, group in frame.groupby("probe_source"):
        source_summary[source] = {"n": len(group)}
        for precision in PRECISIONS:
            labels_array = group[f"{precision}_plus_error"].to_numpy(dtype=int)
            if len(np.unique(labels_array)) == 2:
                source_summary[source][f"{precision}_auc"] = float(
                    roc_auc_score(labels_array, group[f"{precision}_dsde"])
                )

    q4_errors = frame["q4_plus_error"].to_numpy(dtype=int)
    q8_errors = frame["q8_plus_error"].to_numpy(dtype=int)
    q4_scores = frame["q4_dsde"].to_numpy(dtype=float)
    q8_scores = frame["q8_dsde"].to_numpy(dtype=float)
    task_ids = frame["task_id"].astype(str).to_numpy()
    q4_low = low_risk_membership(q4_scores, task_ids, len(frame) // 2)
    q8_low = low_risk_membership(q8_scores, task_ids, len(frame) // 2)
    tie_sensitivity = random_tie_drift_sensitivity(
        q4_scores,
        q8_scores,
        task_ids,
        count=len(frame) // 2,
    )
    (output_dir / "tie_break_sensitivity.json").write_text(
        json.dumps(tie_sensitivity, indent=2), encoding="utf-8"
    )

    augmented_sensitivity: dict[str, Any] = {
        "definition": "EvalPlus evaluator plus field alone; diagnostic, not official Plus correctness",
        "status": "post_unblinding_secondary_sensitivity",
    }
    for precision in PRECISIONS:
        augmented_errors = frame[f"{precision}_augmented_error"].to_numpy(dtype=int)
        augmented_sensitivity[precision] = {
            "errors": int(augmented_errors.sum()),
            "dsde": stratified_bootstrap_auc(
                augmented_errors, frame[f"{precision}_dsde"].to_numpy(dtype=float)
            ),
        }
    q4_augmented_auc = augmented_sensitivity["q4"]["dsde"]["auc"]
    q8_augmented_auc = augmented_sensitivity["q8"]["dsde"]["auc"]
    augmented_sensitivity["auc_difference_q4_minus_q8"] = (
        None
        if q4_augmented_auc is None or q8_augmented_auc is None
        else float(q4_augmented_auc - q8_augmented_auc)
    )
    official_label_disagreement = float(frame["plus_label_disagreement"].mean())
    payload = {
        "n": len(frame),
        "label_disagreement": {
            "base": float((frame["q4_base_error"] != frame["q8_base_error"]).mean()),
            "plus": official_label_disagreement,
            "plus_official_base_and_augmented": official_label_disagreement,
            "augmented_only": float(frame["augmented_label_disagreement"].mean()),
        },
        "augmented_only_sensitivity": augmented_sensitivity,
        "risk": {
            "q4_aurc": aurc(q4_errors, q4_scores, task_ids),
            "q8_aurc": aurc(q8_errors, q8_scores, task_ids),
            "shared_low_risk_failures": int(np.sum(q4_low & q8_low & (q4_errors == 1) & (q8_errors == 1))),
            "tie_rule": "score_then_task_id_then_draw_position",
        },
        "portability": {
            "decision_drift_tie_sensitivity": tie_sensitivity,
            "score_transfer_auc": transfer,
            "cross_xpbd_mean": float(frame["cross_xpbd"].mean()),
            "cross_exact_disagreement_mean": float(frame["cross_exact_disagreement"].mean()),
            "cross_behavior_identical_fraction": float(
                np.mean(frame["cross_exact_disagreement"] == 0)
            ),
            "cross_xpbd_vs_absolute_dsde_change_spearman": float(
                spearmanr(frame["cross_xpbd"], frame["absolute_dsde_change"]).statistic
            ),
        },
        "generation_seconds": {
            precision: {
                "mean_per_task_five_samples": float(
                    frame[f"{precision}_generation_seconds"].map(sum).mean()
                ),
                "median_per_task_five_samples": float(
                    frame[f"{precision}_generation_seconds"].map(sum).median()
                ),
            }
            for precision in PRECISIONS
            if f"{precision}_generation_seconds" in frame
        },
        "probe_execution_seconds": {
            precision: {
                "mean_per_task_five_by_probe_count": float(
                    frame[f"{precision}_execution_seconds"].mean()
                ),
                "median_per_task_five_by_probe_count": float(
                    frame[f"{precision}_execution_seconds"].median()
                ),
            }
            for precision in PRECISIONS
            if f"{precision}_execution_seconds" in frame
        },
        "model_artifacts": {
            precision: {
                "bytes": int(run_manifest["models"][precision]["bytes"]),
                "sha256": run_manifest["models"][precision]["sha256"],
            }
            for precision in PRECISIONS
        },
        "probe_source": source_summary,
        "artifacts": {
            "metric_aurocs": "metric_aurocs.csv",
            "risk_coverage": "risk_coverage.csv",
            "portability_by_coverage": "portability_by_coverage.csv",
            "budget_sensitivity": "budget_sensitivity.csv",
            "tie_break_sensitivity": "tie_break_sensitivity.json",
        },
    }
    frame.drop(columns=["q4_observations", "q8_observations"]).to_csv(
        output_dir / "secondary_joined.csv", index=False
    )
    (output_dir / "secondary_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
