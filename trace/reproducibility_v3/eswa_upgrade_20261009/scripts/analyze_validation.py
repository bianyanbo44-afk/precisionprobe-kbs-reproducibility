"""Analyze the complete six-condition panel under the local locked protocol.

No model-specific task exclusions or favorable-cell selection are permitted.
The task, shared across six conditions, is the primary bootstrap unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_retrospective import (
    bootstrap_comparison, digest, expected_review, policy_evaluation, read_jsonl, tie_aware_aurc,
)

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
OUT = BASE / "analysis/validation"
SCORES = (
    "dsde_1", "dsde_2", "dsde_4", "exact_4", "cluster_entropy",
    "greedy_failure", "mean_nll", "mean_topk_entropy", "public_failure",
    "failure_then_nll", "public_then_nll", "public_then_dsde",
)
PRIMARY_LEFT = "dsde_4"
PRIMARY_RIGHT = "public_then_nll"


def verify_statement_amendment(run):
    """Verify the transparent v1-to-v2 correction without executing programs."""
    receipt_path = run / "statement_correction_receipt.json"
    if not receipt_path.exists():
        raise ValueError("statement-aware evaluation receipt is required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    functional_path = run / "functional_correction_receipt.json"
    prior_comparator = (BASE / "scripts/statement_comparators_v2.py"
                        if functional_path.exists() else BASE / "scripts/statement_comparators.py")
    hashes = {
        "initial_lock_sha256": BASE / "protocol/validation_lock.json",
        "amendment_06_sha256": BASE / "protocol/amendment_06_statement_comparators.md",
        "comparator_script_sha256": prior_comparator,
        "reevaluation_script_sha256": BASE / "scripts/reevaluate_statement_cases.py",
        "original_evaluator_sha256": BASE / "scripts/evaluate_validation.py",
    }
    for key, path in hashes.items():
        if digest(path) != receipt[key]:
            raise ValueError("statement amendment source mismatch: " + key)
    if receipt["assessed_condition_task_pairs"] != 48 or not receipt["applied"]:
        raise ValueError("incomplete statement amendment")
    allowed = {"public_test_passes", "public_failure", "error", "private_tests_checked",
               "private_first_failure", "private_evaluation_seconds", "public_then_nll", "public_then_dsde"}
    details = {(r["condition"], r["task_id"]): r for r in receipt["details"]}
    if len(details) != 48:
        raise ValueError("duplicate or incomplete correction assessments")
    for item in receipt["initial_records"]:
        condition = item["condition"]
        initial = run / "evaluations_v1_exact_match" / (condition + ".jsonl")
        current = run / "evaluations" / (condition + ".jsonl")
        if digest(initial) != item["initial_evaluation_sha256"]:
            raise ValueError("preserved initial evaluation changed")
        if digest(run / "generations" / (condition + ".jsonl")) != item["generation_sha256"]:
            raise ValueError("generation changed during statement correction")
        revised = next(r for r in receipt["revised_records"] if r["condition"] == condition)
        if digest(current) != revised["sha256"]:
            raise ValueError("revised evaluation changed after correction receipt")
        original_rows, current_rows = read_jsonl(initial), read_jsonl(current)
        if [r["task_id"] for r in original_rows] != [r["task_id"] for r in current_rows]:
            raise ValueError("task order changed during correction")
        for before, after in zip(original_rows, current_rows):
            key = (condition, before["task_id"])
            if key not in details:
                if before != after:
                    raise ValueError("unaffected task changed")
                continue
            detail = details[key]
            if any(after[k] != v for k, v in before.items() if k not in allowed):
                raise ValueError("score inputs or acquisition times changed")
            for stage, record in (("before", before), ("after", after)):
                if any(record[k] != v for k, v in detail[stage].items()):
                    raise ValueError("correction receipt contradicts evaluation")
            if not after["error"] and after["private_tests_checked"] != after["private_test_count"]:
                raise ValueError("amended passing target lacks complete private tests")
    result = {"version": receipt["version"], "receipt_sha256": digest(receipt_path),
            "amendment_06_sha256": receipt["amendment_06_sha256"],
            "comparator_script_sha256": receipt["comparator_script_sha256"],
            "changed_error_labels": receipt["changed_error_labels"],
            "changed_public_failure_statuses": receipt["changed_public_failure_statuses"]}
    if functional_path.exists():
        amendment = json.loads(functional_path.read_text(encoding="utf-8"))
        expected = {
            "previous_statement_receipt_sha256": receipt_path,
            "previous_comparator_script_sha256": prior_comparator,
            "comparator_script_sha256": BASE / "scripts/statement_comparators.py",
            "rejudgement_script_sha256": BASE / "scripts/rejudge_functional_tolerance.py",
            "functional_inventory_sha256": BASE / "analysis/validation/functional_oracle_inventory.json",
            "amendment_07_sha256": BASE / "protocol/amendment_07_functional_tolerance_and_decision_reporting.md",
            "initial_lock_sha256": BASE / "protocol/validation_lock.json",
        }
        for key, path in expected.items():
            if digest(path) != amendment[key]:
                raise ValueError("functional amendment source mismatch: " + key)
        if not amendment["applied"] or amendment["assessed_condition_task_pairs"] != 6:
            raise ValueError("incomplete functional amendment")
        for detail in amendment["details"]:
            current = run / "evaluations" / (detail["condition"] + ".jsonl")
            if digest(current) != detail["evaluation_sha256"]:
                raise ValueError("functional amendment evaluation mismatch")
            target = next(r for r in read_jsonl(current) if r["task_id"] == detail["task_id"])
            if (target["solution_sha256"] != detail["solution_sha256"]
                    or target["error"] != detail["after_error"]
                    or target["public_test_passes"] != detail["after_public_passes"]):
                raise ValueError("functional amendment contradicts retained target")
        result.update(version=amendment["version"], functional_receipt_sha256=digest(functional_path),
                      amendment_07_sha256=amendment["amendment_07_sha256"],
                      functional_labels_changed=amendment["changed_error_labels"])
    return result


def load_complete():
    config = json.loads((BASE / "protocol/validation_config.json").read_text())
    lock = json.loads((BASE / "protocol/validation_lock.json").read_text())
    for relative, expected in lock["files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("locked source or data changed: " + relative)
    questions = read_jsonl(BASE / "data/livecodebench/questions.jsonl")
    task_ids = {q["task_id"] for q in questions}
    if len(task_ids) != 175:
        raise ValueError("validation requires all 175 unique tasks")
    run = BASE / config["run_dir"]
    verify_statement_amendment(run)
    frames, manifest = [], []
    for model in config["models"]:
        condition = model["condition"]
        paths = {}
        for kind in ("generation", "evaluation"):
            path = run / (kind + "s") / (condition + ".jsonl")
            receipt_path = run / (condition + "_" + kind + "_complete.json")
            receipt = json.loads(receipt_path.read_text())
            actual = digest(path)
            if actual != receipt["sha256"]:
                raise ValueError("completed output hash mismatch: " + str(path))
            paths[kind] = path
            manifest.append({"condition": condition, "kind": kind,
                             "relative_path": str(path.relative_to(BASE)),
                             "sha256": actual, "records": receipt["records"]})
        generations = read_jsonl(paths["generation"])
        keys = [(r["task_id"], r["seed"]) for r in generations]
        expected_keys = {(t, s) for t in task_ids for s in (0, 11, 29, 47, 71)}
        if len(keys) != 875 or set(keys) != expected_keys:
            raise ValueError("incomplete or duplicate generation keys: " + condition)
        target = {r["task_id"]: r for r in generations if r["seed"] == 0}
        frame = pd.DataFrame(read_jsonl(paths["evaluation"]))
        if len(frame) != 175 or set(frame.task_id) != task_ids:
            raise ValueError("incomplete or duplicate evaluation tasks: " + condition)
        if not (frame.condition == condition).all():
            raise ValueError("wrong condition label")
        if not np.isfinite(frame[list(SCORES)].to_numpy(float)).all():
            raise ValueError("missing or non-finite scores")
        for row in frame.to_dict("records"):
            generated = target[row["task_id"]]
            source_hash = hashlib.sha256(generated["solution"].encode()).hexdigest()
            if source_hash != row["solution_sha256"] or source_hash != generated["solution_sha256"]:
                raise ValueError("target-code join mismatch")
            if row["mean_nll"] != generated["token_confidence"]["mean_nll"]:
                raise ValueError("token-confidence join mismatch")
            if len(row["observations"]) != 5 or any(len(x) != row["probe_count"] for x in row["observations"]):
                raise ValueError("wrong number of public execution observations")
            expected_error = int(not all(row["public_test_passes"]) or row["private_first_failure"] is not None)
            if row["error"] != expected_error:
                raise ValueError("functional-label mismatch")
            if not row["error"] and row["private_tests_checked"] != row["private_test_count"]:
                raise ValueError("passing target did not run every private test")
            if row["private_test_count"] <= 0 or row["probe_count"] <= 0:
                raise ValueError("missing tests")
        frames.append(frame.sort_values("task_id").reset_index(drop=True))
    return pd.concat(frames, ignore_index=True), manifest


def primary_comparison(frame, repeats=4000, seed=20261009):
    conditions = sorted(frame.condition.unique())
    ids = sorted(frame.task_id.unique())
    cells = [frame[frame.condition == c].set_index("task_id").loc[ids] for c in conditions]
    yy = np.stack([c.error.to_numpy() for c in cells])
    ll = np.stack([c[PRIMARY_LEFT].to_numpy() for c in cells])
    rr = np.stack([c[PRIMARY_RIGHT].to_numpy() for c in cells])

    def differences(index):
        return np.array([
            expected_review(y[index], left[index], .25)["residual_error_per_task"]
            - expected_review(y[index], right[index], .25)["residual_error_per_task"]
            for y, left, right in zip(yy, ll, rr)
        ])

    point = differences(np.arange(len(ids)))
    rng = np.random.default_rng(seed)
    draws = np.stack([differences(rng.integers(0, len(ids), len(ids))) for _ in range(repeats)])
    cell_rows = []
    for j, condition in enumerate(conditions):
        left = expected_review(yy[j], ll[j], .25)
        right = expected_review(yy[j], rr[j], .25)
        cell_rows.append({
            "condition": condition, "paired_tasks": len(ids),
            "left_residual_error_per_task": left["residual_error_per_task"],
            "right_residual_error_per_task": right["residual_error_per_task"],
            "difference": float(point[j]),
            "ci95": np.quantile(draws[:, j], [.025, .975]).tolist(),
            "inference_scope": "prespecified cell estimate, no across-cell multiplicity adjustment",
        })
    return {
        "left": PRIMARY_LEFT, "right": PRIMARY_RIGHT,
        "endpoint": "residual functional errors per incoming task at 25% review capacity",
        "direction": "negative favors DSDE4", "bootstrap_replicates": repeats,
        "seed": seed, "sampling_unit": "same resampled task indices in all six conditions",
        "macro": {"difference": float(point.mean()),
                  "left_residual_error_per_task": float(np.mean([x["left_residual_error_per_task"] for x in cell_rows])),
                  "right_residual_error_per_task": float(np.mean([x["right_residual_error_per_task"] for x in cell_rows])),
                  "ci95": np.quantile(draws.mean(axis=1), [.025, .975]).tolist()},
        "cells": cell_rows,
    }, draws


def ranking_tables(frame):
    rows, subgroup = [], []
    for condition, cell in frame.groupby("condition", sort=True):
        y = cell.error.to_numpy()
        for score in SCORES:
            values = cell[score].to_numpy()
            for budget in (.10, .25, .50):
                rows.append({
                    "condition": condition, "model": cell.model.iloc[0],
                    "precision": cell.precision.iloc[0], "score": score,
                    "n": len(cell), "errors": int(y.sum()), "budget": budget,
                    "auroc": float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None,
                    "aurc": tie_aware_aurc(y, values),
                    **expected_review(y, values, budget),
                })
        for field in ("difficulty", "interface"):
            for value, subset in cell.groupby(field, sort=True):
                for score in SCORES:
                    subgroup.append({"condition": condition, "field": field, "value": value,
                                     "n": len(subset), "errors": int(subset.error.sum()), "score": score,
                                     **expected_review(subset.error, subset[score], .25)})
    return pd.DataFrame(rows), pd.DataFrame(subgroup)


def cost_tables(frame):
    rows = []
    for condition, cell in frame.groupby("condition", sort=True):
        for score in SCORES:
            alternatives = int(score[-1]) if score in ("dsde_1", "dsde_2", "dsde_4") else (4 if score in ("exact_4", "cluster_entropy", "public_then_dsde") else 0)
            if alternatives:
                gen = cell[f"generation_seconds_{alternatives}"].to_numpy()
                execution = cell[f"execution_seconds_{alternatives}"].to_numpy()
            else:
                gen = cell.greedy_generation_seconds.to_numpy()
                execution = np.zeros(len(cell)) if score in ("mean_nll", "mean_topk_entropy") else cell.greedy_execution_seconds.to_numpy()
            total = gen + execution
            metric = expected_review(cell.error, cell[score], .25)
            rows.append({"condition": condition, "score": score, "alternatives": alternatives,
                         "n": len(cell), "mean_generation_seconds": float(gen.mean()),
                         "mean_public_execution_seconds": float(execution.mean()),
                         "mean_total_seconds": float(total.mean()), "median_total_seconds": float(np.median(total)),
                         "p95_total_seconds": float(np.quantile(total, .95)),
                         "mean_total_over_mean_greedy": float(total.mean()/cell.greedy_generation_seconds.mean()),
                         **metric})
    return pd.DataFrame(rows)


def public_pass_diagnostic(frame):
    metrics, contrasts = [], []
    for condition, cell in frame.groupby("condition", sort=True):
        subset = cell[cell.public_failure == 0]
        count = len(subset)
        contrast = {"condition": condition, "public_passing_tasks": count,
                    "errors_among_public_passing": int(subset.error.sum()),
                    "public_failing_tasks": len(cell) - count,
                    "public_failures_fill_primary_queue": len(cell) - count >= int(np.ceil(.25*len(cell))),
                    "status": "secondary diagnostic specified in Amendment 02"}
        if count:
            for score in SCORES:
                values, y = subset[score].to_numpy(), subset.error.to_numpy()
                metrics.append({"condition": condition, "score": score, "n": count,
                                "errors": int(y.sum()),
                                "auroc": float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None,
                                **expected_review(y, values, .25)})
            contrast.update(bootstrap_comparison(subset.error, subset.dsde_4, subset.mean_nll, repeats=2000))
        contrasts.append(contrast)
    return pd.DataFrame(metrics), contrasts


def calibration_tables(frame):
    rows = []
    # The count-screened sequential selector is exploratory only (Amendment 01).
    for panel, panel_frame in frame.groupby("panel", sort=True):
        for repetition in range(30):
            salt = "eswa-20261009-split-" + str(repetition)
            for score in ("dsde_4", "dsde_1", "mean_nll", "failure_then_nll", "public_then_nll", "public_then_dsde"):
                for coordinate in ("conditioned", "pooled", "independent"):
                    for selector in ("bonferroni", "fixed_sequence"):
                        for alpha in (.10, .20, .30, .40):
                            rows.extend({"panel": panel, "repetition": repetition, **r}
                                        for r in policy_evaluation(panel_frame, score, coordinate, selector, alpha, salt))
        print("calibration complete", panel, flush=True)
    policies = pd.DataFrame(rows)
    summary = policies.groupby(["panel", "precision", "score", "coordinate", "selector", "alpha"]).agg(
        selection_rate=("selected", "mean"), mean_coverage=("coverage", "mean"),
        mean_risk_when_accepting=("risk", "mean"), unconditional_target_met=("test_target_met", "mean"),
    ).reset_index()
    return policies, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-policies", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, manifest = load_complete()
    frame.to_json(OUT / "task_features.jsonl", orient="records", lines=True, double_precision=15)
    scalar = frame.drop(columns=["observations", "probe_execution_seconds", "public_test_passes", "private_first_failure"])
    scalar.to_csv(OUT / "task_features.csv", index=False)
    primary, draws = primary_comparison(frame)
    (OUT / "primary_comparison.json").write_text(json.dumps(primary, indent=2), encoding="utf-8")
    np.save(OUT / "primary_bootstrap_differences.npy", draws)
    metrics, subgroup = ranking_tables(frame)
    metrics.to_csv(OUT / "ranking_review_metrics.csv", index=False)
    subgroup.to_csv(OUT / "subgroup_metrics.csv", index=False)
    cost_tables(frame).to_csv(OUT / "cost_frontier.csv", index=False)
    public_metrics, public_contrasts = public_pass_diagnostic(frame)
    public_metrics.to_csv(OUT / "public_pass_metrics.csv", index=False)
    (OUT / "public_pass_comparisons.json").write_text(json.dumps(public_contrasts, indent=2), encoding="utf-8")
    if not args.skip_policies:
        policies, summary = calibration_tables(frame)
        policies.to_csv(OUT / "policy_evaluations.csv", index=False)
        summary.to_csv(OUT / "policy_summary.csv", index=False)
    frame.groupby(["condition", "difficulty", "interface"], sort=True).agg(
        tasks=("error", "size"), errors=("error", "sum"),
        public_failures=("public_failure", lambda x: int((x > 0).sum())),
        truncations=("target_finish_reason", lambda x: int((x == "length").sum())),
    ).reset_index().to_csv(OUT / "panel_inventory.csv", index=False)
    (OUT / "analysis_manifest.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(__file__), "calibration_script_sha256": digest(BASE / "scripts/analyze_retrospective.py"),
        "lock_sha256": digest(BASE / "protocol/validation_lock.json"),
        "amendment_01_sha256": digest(BASE / "protocol/amendment_01_inference_scope.md"),
        "amendment_02_sha256": digest(BASE / "protocol/amendment_02_public_pass_diagnostic.md"),
        "amendment_03_sha256": digest(BASE / "protocol/amendment_03_system_restart.md"),
        "amendment_04_sha256": digest(BASE / "protocol/amendment_04_transport_retry.md"),
        "amendment_05_sha256": digest(BASE / "protocol/amendment_05_machine_sleep.md"),
        "statement_correction": verify_statement_amendment(BASE / json.loads(
            (BASE / "protocol/validation_config.json").read_text())["run_dir"]),
        "input_files": manifest, "conditions": 6, "tasks_per_condition": 175,
        "calibration_complete": not args.skip_policies,
    }, indent=2), encoding="utf-8")
    print(json.dumps(primary["macro"], indent=2), flush=True)


if __name__ == "__main__":
    main()
