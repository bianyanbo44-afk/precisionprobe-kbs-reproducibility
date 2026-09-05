from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from precisionprobe.data import load_benchmark, prompt_derived_probe_inputs
from precisionprobe.evalplus_labels import evalplus_error
from precisionprobe.execution import run_candidate
from precisionprobe.scoring import xpbd_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_inputs(values: list[Any], limit: int = 8) -> list[list[Any]]:
    result: list[list[Any]] = []
    for value in values:
        item = list(value) if isinstance(value, tuple) else value
        if not isinstance(item, list):
            item = [item]
        if item not in result:
            result.append(item)
        if len(result) == limit:
            break
    return result


def deterministic_sample(values: list[Any], task_id: str, limit: int = 8) -> list[list[Any]]:
    keyed = []
    for index, value in enumerate(values):
        digest = hashlib.sha256(f"{task_id}|{index}".encode()).hexdigest()
        keyed.append((digest, value))
    return normalize_inputs([value for _, value in sorted(keyed)], limit=limit)


def auc_ci(labels: np.ndarray, scores: np.ndarray, seed: int = 20260811) -> dict[str, Any]:
    observed = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    samples = []
    for _ in range(5000):
        indices = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        samples.append(roc_auc_score(labels[indices], scores[indices]))
    return {
        "auc": observed,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/probe_source_audit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    problems = load_benchmark("humaneval")

    generations: dict[str, dict[str, dict[str, Any]]] = {"greedy": {}, "stochastic": {}}
    for run_name, stochastic_name in (
        ("pilot_qwen15b", "q4.jsonl"),
        ("confirmation_humaneval", "q4_stochastic.jsonl"),
    ):
        generation_dir = root / "runs" / run_name / "generations"
        for row in read_jsonl(generation_dir / "q4_greedy.jsonl"):
            generations["greedy"][row["task_id"]] = row
        for row in read_jsonl(generation_dir / stochastic_name):
            if row.get("seed", 11) == 11:
                generations["stochastic"][row["task_id"]] = row

    labels: dict[str, int] = {}
    evaluation_files = {
        "pilot_qwen15b": "greedy_evalplus_results.json",
        "confirmation_humaneval": "evalplus_results.json",
    }
    for run_name, filename in evaluation_files.items():
        evaluation = json.loads((root / "runs" / run_name / filename).read_text(encoding="utf-8"))
        for row in evaluation["results"]:
            if row["sample_id"].endswith("|q4_greedy"):
                labels[row["task_id"]] = evalplus_error(row, "plus")

    rows = []
    for index, task_id in enumerate(sorted(labels), start=1):
        problem = problems[task_id]
        sources = {
            "prompt": prompt_derived_probe_inputs(problem, max_inputs=8),
            "base_first": normalize_inputs(problem.get("base_input", [])),
            "plus_first": normalize_inputs(problem.get("plus_input", [])),
            "plus_hashed": deterministic_sample(problem.get("plus_input", []), task_id),
        }
        row: dict[str, Any] = {"task_id": task_id, "error": labels[task_id]}
        for name, probes in sources.items():
            row[f"{name}_count"] = len(probes)
            if not probes:
                row[f"{name}_score"] = np.nan
                continue
            left = run_candidate(generations["greedy"][task_id]["solution"], problem["entry_point"], probes)
            right = run_candidate(generations["stochastic"][task_id]["solution"], problem["entry_point"], probes)
            row[f"{name}_score"] = xpbd_score(left["observations"], right["observations"])
        rows.append(row)
        print(f"audit {index}/{len(labels)} {task_id}")

    frame = pd.DataFrame(rows)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "probe_source_scores.csv", index=False)
    summary: dict[str, Any] = {}
    for name in ("prompt", "base_first", "plus_first", "plus_hashed"):
        subset = frame.dropna(subset=[f"{name}_score"])
        if subset["error"].nunique() < 2:
            continue
        summary[name] = {
            "n": len(subset),
            "errors": int(subset["error"].sum()),
            "nonzero": float((subset[f"{name}_score"] > 0).mean()),
            **auc_ci(
                subset["error"].to_numpy(dtype=int),
                subset[f"{name}_score"].to_numpy(dtype=float),
            ),
        }
    (output_dir / "probe_source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
