from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from analyze_pilot import stratified_bootstrap_auc
from precisionprobe.evalplus_labels import evalplus_error
from precisionprobe.ranking import low_risk_membership


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_statistics(frame: pd.DataFrame, repeats: int = 10000) -> dict[str, Any]:
    q4_error = frame["q4_error"].to_numpy(dtype=int)
    q8_error = frame["q8_error"].to_numpy(dtype=int)
    q4_score = frame["q4_dsde"].to_numpy(dtype=float)
    q8_score = frame["q8_dsde"].to_numpy(dtype=float)
    task_ids = frame["task_id"].astype(str).to_numpy()

    def compute(indices: np.ndarray) -> tuple[float, float, float, float]:
        e4, e8 = q4_error[indices], q8_error[indices]
        s4, s8 = q4_score[indices], q8_score[indices]
        sampled_task_ids = task_ids[indices]
        accuracy_difference = float((1 - e4).mean() - (1 - e8).mean())
        auc_difference = float(roc_auc_score(e4, s4) - roc_auc_score(e8, s8))
        rho = float(spearmanr(s4, s8).statistic)
        decision_drift = float(
            np.mean(
                low_risk_membership(s4, sampled_task_ids)
                != low_risk_membership(s8, sampled_task_ids)
            )
        )
        return accuracy_difference, auc_difference, rho, decision_drift

    observed = compute(np.arange(len(frame)))
    rng = np.random.default_rng(20260811)
    samples = []
    for _ in range(repeats):
        indices = rng.integers(0, len(frame), size=len(frame))
        if len(np.unique(q4_error[indices])) < 2 or len(np.unique(q8_error[indices])) < 2:
            continue
        values = compute(indices)
        if np.isfinite(values).all():
            samples.append(values)
    draws = np.asarray(samples)
    return {
        "accuracy_difference_q4_minus_q8": observed[0],
        "accuracy_difference_ci90": [float(x) for x in np.quantile(draws[:, 0], [0.05, 0.95])],
        "auc_difference_q4_minus_q8": observed[1],
        "auc_difference_ci90": [float(x) for x in np.quantile(draws[:, 1], [0.05, 0.95])],
        "spearman": observed[2],
        "spearman_ci95": [float(x) for x in np.quantile(draws[:, 2], [0.025, 0.975])],
        "decision_drift": observed[3],
        "decision_drift_ci95": [float(x) for x in np.quantile(draws[:, 3], [0.025, 0.975])],
        "decision_drift_tie_rule": "score_then_task_id_then_draw_position",
        "bootstrap_repeats": len(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / args.run_dir
    frame = pd.DataFrame(read_jsonl(run_dir / "scores.jsonl"))
    evaluation = json.loads((root / args.evaluation).read_text(encoding="utf-8"))
    labels = {
        (row["task_id"], row["sample_id"].split("|")[-1].split("_")[0]): evalplus_error(row, "plus")
        for row in evaluation["results"]
        if row["sample_id"].endswith(("|q4_greedy", "|q8_greedy"))
    }
    for precision in ("q4", "q8"):
        frame[f"{precision}_error"] = frame["task_id"].map(lambda task: labels[(task, precision)])

    paired = paired_statistics(frame)
    payload: dict[str, Any] = {
        "n": len(frame),
        "q4_errors": int(frame["q4_error"].sum()),
        "q8_errors": int(frame["q8_error"].sum()),
        "label_disagreement": float((frame["q4_error"] != frame["q8_error"]).mean()),
        "paired": paired,
    }
    for precision in ("q4", "q8"):
        payload[f"{precision}_dsde"] = stratified_bootstrap_auc(
            frame[f"{precision}_error"].to_numpy(dtype=int),
            frame[f"{precision}_dsde"].to_numpy(dtype=float),
        )
    payload["gates"] = {
        "accuracy_equivalent": paired["accuracy_difference_ci90"][0] >= -0.03
        and paired["accuracy_difference_ci90"][1] <= 0.03,
        "auc_equivalent": paired["auc_difference_ci90"][0] >= -0.05
        and paired["auc_difference_ci90"][1] <= 0.05,
        "rank_portability_failure": paired["spearman_ci95"][1] < 0.85,
        "decision_drift": paired["decision_drift_ci95"][0] > 0.20,
        "both_informative": False,
    }
    payload["gates"]["both_informative"] = (
        payload["q4_dsde"]["ci95"][0] > 0.50 and payload["q8_dsde"]["ci95"][0] > 0.50
    )
    frame.to_csv(run_dir / "joined.csv", index=False)
    (run_dir / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
