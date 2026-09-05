from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def low_risk_set(scores: np.ndarray, count: int) -> set[int]:
    return set(np.lexsort((np.arange(len(scores)), scores))[:count].tolist())


def main() -> None:
    # Thirty correct and ten incorrect tasks.  Both scores rank every error
    # above every correct task, but reverse task order within each label block.
    errors = np.concatenate([np.zeros(30, dtype=int), np.ones(10, dtype=int)])
    q4 = np.arange(1, 41, dtype=float)
    q8 = np.concatenate([np.arange(30, 0, -1), np.arange(40, 30, -1)]).astype(float)
    q4_low = low_risk_set(q4, 20)
    q8_low = low_risk_set(q8, 20)
    payload = {
        "n": len(errors),
        "accuracy_q4": float(1 - errors.mean()),
        "accuracy_q8": float(1 - errors.mean()),
        "auc_q4": float(roc_auc_score(errors, q4)),
        "auc_q8": float(roc_auc_score(errors, q8)),
        "spearman": float(spearmanr(q4, q8).statistic),
        "low_risk_50_membership_drift": len(q4_low ^ q8_low) / len(errors),
        "q4_low_risk_indices": sorted(q4_low),
        "q8_low_risk_indices": sorted(q8_low),
    }
    assert payload["auc_q4"] == payload["auc_q8"] == 1.0
    assert payload["spearman"] < 0.85
    assert payload["low_risk_50_membership_drift"] == 0.5
    output = Path(__file__).resolve().parents[1] / "runs" / "auc_portability_counterexample.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
