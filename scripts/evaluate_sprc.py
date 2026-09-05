from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from precisionprobe.risk_control import (
    apply_threshold,
    select_precision_envelope_threshold,
    select_risk_controlling_threshold,
)


PRECISIONS = ("q4", "q8")
GRID = np.linspace(0.0, 1.0, 41)
DELTA = 0.10
MIN_ACCEPTED = 20


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = frame.assign(
        _split_key=frame["task_id"].map(
            lambda task: hashlib.sha256(f"20260811|{task}".encode("utf-8")).hexdigest()
        )
    ).sort_values("_split_key")
    cut = (len(keyed) + 1) // 2
    return keyed.iloc[:cut].copy(), keyed.iloc[cut:].copy()


def held_out(frame: pd.DataFrame, precision: str, threshold: float | None) -> dict[str, Any]:
    accepted = apply_threshold(frame[f"{precision}_dsde"], threshold)
    count = int(accepted.sum())
    errors = int(frame.loc[accepted, f"{precision}_error"].sum()) if count else 0
    return {
        "threshold": threshold,
        "accepted": count,
        "coverage": count / len(frame),
        "errors": errors,
        "empirical_risk": errors / count if count else None,
    }


def always_accept(frame: pd.DataFrame, precision: str) -> dict[str, Any]:
    count = len(frame)
    errors = int(frame[f"{precision}_error"].sum())
    return {
        "accepted": count,
        "coverage": 1.0 if count else 0.0,
        "errors": errors,
        "empirical_risk": errors / count if count else None,
    }


def hash_random_matched(frame: pd.DataFrame, precision: str, count: int) -> dict[str, Any]:
    """Deterministic label-independent random-order baseline at fixed coverage."""

    count = min(max(int(count), 0), len(frame))
    keyed = frame.assign(
        _random_key=frame["task_id"].map(
            lambda task: hashlib.sha256(
                f"random-baseline|20260811|{precision}|{task}".encode("utf-8")
            ).hexdigest()
        )
    ).sort_values("_random_key")
    selected = keyed.iloc[:count]
    errors = int(selected[f"{precision}_error"].sum()) if count else 0
    return {
        "accepted": count,
        "coverage": count / len(frame) if len(frame) else 0.0,
        "errors": errors,
        "empirical_risk": errors / count if count else None,
    }


def evaluate_alpha(calibration: pd.DataFrame, test: pd.DataFrame, alpha: float) -> dict[str, Any]:
    scores = {precision: calibration[f"{precision}_dsde"] for precision in PRECISIONS}
    errors = {precision: calibration[f"{precision}_error"] for precision in PRECISIONS}
    envelope = select_precision_envelope_threshold(
        scores,
        errors,
        alpha=alpha,
        delta=DELTA,
        grid=GRID,
        min_accepted=MIN_ACCEPTED,
    )
    separate = {
        precision: select_risk_controlling_threshold(
            scores[precision],
            errors[precision],
            alpha=alpha,
            delta=DELTA,
            grid=GRID,
            min_accepted=MIN_ACCEPTED,
        )
        for precision in PRECISIONS
    }
    q8_transfer = select_risk_controlling_threshold(
        scores["q8"],
        errors["q8"],
        alpha=alpha,
        delta=DELTA,
        grid=GRID,
        min_accepted=MIN_ACCEPTED,
    )
    pooled = select_risk_controlling_threshold(
        np.concatenate([scores[precision].to_numpy() for precision in PRECISIONS]),
        np.concatenate([errors[precision].to_numpy() for precision in PRECISIONS]),
        alpha=alpha,
        delta=DELTA,
        grid=GRID,
        min_accepted=MIN_ACCEPTED,
    )
    sprc_test = {
        precision: held_out(test, precision, envelope.threshold) for precision in PRECISIONS
    }

    return {
        "alpha": alpha,
        "sprc": {
            "calibration": envelope.to_dict(),
            "test": sprc_test,
        },
        "separate": {
            precision: {
                "calibration": separate[precision].to_dict(),
                "test": held_out(test, precision, separate[precision].threshold),
            }
            for precision in PRECISIONS
        },
        "q8_transfer": {
            "calibration": q8_transfer.to_dict(),
            "test": {
                precision: held_out(test, precision, q8_transfer.threshold)
                for precision in PRECISIONS
            },
        },
        "naive_pooled": {
            "calibration": pooled.to_dict(),
            "test": {
                precision: held_out(test, precision, pooled.threshold)
                for precision in PRECISIONS
            },
            "warning": "Paired precision rows are dependent; pooled binomial bounds are descriptive only.",
        },
        "always_accept": {
            "test": {precision: always_accept(test, precision) for precision in PRECISIONS}
        },
        "hash_random_matched_to_sprc": {
            "selection_rule": "ascending SHA256('random-baseline|20260811|' + precision + '|' + task_id)",
            "test": {
                precision: hash_random_matched(
                    test, precision, sprc_test[precision]["accepted"]
                )
                for precision in PRECISIONS
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / args.run_dir
    frame = pd.read_csv(run_dir / "joined.csv")
    calibration, test = split_frame(frame)
    results = {str(alpha): evaluate_alpha(calibration, test, alpha) for alpha in (0.20, 0.30, 0.40)}
    primary = results["0.3"]["sprc"]
    primary_test = primary["test"]
    promotion = (
        primary["calibration"]["status"] == "selected"
        and all(primary_test[p]["coverage"] >= 0.10 for p in PRECISIONS)
        and all(
            primary_test[p]["empirical_risk"] is not None
            and primary_test[p]["empirical_risk"] <= 0.30
            for p in PRECISIONS
        )
    )
    payload = {
        "split_rule": "SHA256('20260811|' + task_id), first half calibration",
        "n": len(frame),
        "calibration_n": len(calibration),
        "test_n": len(test),
        "delta": DELTA,
        "grid": GRID.tolist(),
        "min_accepted": MIN_ACCEPTED,
        "operating_points": results,
        "cell_level_promotion_criteria_met": promotion,
    }
    (run_dir / "sprc_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    calibration[["task_id"]].to_csv(run_dir / "sprc_calibration_tasks.csv", index=False)
    test[["task_id"]].to_csv(run_dir / "sprc_test_tasks.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
