from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_pcqrc_extension import pcqrc_once  # noqa: E402


PRECISIONS = ("q4", "q8")
ALPHAS = (0.20, 0.40)
REPEATS = 100


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    q10, median, q90 = np.quantile(np.asarray(values, dtype=float), [0.10, 0.50, 0.90])
    return {"median": float(median), "q10": float(q10), "q90": float(q90)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["policy"]["selection"]["status"] == "selected"
    ]
    return {
        "repeats": len(rows),
        "selected_count": len(selected),
        "selection_rate": len(selected) / len(rows),
        "qualification_rate": float(np.mean([row["qualifies"] for row in rows])),
        "contradiction_rate": float(
            np.mean([row["material_contradiction"] for row in rows])
        ),
        "selected_split_qualification_rate": (
            float(np.mean([row["qualifies"] for row in selected])) if selected else None
        ),
        "threshold": quantiles(
            [float(row["policy"]["selection"]["threshold"]) for row in selected]
        ),
        "selected_split": {
            precision: {
                "coverage": quantiles(
                    [row["held_out"][precision]["coverage"] for row in selected]
                ),
                "empirical_risk": quantiles(
                    [
                        row["held_out"][precision]["empirical_risk"]
                        for row in selected
                        if row["held_out"][precision]["empirical_risk"] is not None
                    ]
                ),
            }
            for precision in PRECISIONS
        },
    }


def analyze_panel(frame: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for alpha in ALPHAS:
        rows = [
            pcqrc_once(
                frame,
                alpha=alpha,
                salt=f"pcqrc-robust-v1-{seed:03d}",
            )
            for seed in range(REPEATS)
        ]
        output[str(alpha)] = {"summary": summarize(rows), "rows": rows}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", default="configs/pcqrc_extension_registry.yaml"
    )
    parser.add_argument(
        "--output",
        default="phase4_extension/results/pcqrc_sensitivity_robustness.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="phase4_extension/results/pcqrc_sensitivity_robustness_summary.csv",
    )
    args = parser.parse_args()

    registry = yaml.safe_load((ROOT / args.registry).read_text(encoding="utf-8"))
    panels = {}
    summary_rows = []
    for name, panel in registry["panels"].items():
        frame = pd.read_csv(ROOT / panel["run_dir"] / "joined.csv")
        alpha_results = analyze_panel(frame)
        panels[name] = {
            "role": panel["role"],
            "model": panel["model"],
            "dataset": panel["dataset"],
            "n": len(frame),
            "alphas": alpha_results,
        }
        for alpha, result in alpha_results.items():
            row = result["summary"]
            summary_rows.append(
                {
                    "panel": name,
                    "role": panel["role"],
                    "alpha": float(alpha),
                    "repeats": row["repeats"],
                    "selected_count": row["selected_count"],
                    "selection_rate": row["selection_rate"],
                    "qualification_rate": row["qualification_rate"],
                    "contradiction_rate": row["contradiction_rate"],
                    "selected_split_qualification_rate": row[
                        "selected_split_qualification_rate"
                    ],
                }
            )

    payload = {
        "protocol_basis": "phase4_extension/locked_protocol.md",
        "split_family": "pcqrc-robust-v1-000 through pcqrc-robust-v1-099",
        "alpha_values": list(ALPHAS),
        "panels": panels,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(ROOT / args.summary_csv, index=False)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
