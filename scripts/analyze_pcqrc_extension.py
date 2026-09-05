from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from analyze_pilot import stratified_bootstrap_auc
    from analyze_quartz_secondary import aurc
except ModuleNotFoundError:
    from scripts.analyze_pilot import stratified_bootstrap_auc
    from scripts.analyze_quartz_secondary import aurc
from precisionprobe.precision_calibration import (
    apply_split_precision_quantile_policy,
    apply_precision_quantile_policy,
    empirical_percentiles,
    fit_split_precision_quantile_policy,
    fit_precision_quantile_policy,
)
from precisionprobe.risk_control import (
    apply_threshold,
    select_fixed_sequence_precision_threshold,
    select_precision_envelope_threshold,
    select_risk_controlling_threshold,
)


PRECISIONS = ("q4", "q8")
GRID = np.linspace(0.0, 1.0, 41)
PRIMARY_ALPHA = 0.30
DELTA = 0.10
MIN_ACCEPTED = 20
FUSION_FEATURES = (
    "dsde",
    "sde",
    "ast_mean",
    "distinct_ratio",
    "mean_nll",
)
METRICS = (
    "dsde",
    "sde",
    "exact_dsde",
    "exact_sde",
    "ast_mean",
    "distinct_ratio",
    "mean_nll",
    "worst_decile_nll",
    "mean_topk_entropy",
    "margin_uncertainty",
    "fusion_oof",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def hash_split(frame: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = frame.assign(
        _split_key=frame["task_id"].map(
            lambda task: hashlib.sha256(f"{salt}|{task}".encode("utf-8")).hexdigest()
        )
    ).sort_values("_split_key")
    cut = (len(keyed) + 1) // 2
    return keyed.iloc[:cut].copy(), keyed.iloc[cut:].copy()


def cross_fitted_fusion(frame: pd.DataFrame, precision: str) -> np.ndarray:
    columns = [f"{precision}_{feature}" for feature in FUSION_FEATURES]
    features = frame[columns].to_numpy(dtype=float)
    labels = frame[f"{precision}_error"].to_numpy(dtype=int)
    if not np.isfinite(features).all():
        raise ValueError(f"nonfinite fusion features for {precision}")
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=20260826,
    )
    total = np.zeros(len(frame), dtype=float)
    counts = np.zeros(len(frame), dtype=int)
    for train, test in splitter.split(features, labels):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        )
        model.fit(features[train], labels[train])
        total[test] += model.predict_proba(features[test])[:, 1]
        counts[test] += 1
    if np.any(counts != 10):
        raise RuntimeError("unexpected repeated-cross-fitting coverage")
    return total / counts


def load_panel(root: Path, panel: dict[str, Any]) -> pd.DataFrame:
    run_dir = root / panel["run_dir"]
    frame = pd.read_csv(run_dir / "joined.csv")
    token_rows = pd.DataFrame(read_jsonl(run_dir / "token_confidence.jsonl"))
    if token_rows.empty or set(token_rows["status"]) != {"matched"}:
        raise ValueError(f"token confidence is incomplete or mismatched in {run_dir}")
    expected = len(frame) * len(PRECISIONS)
    if len(token_rows) != expected or token_rows[["task_id", "precision"]].duplicated().any():
        raise ValueError(f"unexpected token-confidence key set in {run_dir}")
    for precision in PRECISIONS:
        subset = token_rows[token_rows["precision"] == precision].set_index("task_id")
        for metric in (
            "mean_nll",
            "worst_decile_nll",
            "mean_topk_entropy",
            "mean_top1_margin",
        ):
            frame[f"{precision}_{metric}"] = frame["task_id"].map(subset[metric])
        frame[f"{precision}_margin_uncertainty"] = -frame[f"{precision}_mean_top1_margin"]
        frame[f"{precision}_distinct_ratio"] = (
            frame[f"{precision}_distinct_programs"] - 1
        ) / 4.0
        frame[f"{precision}_fusion_oof"] = cross_fitted_fusion(frame, precision)
    return frame


def held_out_summary(
    test: pd.DataFrame,
    precision: str,
    accepted: np.ndarray,
) -> dict[str, Any]:
    count = int(accepted.sum())
    errors = int(test.loc[accepted, f"{precision}_error"].sum()) if count else 0
    return {
        "accepted": count,
        "coverage": count / len(test),
        "errors": errors,
        "empirical_risk": errors / count if count else None,
    }


def pcqrc_once(frame: pd.DataFrame, *, alpha: float, salt: str) -> dict[str, Any]:
    calibration, test = hash_split(frame, salt)
    scores = {
        precision: calibration[f"{precision}_dsde"].to_numpy(dtype=float)
        for precision in PRECISIONS
    }
    errors = {
        precision: calibration[f"{precision}_error"].to_numpy(dtype=int)
        for precision in PRECISIONS
    }
    policy = fit_precision_quantile_policy(
        scores,
        errors,
        alpha=alpha,
        delta=DELTA,
        grid=GRID,
        min_accepted=MIN_ACCEPTED,
    )
    selected = apply_precision_quantile_policy(
        policy,
        scores,
        {
            precision: test[f"{precision}_dsde"].to_numpy(dtype=float)
            for precision in PRECISIONS
        },
    )
    held_out = {
        precision: held_out_summary(test, precision, selected[precision])
        for precision in PRECISIONS
    }
    qualifies = policy.selection.status == "selected" and all(
        held_out[precision]["coverage"] >= 0.10
        and held_out[precision]["empirical_risk"] is not None
        and held_out[precision]["empirical_risk"] <= alpha
        for precision in PRECISIONS
    )
    contradiction = policy.selection.status == "selected" and any(
        held_out[precision]["empirical_risk"] is not None
        and held_out[precision]["empirical_risk"] > alpha
        for precision in PRECISIONS
    )
    return {
        "salt": salt,
        "alpha": alpha,
        "calibration_n": len(calibration),
        "test_n": len(test),
        "policy": policy.to_dict(),
        "held_out": held_out,
        "qualifies": qualifies,
        "material_contradiction": contradiction,
    }


def split_calibration_sets(
    frame: pd.DataFrame, salt: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_calibration, test = hash_split(frame, salt)
    reference, risk_calibration = hash_split(
        outer_calibration, f"split-pcqrc-inner-v1|{salt}"
    )
    return reference, risk_calibration, test


def policy_outcome(
    test: pd.DataFrame,
    selected: dict[str, np.ndarray],
    *,
    alpha: float,
    status: str,
) -> tuple[dict[str, Any], bool, bool]:
    held_out = {
        precision: held_out_summary(test, precision, selected[precision])
        for precision in PRECISIONS
    }
    qualifies = status == "selected" and all(
        held_out[precision]["coverage"] >= 0.10
        and held_out[precision]["empirical_risk"] is not None
        and held_out[precision]["empirical_risk"] <= alpha
        for precision in PRECISIONS
    )
    contradiction = status == "selected" and any(
        held_out[precision]["empirical_risk"] is not None
        and held_out[precision]["empirical_risk"] > alpha
        for precision in PRECISIONS
    )
    return held_out, qualifies, contradiction


def split_pcqrc_once(frame: pd.DataFrame, *, alpha: float, salt: str) -> dict[str, Any]:
    reference, risk_calibration, test = split_calibration_sets(frame, salt)
    reference_scores = {
        precision: reference[f"{precision}_dsde"].to_numpy(dtype=float)
        for precision in PRECISIONS
    }
    policy = fit_split_precision_quantile_policy(
        reference_scores,
        {
            precision: risk_calibration[f"{precision}_dsde"].to_numpy(dtype=float)
            for precision in PRECISIONS
        },
        {
            precision: risk_calibration[f"{precision}_error"].to_numpy(dtype=int)
            for precision in PRECISIONS
        },
        alpha=alpha,
        delta=DELTA,
        grid=GRID,
        min_accepted=MIN_ACCEPTED,
    )
    selected = apply_split_precision_quantile_policy(
        policy,
        reference_scores,
        {
            precision: test[f"{precision}_dsde"].to_numpy(dtype=float)
            for precision in PRECISIONS
        },
    )
    held_out, qualifies, contradiction = policy_outcome(
        test,
        selected,
        alpha=alpha,
        status=policy.selection.status,
    )
    return {
        "method": "split_pcqrc",
        "salt": salt,
        "alpha": alpha,
        "reference_n": len(reference),
        "risk_calibration_n": len(risk_calibration),
        "test_n": len(test),
        "policy": policy.to_dict(),
        "held_out": held_out,
        "qualifies": qualifies,
        "material_contradiction": contradiction,
    }


def shared_raw_once(frame: pd.DataFrame, *, alpha: float, salt: str) -> dict[str, Any]:
    reference, risk_calibration, test = split_calibration_sets(frame, salt)
    selection = select_precision_envelope_threshold(
        {
            precision: risk_calibration[f"{precision}_dsde"].to_numpy(dtype=float)
            for precision in PRECISIONS
        },
        {
            precision: risk_calibration[f"{precision}_error"].to_numpy(dtype=int)
            for precision in PRECISIONS
        },
        alpha=alpha,
        delta=DELTA,
        grid=GRID,
        min_accepted=MIN_ACCEPTED,
    )
    selected = {
        precision: apply_threshold(
            test[f"{precision}_dsde"].to_numpy(dtype=float), selection.threshold
        )
        for precision in PRECISIONS
    }
    held_out, qualifies, contradiction = policy_outcome(
        test,
        selected,
        alpha=alpha,
        status=selection.status,
    )
    return {
        "method": "shared_raw_dsde",
        "salt": salt,
        "alpha": alpha,
        "reference_n": len(reference),
        "risk_calibration_n": len(risk_calibration),
        "test_n": len(test),
        "policy": selection.to_dict(),
        "held_out": held_out,
        "qualifies": qualifies,
        "material_contradiction": contradiction,
    }


def precision_specific_quantile_once(
    frame: pd.DataFrame, *, alpha: float, salt: str
) -> dict[str, Any]:
    reference, risk_calibration, test = split_calibration_sets(frame, salt)
    selections = {}
    selected = {}
    for precision in PRECISIONS:
        reference_scores = reference[f"{precision}_dsde"].to_numpy(dtype=float)
        calibration_percentiles = empirical_percentiles(
            reference_scores,
            risk_calibration[f"{precision}_dsde"].to_numpy(dtype=float),
        )
        selection = select_risk_controlling_threshold(
            calibration_percentiles,
            risk_calibration[f"{precision}_error"].to_numpy(dtype=int),
            alpha=alpha,
            delta=DELTA / len(PRECISIONS),
            grid=GRID,
            min_accepted=MIN_ACCEPTED,
        )
        selections[precision] = selection.to_dict()
        selected[precision] = apply_threshold(
            empirical_percentiles(
                reference_scores,
                test[f"{precision}_dsde"].to_numpy(dtype=float),
            ),
            selection.threshold,
        )
    status = (
        "selected"
        if all(row["status"] == "selected" for row in selections.values())
        else "no_feasible_threshold"
    )
    held_out, qualifies, contradiction = policy_outcome(
        test,
        selected,
        alpha=alpha,
        status=status,
    )
    return {
        "method": "precision_specific_quantile",
        "salt": salt,
        "alpha": alpha,
        "reference_n": len(reference),
        "risk_calibration_n": len(risk_calibration),
        "test_n": len(test),
        "status": status,
        "policies": selections,
        "held_out": held_out,
        "qualifies": qualifies,
        "material_contradiction": contradiction,
    }


def strengthening_once(frame: pd.DataFrame, *, alpha: float, salt: str) -> dict[str, Any]:
    return {
        "split_pcqrc": split_pcqrc_once(frame, alpha=alpha, salt=salt),
        "shared_raw_dsde": shared_raw_once(frame, alpha=alpha, salt=salt),
        "precision_specific_quantile": precision_specific_quantile_once(
            frame, alpha=alpha, salt=salt
        ),
    }


def strengthening_robustness(frame: pd.DataFrame) -> dict[str, Any]:
    repeats = [
        strengthening_once(
            frame,
            alpha=PRIMARY_ALPHA,
            salt=f"split-pcqrc-robust-v1-{seed:03d}",
        )
        for seed in range(100)
    ]
    summary = {}
    for method in (
        "split_pcqrc",
        "shared_raw_dsde",
        "precision_specific_quantile",
    ):
        rows = [repeat[method] for repeat in repeats]
        if method == "precision_specific_quantile":
            statuses = [row["status"] for row in rows]
        else:
            statuses = [
                row["policy"]["selection"]["status"]
                if method == "split_pcqrc"
                else row["policy"]["status"]
                for row in rows
            ]
        summary[method] = {
            "selection_rate": float(np.mean(np.asarray(statuses) == "selected")),
            "qualification_rate": float(np.mean([row["qualifies"] for row in rows])),
            "contradiction_rate": float(
                np.mean([row["material_contradiction"] for row in rows])
            ),
            "coverage": {
                precision: {
                    "median": float(
                        np.median(
                            [row["held_out"][precision]["coverage"] for row in rows]
                        )
                    ),
                    "q10_q90": [
                        float(value)
                        for value in np.quantile(
                            [row["held_out"][precision]["coverage"] for row in rows],
                            [0.10, 0.90],
                        )
                    ],
                }
                for precision in PRECISIONS
            },
        }
    return {"repeats": len(repeats), "summary": summary, "rows": repeats}


def sequential_split_once(
    frame: pd.DataFrame,
    *,
    alpha: float,
    salt: str,
    coordinate: str,
) -> dict[str, Any]:
    if coordinate not in {"percentile", "raw"}:
        raise ValueError("coordinate must be percentile or raw")
    reference, risk_calibration, test = split_calibration_sets(frame, salt)
    calibration_scores = {}
    test_scores = {}
    for precision in PRECISIONS:
        if coordinate == "percentile":
            reference_scores = reference[f"{precision}_dsde"].to_numpy(dtype=float)
            calibration_scores[precision] = empirical_percentiles(
                reference_scores,
                risk_calibration[f"{precision}_dsde"].to_numpy(dtype=float),
            )
            test_scores[precision] = empirical_percentiles(
                reference_scores,
                test[f"{precision}_dsde"].to_numpy(dtype=float),
            )
        else:
            calibration_scores[precision] = risk_calibration[
                f"{precision}_dsde"
            ].to_numpy(dtype=float)
            test_scores[precision] = test[f"{precision}_dsde"].to_numpy(dtype=float)
    selection = select_fixed_sequence_precision_threshold(
        calibration_scores,
        {
            precision: risk_calibration[f"{precision}_error"].to_numpy(dtype=int)
            for precision in PRECISIONS
        },
        alpha=alpha,
        delta=DELTA,
        grid=GRID,
        min_accepted=MIN_ACCEPTED,
    )
    selected = {
        precision: apply_threshold(test_scores[precision], selection.threshold)
        for precision in PRECISIONS
    }
    held_out, qualifies, contradiction = policy_outcome(
        test,
        selected,
        alpha=alpha,
        status=selection.status,
    )
    return {
        "method": f"fixed_sequence_{coordinate}",
        "salt": salt,
        "alpha": alpha,
        "reference_n": len(reference),
        "risk_calibration_n": len(risk_calibration),
        "test_n": len(test),
        "policy": selection.to_dict(),
        "held_out": held_out,
        "qualifies": qualifies,
        "material_contradiction": contradiction,
    }


def sequential_strengthening_robustness(frame: pd.DataFrame) -> dict[str, Any]:
    rows = [
        {
            coordinate: sequential_split_once(
                frame,
                alpha=PRIMARY_ALPHA,
                salt=f"sequential-pcqrc-robust-v1-{seed:03d}",
                coordinate=coordinate,
            )
            for coordinate in ("percentile", "raw")
        }
        for seed in range(100)
    ]
    summary = {}
    for coordinate in ("percentile", "raw"):
        method_rows = [row[coordinate] for row in rows]
        summary[coordinate] = {
            "selection_rate": float(
                np.mean(
                    [row["policy"]["status"] == "selected" for row in method_rows]
                )
            ),
            "qualification_rate": float(
                np.mean([row["qualifies"] for row in method_rows])
            ),
            "contradiction_rate": float(
                np.mean([row["material_contradiction"] for row in method_rows])
            ),
            "coverage": {
                precision: {
                    "median": float(
                        np.median(
                            [
                                row["held_out"][precision]["coverage"]
                                for row in method_rows
                            ]
                        )
                    ),
                    "q10_q90": [
                        float(value)
                        for value in np.quantile(
                            [
                                row["held_out"][precision]["coverage"]
                                for row in method_rows
                            ],
                            [0.10, 0.90],
                        )
                    ],
                }
                for precision in PRECISIONS
            },
        }
    return {"repeats": len(rows), "summary": summary, "rows": rows}


def robustness_summary(frame: pd.DataFrame) -> dict[str, Any]:
    rows = [
        pcqrc_once(frame, alpha=PRIMARY_ALPHA, salt=f"pcqrc-robust-v1-{seed:03d}")
        for seed in range(100)
    ]
    selected = [row for row in rows if row["policy"]["selection"]["status"] == "selected"]
    return {
        "repeats": len(rows),
        "selection_rate": len(selected) / len(rows),
        "qualification_rate": float(np.mean([row["qualifies"] for row in rows])),
        "contradiction_rate": float(
            np.mean([row["material_contradiction"] for row in rows])
        ),
        "selected_split_qualification_rate": (
            float(np.mean([row["qualifies"] for row in selected])) if selected else None
        ),
        "coverage": {
            precision: {
                "median": float(
                    np.median([row["held_out"][precision]["coverage"] for row in rows])
                ),
                "q10_q90": [
                    float(value)
                    for value in np.quantile(
                        [row["held_out"][precision]["coverage"] for row in rows],
                        [0.10, 0.90],
                    )
                ],
            }
            for precision in PRECISIONS
        },
        "rows": rows,
    }


def paired_auc_difference(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    repeats: int = 4000,
    seed: int = 20260826,
) -> dict[str, Any]:
    point = float(roc_auc_score(labels, left) - roc_auc_score(labels, right))
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        indices = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        draws.append(
            roc_auc_score(labels[indices], left[indices])
            - roc_auc_score(labels[indices], right[indices])
        )
    return {
        "difference": point,
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "probability_difference_le_zero": float(np.mean(np.asarray(draws) <= 0.0)),
        "valid_repeats": len(draws),
    }


def analyze_panel(name: str, panel: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    metric_rows = []
    comparisons = {}
    for precision in PRECISIONS:
        labels = frame[f"{precision}_error"].to_numpy(dtype=int)
        task_ids = frame["task_id"].astype(str).to_numpy()
        estimates = {}
        for metric in METRICS:
            scores = frame[f"{precision}_{metric}"].to_numpy(dtype=float)
            estimate = stratified_bootstrap_auc(
                labels,
                scores,
                repeats=4000,
                seed=20260826,
            )
            estimates[metric] = estimate
            metric_rows.append(
                {
                    "panel": name,
                    "role": panel["role"],
                    "model": panel["model"],
                    "dataset": panel["dataset"],
                    "precision": precision,
                    "metric": metric,
                    "n": len(frame),
                    "errors": int(labels.sum()),
                    "auroc": estimate["auc"],
                    "ci95_low": estimate["ci95"][0],
                    "ci95_high": estimate["ci95"][1],
                    "aurc": aurc(labels, scores, task_ids),
                }
            )
        non_dsde = [metric for metric in METRICS if metric not in {"dsde", "fusion_oof"}]
        strongest = max(non_dsde, key=lambda metric: estimates[metric]["auc"])
        comparisons[precision] = {
            "dsde_minus_mean_nll": paired_auc_difference(
                labels,
                frame[f"{precision}_dsde"].to_numpy(dtype=float),
                frame[f"{precision}_mean_nll"].to_numpy(dtype=float),
            ),
            "dsde_minus_strongest_non_dsde": {
                "baseline": strongest,
                **paired_auc_difference(
                    labels,
                    frame[f"{precision}_dsde"].to_numpy(dtype=float),
                    frame[f"{precision}_{strongest}"].to_numpy(dtype=float),
                ),
            },
            "fusion_minus_dsde": paired_auc_difference(
                labels,
                frame[f"{precision}_fusion_oof"].to_numpy(dtype=float),
                frame[f"{precision}_dsde"].to_numpy(dtype=float),
            ),
        }

    primary = pcqrc_once(frame, alpha=PRIMARY_ALPHA, salt="pcqrc-primary-v1")
    sensitivity = {
        str(alpha): pcqrc_once(frame, alpha=alpha, salt="pcqrc-primary-v1")
        for alpha in (0.20, 0.40)
    }
    return {
        "panel": name,
        "role": panel["role"],
        "model": panel["model"],
        "dataset": panel["dataset"],
        "n": len(frame),
        "metric_rows": metric_rows,
        "comparisons": comparisons,
        "pcqrc_primary": primary,
        "pcqrc_sensitivity": sensitivity,
        "robustness": robustness_summary(frame),
        "exploratory_strengthening": {
            "primary_split": strengthening_once(
                frame, alpha=PRIMARY_ALPHA, salt="pcqrc-primary-v1"
            ),
            "robustness": strengthening_robustness(frame),
            "fixed_sequence_primary": {
                coordinate: sequential_split_once(
                    frame,
                    alpha=PRIMARY_ALPHA,
                    salt="pcqrc-primary-v1",
                    coordinate=coordinate,
                )
                for coordinate in ("percentile", "raw")
            },
            "fixed_sequence_robustness": sequential_strengthening_robustness(frame),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/pcqrc_extension_registry.yaml")
    parser.add_argument("--output-dir", default="phase4_extension/results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load((root / args.registry).read_text(encoding="utf-8"))
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_results = {}
    metric_rows = []
    fusion_rows = []
    for name, panel in registry["panels"].items():
        frame = load_panel(root, panel)
        result = analyze_panel(name, panel, frame)
        panel_results[name] = result
        metric_rows.extend(result.pop("metric_rows"))
        for precision in PRECISIONS:
            fusion_rows.extend(
                {
                    "panel": name,
                    "role": panel["role"],
                    "task_id": row.task_id,
                    "precision": precision,
                    "error": int(getattr(row, f"{precision}_error")),
                    "fusion_oof": float(getattr(row, f"{precision}_fusion_oof")),
                }
                for row in frame.itertuples(index=False)
            )

    validation = [row for row in panel_results.values() if row["role"] == "validation"]
    selected = [
        row
        for row in validation
        if row["pcqrc_primary"]["policy"]["selection"]["status"] == "selected"
    ]
    promotion = any(row["pcqrc_primary"]["qualifies"] for row in validation) and not any(
        row["pcqrc_primary"]["material_contradiction"] for row in selected
    )
    strong = bool(validation) and all(row["pcqrc_primary"]["qualifies"] for row in validation)
    validation_metric_rows = [row for row in metric_rows if row["role"] == "validation"]
    discrimination = all(
        row["ci95_low"] > 0.50
        for row in validation_metric_rows
        if row["metric"] == "dsde"
    )
    payload = {
        "protocol": registry["protocol"],
        "protocol_lock": registry["protocol_lock"],
        "exploratory_protocol": registry.get("exploratory_protocol"),
        "exploratory_protocol_lock": registry.get("exploratory_protocol_lock"),
        "sequential_protocol": registry.get("sequential_protocol"),
        "sequential_protocol_lock": registry.get("sequential_protocol_lock"),
        "panels": panel_results,
        "confirmatory": {
            "primary_pcqrc_promotion": promotion,
            "strong_both_panels": strong,
            "dsde_discrimination_all_validation_deployments": discrimination,
        },
    }
    (output_dir / "extension_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(metric_rows).to_csv(output_dir / "metric_summary.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(output_dir / "fusion_oof_predictions.csv", index=False)
    print(json.dumps(payload["confirmatory"], indent=2))


if __name__ == "__main__":
    main()
