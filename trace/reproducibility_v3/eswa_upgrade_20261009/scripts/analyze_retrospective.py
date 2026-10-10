from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ROOT / "src"))
from precisionprobe.scoring import (  # noqa: E402
    dominant_semantic_distance_entropy,
    exact_dominant_semantic_distance_entropy,
    failure_fraction,
)

PANELS = {
    "qwen_mbpp": ("quartz_qwen_mbpp", "Qwen2.5-Coder-1.5B", "MBPP+"),
    "qwen_humaneval": ("pcqrc_qwen_humaneval", "Qwen2.5-Coder-1.5B", "HumanEval+"),
    "deepseek_mbpp": ("pcqrc_deepseek_mbpp", "DeepSeek-Coder-1.3B", "MBPP+"),
    "deepseek_humaneval": ("quartz_deepseek_humaneval", "DeepSeek-Coder-1.3B", "HumanEval+"),
    "qwen3_mbpp60": ("rapid_qwen3_confirm60", "Qwen3-4B-Instruct-2507", "MBPP+ (60 tasks)"),
}
SCORES = [
    "dsde_1", "dsde_2", "dsde_4", "exact_4", "cluster_entropy",
    "greedy_failure", "mean_nll", "mean_topk_entropy", "failure_then_nll",
]
GRID = np.linspace(0.0, 1.0, 41)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_review(y, score, fraction):
    """Label-blind expected allocation when the review boundary crosses a tie."""
    y, score = np.asarray(y, float), np.asarray(score, float)
    if len(y) == 0 or not np.isfinite(score).all():
        raise ValueError("invalid review arrays")
    capacity = min(len(y), max(0, int(math.ceil(fraction * len(y)))))
    weights = np.zeros(len(y), float)
    if capacity:
        cutoff = np.sort(score)[len(y) - capacity]
        weights[score > cutoff] = 1.0
        ties = score == cutoff
        weights[ties] = (capacity - int((score > cutoff).sum())) / int(ties.sum())
    reviewed_errors = float(np.dot(weights, y))
    all_errors = float(y.sum())
    remainder = len(y) - capacity
    return {
        "reviewed": capacity,
        "review_fraction": capacity / len(y),
        "reviewed_errors_expected": reviewed_errors,
        "error_capture": reviewed_errors / all_errors if all_errors else None,
        "residual_error_per_task": (all_errors - reviewed_errors) / len(y),
        "unreviewed_error_rate": (all_errors - reviewed_errors) / remainder if remainder else None,
        "review_yield": reviewed_errors / capacity if capacity else None,
    }


def tie_aware_aurc(y, score):
    """Expected AURC under uniform ordering inside each equal-score group."""
    y, score = np.asarray(y, float), np.asarray(score, float)
    order = np.argsort(score, kind="stable")
    sy, ss = y[order], score[order]
    prior_n = 0
    prior_errors = 0.0
    total = 0.0
    for value in np.unique(ss):
        group = sy[ss == value]
        k = np.arange(1, len(group) + 1)
        total += float(np.sum((prior_errors + k * group.mean()) / (prior_n + k)))
        prior_n += len(group)
        prior_errors += float(group.sum())
    return total / len(y)


def bootstrap_comparison(y, left, right, repeats=2000, seed=20261009):
    y, left, right = (np.asarray(v) for v in (y, left, right))
    rng = np.random.default_rng(seed)
    auc_d, residual_d = [], []
    for _ in range(repeats):
        index = rng.integers(0, len(y), len(y))
        yy, ll, rr = y[index], left[index], right[index]
        if len(np.unique(yy)) == 2:
            auc_d.append(roc_auc_score(yy, ll) - roc_auc_score(yy, rr))
        residual_d.append(
            expected_review(yy, ll, .25)["residual_error_per_task"]
            - expected_review(yy, rr, .25)["residual_error_per_task"]
        )
    return {
        "auc_difference": float(roc_auc_score(y, left) - roc_auc_score(y, right)) if len(np.unique(y)) == 2 else None,
        "residual_difference": expected_review(y, left, .25)["residual_error_per_task"] - expected_review(y, right, .25)["residual_error_per_task"],
        "auc_difference_ci95": np.quantile(auc_d, [.025, .975]).tolist() if auc_d else [None, None],
        "residual_difference_ci95": np.quantile(residual_d, [.025, .975]).tolist(),
        "repeats": repeats,
    }


def load_features():
    rows, provenance = [], []
    for panel, (folder, model, dataset) in PANELS.items():
        run = ROOT / "runs" / folder
        label_file = run / ("analysis/joined_metrics.csv" if panel == "qwen3_mbpp60" else "joined.csv")
        labels = pd.read_csv(label_file).set_index("task_id")
        token = {(r["task_id"], r["precision"]): r for r in read_jsonl(run / "token_confidence.jsonl")}
        for path in [label_file, run / "scores.jsonl", run / "token_confidence.jsonl"]:
            provenance.append({"panel": panel, "path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        for row in read_jsonl(run / "scores.jsonl"):
            task = row["task_id"]
            for precision in ("q4", "q8"):
                observations = row[f"{precision}_observations"]
                if len(observations) != 5:
                    raise ValueError(f"incomplete candidates {panel} {task} {precision}")
                tok = token[(task, precision)]
                matched = tok.get("status") == "matched"
                times = row[f"{precision}_generation_seconds"]
                f = {
                    "panel": panel, "model": model, "dataset": dataset,
                    "task_id": task, "precision": precision,
                    "error": int(labels.loc[task, f"{precision}_error"]),
                    "probe_count": row["probe_count"], "probe_source": row["probe_source"],
                    "token_replay_status": tok.get("status"),
                    "mean_nll": float(tok["mean_nll"]) if matched else np.nan,
                    "mean_topk_entropy": float(tok["mean_topk_entropy"]) if matched else np.nan,
                    "greedy_failure": failure_fraction(observations[0]),
                    "greedy_generation_seconds": float(times[0]),
                    "full_execution_seconds": float(row[f"{precision}_execution_seconds"]),
                    "full_generation_seconds": float(sum(times)),
                }
                for k in (1, 2, 4):
                    f[f"dsde_{k}"] = dominant_semantic_distance_entropy(observations[0], observations[:k+1])
                    f[f"generation_seconds_{k}"] = float(sum(times[:k+1]))
                f["exact_4"] = exact_dominant_semantic_distance_entropy(observations[0], observations)
                counts = np.array(list(Counter(json.dumps(x, sort_keys=True) for x in observations).values()))
                prob = counts / counts.sum()
                f["cluster_entropy"] = float(-np.dot(prob, np.log(prob)))
                f["failure_then_nll"] = 2.0 * int(f["greedy_failure"] > 0) + f["mean_nll"] / (1 + f["mean_nll"])
                if not np.isclose(f["dsde_4"], row[f"{precision}_dsde"], atol=1e-12):
                    raise ValueError(f"score mismatch {panel} {task} {precision}")
                rows.append(f)
    return pd.DataFrame(rows), provenance


def split_ids(ids, salt):
    ordered = sorted(set(ids), key=lambda t: hashlib.sha256(f"{salt}|{t}".encode()).hexdigest())
    nref = len(ordered) // 4
    ncal = len(ordered) // 4
    return set(ordered[:nref]), set(ordered[nref:nref+ncal]), set(ordered[nref+ncal:])


def percentile(reference, values):
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


def upper_cp(errors, total, delta):
    if not total or errors == total:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, total - errors))


def select_threshold(cal_scores, cal_y, alpha, selector, delta=.1, min_accept=20):
    keys = sorted(cal_scores)
    local_delta = delta / (len(keys) * len(GRID)) if selector == "bonferroni" else delta
    selected = None
    for threshold in GRID:
        masks = {p: cal_scores[p] <= threshold for p in keys}
        if min(int(a.sum()) for a in masks.values()) < min_accept:
            continue
        passed = all(upper_cp(int(cal_y[p][masks[p]].sum()), int(masks[p].sum()), local_delta) <= alpha for p in keys)
        if passed:
            selected = float(threshold)
        elif selector == "fixed_sequence":
            break
    return selected


def policy_evaluation(frame, score, coordinate, selector, alpha, salt):
    refs, cals, tests = split_ids(frame.task_id, salt)
    slices = {p: frame[frame.precision == p].set_index("task_id") for p in ("q4", "q8")}
    arrays = {}
    pool_ref = np.concatenate([s.loc[sorted(refs), score].to_numpy() for s in slices.values()])
    for p, s in slices.items():
        ref = s.loc[sorted(refs), score].to_numpy() if coordinate != "pooled" else pool_ref
        arrays[p] = {
            "cal": percentile(ref, s.loc[sorted(cals), score].to_numpy()),
            "test": percentile(ref, s.loc[sorted(tests), score].to_numpy()),
            "cy": s.loc[sorted(cals), "error"].to_numpy(),
            "ty": s.loc[sorted(tests), "error"].to_numpy(),
        }
    if coordinate == "independent":
        # Two independent precision rules require across-precision control.
        thresholds = {p: select_threshold({p:a["cal"]}, {p:a["cy"]}, alpha, selector, delta=.05) for p,a in arrays.items()}
    else:
        shared = select_threshold({p:a["cal"] for p,a in arrays.items()}, {p:a["cy"] for p,a in arrays.items()}, alpha, selector)
        thresholds = {p: shared for p in arrays}
    result = []
    for p, a in arrays.items():
        threshold = thresholds[p]
        accepted = a["test"] <= threshold if threshold is not None else np.zeros(len(a["test"]), bool)
        naccept, nerror = int(accepted.sum()), int(a["ty"][accepted].sum())
        result.append({
            "precision":p, "score":score, "coordinate":coordinate, "selector":selector,
            "alpha":alpha, "salt":salt, "reference_n":len(refs), "calibration_n":len(cals),
            "test_n":len(tests), "selected":threshold is not None, "threshold":threshold,
            "accepted":naccept, "errors":nerror, "coverage":naccept/len(tests),
            "risk":nerror/naccept if naccept else None,
            "test_target_met":nerror/naccept <= alpha if naccept else False,
        })
    return result


def analyze(frame, bootstrap=2000, split_repeats=30):
    metric_rows, comparisons, policies = [], [], []
    for (panel, precision), cell in frame.groupby(["panel", "precision"], sort=True):
        y = cell.error.to_numpy()
        for score in SCORES:
            valid = np.isfinite(cell[score].to_numpy())
            vals = cell.loc[valid, score].to_numpy()
            yy = cell.loc[valid, "error"].to_numpy()
            for budget in (.10, .25, .50):
                metric_rows.append({
                    "panel":panel, "precision":precision, "score":score, "n":int(valid.sum()),
                    "excluded_unmatched":int((~valid).sum()),
                    "errors":int(yy.sum()), "budget":budget,
                    "auroc":float(roc_auc_score(yy, vals)) if len(np.unique(yy)) == 2 else None,
                    "aurc":tie_aware_aurc(yy, vals),
                    **expected_review(yy, vals, budget),
                })
        for baseline in ("mean_nll", "greedy_failure", "failure_then_nll", "exact_4", "dsde_1", "dsde_2"):
            paired = cell[np.isfinite(cell[baseline])]
            comparisons.append({"panel":panel, "precision":precision, "left":"dsde_4", "right":baseline,
                "paired_n":len(paired), "excluded_unmatched":len(cell)-len(paired),
                **bootstrap_comparison(paired.error, paired.dsde_4, paired[baseline], repeats=bootstrap)})
        print(f"ranking complete: {panel}/{precision}, n={len(cell)}", flush=True)
    for panel, panel_frame in frame.groupby("panel", sort=True):
        if panel_frame.task_id.nunique() < 80:
            continue
        for repetition in range(split_repeats):
            salt = "eswa-20261009-split-" + str(repetition)
            for score in ("dsde_4", "dsde_1", "mean_nll", "failure_then_nll"):
                for coordinate in ("conditioned", "pooled", "independent"):
                    for selector in ("bonferroni", "fixed_sequence"):
                        for alpha in (.10, .20, .30, .40):
                            policies.extend({"panel":panel, **r} for r in policy_evaluation(panel_frame, score, coordinate, selector, alpha, salt))
        print(f"calibration complete: {panel}, splits={split_repeats}", flush=True)
    return pd.DataFrame(metric_rows), comparisons, pd.DataFrame(policies)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--split-repeats", type=int, default=30)
    args=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, provenance=load_features()
    frame.to_csv(OUT / "task_features.csv", index=False)
    (OUT / "input_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    metrics, comparisons, policies=analyze(frame, args.bootstrap, args.split_repeats)
    metrics.to_csv(OUT / "ranking_review_metrics.csv", index=False)
    (OUT / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    policies.to_csv(OUT / "policy_evaluations.csv", index=False)
    summary=policies.groupby(["panel","precision","score","coordinate","selector","alpha"], dropna=False).agg(
        selection_rate=("selected","mean"), mean_coverage=("coverage","mean"),
        mean_risk_when_accepting=("risk","mean"), unconditional_target_met=("test_target_met","mean"),
    ).reset_index()
    summary.to_csv(OUT / "policy_summary.csv", index=False)
    print("finished",len(frame),"rows", flush=True)


if __name__ == "__main__":
    main()
