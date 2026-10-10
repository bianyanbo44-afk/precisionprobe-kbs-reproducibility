"""Replay the statistical analysis from retained records, without model inference.

This entry point is portable. The original Windows/WSL generation scripts and
their protocol lock are preserved separately and are not rewritten by replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_retrospective as retro
import analyze_validation as validation

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def verify_bundle():
    manifest_path = ROOT / "artifact_manifest.json"
    if not manifest_path.exists():
        # The live research workspace is not a packaged artifact.
        return {"mode": "local_workspace", "bundle_files_verified": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()) or retro.digest(path) != item["sha256"]:
            raise ValueError("artifact integrity failure: " + item["path"])
    return {"mode": "packaged_artifact", "bundle_files_verified": len(manifest["files"])}


def equal_tables(actual, expected, label):
    if set(actual.columns) != set(expected.columns):
        raise ValueError(label + ": different columns")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected[actual.columns].reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        obj=label,
    )


def close(value, expected, label):
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(label + ": stored value disagrees with retained observations")


def load_retrospective():
    provenance = json.loads((BASE / "analysis/input_manifest.json").read_text())
    for item in provenance:
        path = ROOT.joinpath(*item["path"].replace("\\", "/").split("/"))
        if retro.digest(path) != item["sha256"]:
            raise ValueError("archived input hash mismatch: " + item["path"])
    frame, _ = retro.load_features()
    expected = pd.read_csv(BASE / "analysis/task_features.csv", float_precision="round_trip")
    equal_tables(frame, expected, "retrospective feature reconstruction")
    if len(frame) != 2288 or frame.duplicated(["panel", "precision", "task_id"]).any():
        raise ValueError("wrong retrospective task keys")
    for panel, (folder, _, _) in retro.PANELS.items():
        run = ROOT / "runs" / folder
        labels = json.loads((run / "evalplus_results.json").read_text())["results"]
        labels = {r["sample_id"]: int(r["base"][0] != "pass" or r["plus"][0] != "pass") for r in labels}
        tokens = {(r["task_id"], r["precision"]): r for r in retro.read_jsonl(run / "token_confidence.jsonl")}
        for precision in ("q4", "q8"):
            generation_path = run / "generations" / (precision + ".jsonl")
            targets = {r["task_id"]: r for r in retro.read_jsonl(generation_path) if r["kind"] == "greedy"} if generation_path.exists() else {}
            subset = frame[(frame.panel == panel) & (frame.precision == precision)]
            for row in subset.to_dict("records"):
                if labels[row["task_id"] + "|" + precision + "_greedy"] != row["error"]:
                    raise ValueError("archived functional label mismatch")
                token = tokens[(row["task_id"], precision)]
                if targets:
                    target = targets[row["task_id"]]
                    actual = hashlib.sha256(target["solution"].encode()).hexdigest()
                    if actual != target["solution_sha256"] or actual != token["expected_solution_sha256"]:
                        raise ValueError("archived target hash mismatch")
                if token["status"] == "matched" and token["generated_solution_sha256"] != token["expected_solution_sha256"]:
                    raise ValueError("archived confidence marked matched without equal code hashes")
    return frame


def load_validation():
    config = json.loads((BASE / "protocol/validation_config.json").read_text())
    task_path = BASE / "data/livecodebench/task_metadata.csv"
    if task_path.exists():
        task_ids = set(pd.read_csv(task_path).task_id)
    else:
        task_ids = {r["task_id"] for r in retro.read_jsonl(BASE / "data/livecodebench/questions.jsonl")}
    if len(task_ids) != 175:
        raise ValueError("the complete 175-task validation increment is required")
    frames = []
    run = BASE / config["run_dir"]
    validation.verify_statement_amendment(run)
    lock_hash = retro.digest(BASE / "protocol/validation_lock.json")
    for spec in config["models"]:
        condition = spec["condition"]
        records = {}
        for kind in ("generation", "evaluation"):
            path = run / (kind + "s") / (condition + ".jsonl")
            receipt = json.loads((run / (condition + "_" + kind + "_complete.json")).read_text())
            rows = retro.read_jsonl(path)
            if retro.digest(path) != receipt["sha256"] or len(rows) != receipt["records"]:
                raise ValueError("validation receipt mismatch: " + str(path))
            records[kind] = rows
        generated = records["generation"]
        expected_keys = {(task, seed) for task in task_ids for seed in (0, 11, 29, 47, 71)}
        if len(generated) != 875 or {(r["task_id"], r["seed"]) for r in generated} != expected_keys:
            raise ValueError("generation key mismatch: " + condition)
        indexed = {(r["task_id"], r["seed"]): r for r in generated}
        for row in generated:
            if row["condition"] != condition or row["model_sha256"] != spec["sha256"]:
                raise ValueError("generation artifact mismatch")
            if row["protocol_lock_sha256"] != lock_hash:
                raise ValueError("generation protocol mismatch")
            if hashlib.sha256(row["solution"].encode()).hexdigest() != row["solution_sha256"]:
                raise ValueError("generated solution hash mismatch")
        frame = pd.DataFrame(records["evaluation"])
        if len(frame) != 175 or set(frame.task_id) != task_ids or not (frame.condition == condition).all():
            raise ValueError("evaluation key mismatch: " + condition)
        if not np.isfinite(frame[list(validation.SCORES)].to_numpy(float)).all():
            raise ValueError("nonfinite screening score")
        for row in records["evaluation"]:
            candidates = [indexed[(row["task_id"], seed)] for seed in (0, 11, 29, 47, 71)]
            target = candidates[0]
            if row["solution_sha256"] != target["solution_sha256"]:
                raise ValueError("target join mismatch")
            for key in ("mean_nll", "mean_topk_entropy"):
                close(row[key], target["token_confidence"][key], key)
            observations = row["observations"]
            if len(observations) != 5 or any(len(x) != row["probe_count"] for x in observations):
                raise ValueError("public observation shape mismatch")
            if len(row["public_test_passes"]) != row["probe_count"] or row["probe_count"] <= 0:
                raise ValueError("public correctness count mismatch")
            for k in (1, 2, 4):
                close(row[f"dsde_{k}"], retro.dominant_semantic_distance_entropy(observations[0], observations[:k+1]), "DSDE")
                close(row[f"generation_seconds_{k}"], sum(r["elapsed_seconds"] for r in candidates[:k+1]), "generation cost")
                close(row[f"execution_seconds_{k}"], sum(sum(x) for x in row["probe_execution_seconds"][:k+1]), "execution cost")
            close(row["exact_4"], retro.exact_dominant_semantic_distance_entropy(observations[0], observations), "exact disagreement")
            close(row["greedy_failure"], retro.failure_fraction(observations[0]), "execution failure")
            counts = Counter(json.dumps(x, sort_keys=True) for x in observations)
            close(row["cluster_entropy"], -sum(n/5 * math.log(n/5) for n in counts.values()), "cluster entropy")
            public_failures = sum(not p for p in row["public_test_passes"])
            close(row["public_failure"], public_failures / row["probe_count"], "public failure")
            for score, guard, tail in (("failure_then_nll", row["greedy_failure"] > 0, row["mean_nll"]),
                                       ("public_then_nll", public_failures > 0, row["mean_nll"]),
                                       ("public_then_dsde", public_failures > 0, row["dsde_4"])):
                close(row[score], 2*int(guard) + tail/(1+tail), score)
            error = int(bool(public_failures or row["private_first_failure"] is not None))
            if row["error"] != error or row["private_test_count"] <= 0:
                raise ValueError("functional label mismatch")
            if not error and row["private_tests_checked"] != row["private_test_count"]:
                raise ValueError("passing target lacks complete private tests")
        frames.append(frame.sort_values("task_id").reset_index(drop=True))
    frame = pd.concat(frames, ignore_index=True)
    scalar = frame.drop(columns=["observations", "probe_execution_seconds", "public_test_passes", "private_first_failure"])
    expected = pd.read_csv(BASE / "analysis/validation/task_features.csv", float_precision="round_trip")
    equal_tables(scalar, expected, "prospective feature reconstruction")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "retrospective", "validation"), default="all")
    parser.add_argument("--verify-only", action="store_true", help="check raw-to-feature integrity without resampling")
    parser.add_argument("--output", type=Path, default=BASE / "replay_results")
    args = parser.parse_args()
    status = verify_bundle()
    status.update(stage=args.stage, verify_only=args.verify_only, reconstructed={})
    for stage in (("retrospective", "validation") if args.stage == "all" else (args.stage,)):
        frame = load_retrospective() if stage == "retrospective" else load_validation()
        status["reconstructed"][stage] = len(frame)
        print("source records verified", stage, len(frame), flush=True)
        if args.verify_only:
            continue
        output = args.output / stage
        output.mkdir(parents=True, exist_ok=True)
        if stage == "retrospective":
            metrics, comparisons, policies = retro.analyze(frame)
            metrics.to_csv(output / "ranking_review_metrics.csv", index=False)
            json_write(output / "paired_comparisons.json", comparisons)
            policies.to_csv(output / "policy_evaluations.csv", index=False)
        else:
            primary, draws = validation.primary_comparison(frame)
            json_write(output / "primary_comparison.json", primary)
            expected = json.loads((BASE / "analysis/validation/primary_comparison.json").read_text())
            if primary != expected:
                raise ValueError("primary replay differs from retained analysis")
            np.save(output / "primary_bootstrap_differences.npy", draws)
            expected_draws = np.load(BASE / "analysis/validation/primary_bootstrap_differences.npy")
            np.testing.assert_array_equal(draws, expected_draws)
            metrics, subgroups = validation.ranking_tables(frame)
            metrics.to_csv(output / "ranking_review_metrics.csv", index=False)
            subgroups.to_csv(output / "subgroup_metrics.csv", index=False)
            validation.cost_tables(frame).to_csv(output / "cost_frontier.csv", index=False)
            public_metrics, public_contrasts = validation.public_pass_diagnostic(frame)
            public_metrics.to_csv(output / "public_pass_metrics.csv", index=False)
            json_write(output / "public_pass_comparisons.json", public_contrasts)
            policies, summary = validation.calibration_tables(frame)
            policies.to_csv(output / "policy_evaluations.csv", index=False)
            summary.to_csv(output / "policy_summary.csv", index=False)
        for filename in ("ranking_review_metrics.csv", "policy_evaluations.csv"):
            reference = BASE / "analysis" / ("validation/" if stage == "validation" else "") / filename
            equal_tables(pd.read_csv(output / filename, float_precision="round_trip"),
                         pd.read_csv(reference, float_precision="round_trip"), stage + ": " + filename)
        print("statistical replay matches", stage, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    json_write(args.output / "verification.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
