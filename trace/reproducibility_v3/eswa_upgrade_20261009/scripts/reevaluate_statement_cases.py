"""Amend correctness judging without changing locked generation or score inputs.

Run inside Linux/WSL. Generated code is executed only by the original bubblewrap
sandbox. All affected targets are assessed before any reference file is replaced.
The initial evaluations and their completion receipts remain available as v1.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from evaluate_validation import compact, read_jsonl, run_case, sha
from statement_comparators import TASK_POLICIES, matches_statement

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
CHANGED_FIELDS = {
    "public_test_passes", "public_failure", "error", "private_tests_checked",
    "private_first_failure", "private_evaluation_seconds", "public_then_nll",
    "public_then_dsde",
}


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".amendment06.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def assess(job):
    condition, original, question, tests, target, timeout = job
    task = original["task_id"]
    if target["seed"] != 0 or sha_solution(target["solution"]) != original["solution_sha256"]:
        raise ValueError("target identity mismatch: " + condition + "/" + task)
    started = time.perf_counter()
    public_passes, observations, public_seconds = [], [], 0.0
    private_checked, private_failure, private_seconds = 0, None, 0.0
    with tempfile.TemporaryDirectory(prefix="trace-statement-") as directory:
        source = Path(directory) / "candidate.py"
        source.write_text(target["solution"], encoding="utf-8")
        function = question["metadata"].get("func_name")
        for case in tests["public"]:
            observation, seconds = run_case(source, case["input"], case["testtype"], function, timeout)
            public_seconds += seconds
            observations.append(compact(observation))
            public_passes.append(matches_statement(
                observation, case["output"], case["testtype"], task, case["input"]))
        if all(public_passes):
            for index, case in enumerate(tests["private"]):
                observation, seconds = run_case(source, case["input"], case["testtype"], function, timeout)
                private_seconds += seconds
                private_checked += 1
                if not matches_statement(observation, case["output"], case["testtype"], task, case["input"]):
                    private_failure = {"index": index, "status": observation["status"],
                                       "exception": observation.get("exception")}
                    break
    failed = sum(not passed for passed in public_passes)
    amended = deepcopy(original)
    amended.update(public_test_passes=public_passes, public_failure=failed / len(public_passes),
                   error=int(bool(failed or private_failure is not None)),
                   private_tests_checked=private_checked, private_first_failure=private_failure,
                   private_evaluation_seconds=private_seconds)
    for key, tail in (("public_then_nll", original["mean_nll"]),
                      ("public_then_dsde", original["dsde_4"])):
        amended[key] = 2 * int(failed > 0) + tail / (1 + tail)
    preserved = all(amended[key] == value for key, value in original.items() if key not in CHANGED_FIELDS)
    if not preserved:
        raise ValueError("generation, score input or acquisition time changed")
    if not amended["error"] and private_checked != len(tests["private"]):
        raise ValueError("passing target lacks complete private execution")
    correctness = ("error", "public_test_passes", "public_failure", "private_tests_checked",
                   "private_first_failure", "public_then_nll", "public_then_dsde")
    detail = {
        "condition": condition, "task_id": task, "solution_sha256": original["solution_sha256"],
        "policy": TASK_POLICIES[task]["kind"],
        "before": {key: original[key] for key in correctness},
        "after": {key: amended[key] for key in correctness},
        "private_test_count": len(tests["private"]),
        "fresh_public_execution_seconds": public_seconds,
        "fresh_private_execution_seconds": private_seconds,
        "fresh_assessment_wall_seconds": time.perf_counter() - started,
        "retained_public_observations_reproduced": observations == original["observations"][0],
        "score_inputs_and_acquisition_times_unchanged": preserved,
        "changed_correctness_fields": [key for key in correctness if original[key] != amended[key]],
    }
    return condition, task, amended, detail


def sha_solution(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    if sys.platform != "linux":
        raise RuntimeError("execute only inside the Linux bubblewrap environment")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--apply", action="store_true", help="replace current correctness records after all assessments pass")
    args = parser.parse_args()
    config = json.loads((BASE / "protocol/validation_config.json").read_text())
    lock = json.loads((BASE / "protocol/validation_lock.json").read_text())
    for relative, expected in lock["files"].items():
        path = ROOT.joinpath(*relative.replace("\\", "/").split("/"))
        if sha(path) != expected:
            raise ValueError("initial locked input changed: " + relative)
    questions = {row["task_id"]: row for row in read_jsonl(BASE / "data/livecodebench/questions.jsonl")}
    tests = {row["task_id"]: row for row in read_jsonl(BASE / "data/livecodebench/tests.jsonl")}
    if set(TASK_POLICIES) - questions.keys():
        raise ValueError("task policy absent from complete panel")
    run = BASE / config["run_dir"]
    initial = run / "evaluations_v1_exact_match"
    baseline = BASE / "revision_trace_20261009/baseline" / config["run_dir"]
    output = BASE / "revision_trace_20261009/statement_evaluation_v2"
    output.mkdir(parents=True, exist_ok=False)
    initial.mkdir(parents=True, exist_ok=True)
    jobs, originals, input_receipts = [], {}, []
    for spec in config["models"]:
        condition = spec["condition"]
        filename = condition + ".jsonl"
        original_path = baseline / "evaluations" / filename
        receipt_name = condition + "_evaluation_complete.json"
        receipt = json.loads((baseline / receipt_name).read_text())
        if sha(original_path) != receipt["sha256"]:
            raise ValueError("baseline correctness hash mismatch")
        for source, destination in ((original_path, initial / filename),
                                    (baseline / receipt_name, initial / receipt_name)):
            if destination.exists():
                if sha(source) != sha(destination):
                    raise ValueError("preserved v1 differs from baseline")
            else:
                shutil.copy2(source, destination)
        rows = read_jsonl(original_path)
        if len(rows) != 175 or {row["task_id"] for row in rows} != questions.keys():
            raise ValueError("incomplete baseline targets")
        originals[condition] = rows
        generation_path = run / "generations" / filename
        generation_receipt = json.loads((run / (condition + "_generation_complete.json")).read_text())
        if sha(generation_path) != generation_receipt["sha256"]:
            raise ValueError("generation receipt mismatch")
        targets = {row["task_id"]: row for row in read_jsonl(generation_path) if row["seed"] == 0}
        for row in rows:
            task = row["task_id"]
            if task in TASK_POLICIES:
                jobs.append((condition, row, questions[task], tests[task], targets[task],
                             config["execution_timeout_seconds"]))
        input_receipts.append({"condition": condition, "initial_evaluation_sha256": sha(original_path),
                               "generation_sha256": sha(generation_path)})
    updates, details = {}, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(assess, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            condition, task, row, detail = future.result()
            updates[(condition, task)] = row
            details.append(detail)
            print(f"statement assessment {len(details)}/{len(jobs)} {condition} {task} "
                  f"error {detail['before']['error']}->{detail['after']['error']} "
                  f"private {row['private_tests_checked']}/{row['private_test_count']}", flush=True)
    if len(details) != 48:
        raise ValueError("all eight tasks in every condition must be assessed")
    outputs = []
    for condition, original_rows in originals.items():
        revised_rows = [updates.get((condition, row["task_id"]), row) for row in original_rows]
        path = output / (condition + ".jsonl")
        atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in revised_rows))
        receipt = {"records": len(revised_rows), "sha256": sha(path)}
        atomic_text(output / (condition + "_evaluation_complete.json"), json.dumps(receipt, indent=2))
        outputs.append({"condition": condition, "records": len(revised_rows), "sha256": sha(path)})
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "statement-aware-v2", "applied": args.apply,
        "amendment_06_sha256": sha(BASE / "protocol/amendment_06_statement_comparators.md"),
        "initial_lock_sha256": sha(BASE / "protocol/validation_lock.json"),
        "reevaluation_script_sha256": sha(__file__),
        "comparator_script_sha256": sha(BASE / "scripts/statement_comparators.py"),
        "original_evaluator_sha256": sha(BASE / "scripts/evaluate_validation.py"),
        "questions_sha256": sha(BASE / "data/livecodebench/questions.jsonl"),
        "tests_sha256": sha(BASE / "data/livecodebench/tests.jsonl"),
        "policies": TASK_POLICIES, "initial_records": input_receipts, "revised_records": outputs,
        "assessed_condition_task_pairs": len(details),
        "changed_error_labels": sum(row["before"]["error"] != row["after"]["error"] for row in details),
        "changed_public_failure_statuses": sum(bool(row["before"]["public_failure"]) !=
                                                bool(row["after"]["public_failure"]) for row in details),
        "unchanged_generation_score_inputs_and_acquisition_times": all(
            row["score_inputs_and_acquisition_times_unchanged"] for row in details),
        "fresh_correctness_execution_excluded_from_acquisition_cost": True,
        "details": sorted(details, key=lambda row: (row["condition"], row["task_id"])),
    }
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    atomic_text(output / "statement_correction_receipt.json", report_text)
    if args.apply:
        for item in outputs:
            condition = item["condition"]
            for relative in (condition + ".jsonl", condition + "_evaluation_complete.json"):
                destination = run / "evaluations" / relative if relative.endswith(".jsonl") else run / relative
                atomic_text(destination, (output / relative).read_text(encoding="utf-8"))
        atomic_text(run / "statement_correction_receipt.json", report_text)
    print(json.dumps({key: report[key] for key in (
        "version", "applied", "assessed_condition_task_pairs", "changed_error_labels",
        "changed_public_failure_statuses", "unchanged_generation_score_inputs_and_acquisition_times")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
