from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from precisionprobe.data import load_benchmark


REQUIRED_CODE_PATHS = {
    "run_quartz_cell.py": Path("scripts/run_quartz_cell.py"),
    "execution.py": Path("src/precisionprobe/execution.py"),
    "inference.py": Path("src/precisionprobe/inference.py"),
    "scoring.py": Path("src/precisionprobe/scoring.py"),
    "data.py": Path("src/precisionprobe/data.py"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        recorded_path = str(resolved.relative_to(root.resolve()))
    except ValueError:
        recorded_path = str(resolved)
    return {
        "path": recorded_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def latest_phase(manifest: dict[str, Any], phase: str) -> dict[str, Any] | None:
    records = manifest.get("phases", {}).get(phase, [])
    return records[-1] if isinstance(records, list) and records else None


def verify_artifact(
    root: Path,
    expected_path: Path,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = expected_path.resolve()
    result: dict[str, Any] = {
        "expected_path": str(expected_path),
        "recorded": record is not None,
        "exists": expected.exists(),
        "path_matches": False,
        "bytes_match": False,
        "sha256_match": False,
    }
    if record is None or not expected.exists():
        result["valid"] = False
        return result
    recorded_path = resolve_path(root, record.get("path", "")).resolve()
    actual_bytes = expected.stat().st_size
    actual_sha256 = sha256_file(expected)
    result.update(
        {
            "recorded_path": record.get("path"),
            "recorded_bytes": record.get("bytes"),
            "actual_bytes": actual_bytes,
            "recorded_sha256": record.get("sha256"),
            "actual_sha256": actual_sha256,
            "path_matches": recorded_path == expected,
            "bytes_match": record.get("bytes") == actual_bytes,
            "sha256_match": record.get("sha256") == actual_sha256,
        }
    )
    result["valid"] = all(
        result[key] for key in ("path_matches", "bytes_match", "sha256_match")
    )
    return result


def register_evaluation(
    manifest: dict[str, Any],
    evaluation_path: Path,
    root: Path,
) -> bool:
    record = artifact_record(evaluation_path, root)
    latest = latest_phase(manifest, "evaluation")
    if latest is not None and latest.get("outputs", {}).get("evaluation") == record:
        return False
    generation = latest_phase(manifest, "generation")
    recorded_at = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("phases", {}).setdefault("evaluation", []).append(
        {
            "recorded_at": recorded_at,
            "status": "CAPTURED_EXTERNAL",
            "recorded_by": "audit_quartz_run.py",
            "config_sha256": manifest.get("config_sha256"),
            "inputs": {
                "generations": (
                    generation.get("outputs", {}) if generation is not None else {}
                )
            },
            "outputs": {"evaluation": record},
        }
    )
    manifest["updated_at"] = recorded_at
    return True


def audit_manifest_hashes(
    *,
    root: Path,
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    expected_tasks: int,
    manifest: dict[str, Any],
    evaluation_path: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    config_actual = sha256_file(config_path)
    config_check = {
        "recorded_sha256": manifest.get("config_sha256"),
        "actual_sha256": config_actual,
        "valid": manifest.get("config_sha256") == config_actual,
    }
    if not config_check["valid"]:
        failures.append("manifest_config_hash")

    code_checks = {}
    recorded_code = manifest.get("code_sha256", {})
    for name, relative_path in REQUIRED_CODE_PATHS.items():
        path = root / relative_path
        actual = sha256_file(path)
        check = {
            "recorded_sha256": recorded_code.get(name),
            "actual_sha256": actual,
            "valid": recorded_code.get(name) == actual,
        }
        code_checks[name] = check
        if not check["valid"]:
            failures.append(f"manifest_code_hash:{name}")

    generation_phase = latest_phase(manifest, "generation")
    generation_checks = {}
    for precision in config["precisions"]:
        record = (
            generation_phase.get("outputs", {}).get(precision)
            if generation_phase is not None
            else None
        )
        check = verify_artifact(
            root,
            run_dir / "generations" / f"{precision}.jsonl",
            record,
        )
        generation_checks[precision] = check
        if not check["valid"]:
            failures.append(f"manifest_generation_hash:{precision}")

    scoring_phase = latest_phase(manifest, "scoring")
    score_paths = {
        "scores": run_dir / "scores.jsonl",
        "evalplus_samples": run_dir / "evalplus_samples.jsonl",
    }
    scoring_checks = {}
    for name, path in score_paths.items():
        record = (
            scoring_phase.get("outputs", {}).get(name) if scoring_phase is not None else None
        )
        check = verify_artifact(root, path, record)
        scoring_checks[name] = check
        if not check["valid"]:
            failures.append(f"manifest_scoring_hash:{name}")

    scoring_input_matches = {}
    for precision in config["precisions"]:
        generation_hash = (
            generation_phase.get("outputs", {}).get(precision, {}).get("sha256")
            if generation_phase is not None
            else None
        )
        scoring_hash = (
            scoring_phase.get("inputs", {})
            .get("generations", {})
            .get(precision, {})
            .get("sha256")
            if scoring_phase is not None
            else None
        )
        scoring_input_matches[precision] = (
            generation_hash is not None and generation_hash == scoring_hash
        )
        if not scoring_input_matches[precision]:
            failures.append(f"scoring_generation_link:{precision}")

    phase_links = {
        "generation_config": (
            generation_phase is not None
            and generation_phase.get("config_sha256") == config_actual
        ),
        "generation_expected_tasks": (
            generation_phase is not None
            and generation_phase.get("expected_tasks") == expected_tasks
        ),
        "scoring_config": (
            scoring_phase is not None
            and scoring_phase.get("config_sha256") == config_actual
        ),
        "scoring_expected_tasks": (
            scoring_phase is not None
            and scoring_phase.get("expected_tasks") == expected_tasks
        ),
        "scoring_code": (
            scoring_phase is not None
            and all(
                scoring_phase.get("code_sha256", {}).get(name) == recorded_code.get(name)
                for name in REQUIRED_CODE_PATHS
            )
        ),
    }
    for name, valid in phase_links.items():
        if not valid:
            failures.append(f"manifest_phase_link:{name}")

    evaluation_phase = latest_phase(manifest, "evaluation")
    evaluation_check = None
    evaluation_generation_links = None
    if evaluation_path is not None:
        record = (
            evaluation_phase.get("outputs", {}).get("evaluation")
            if evaluation_phase is not None
            else None
        )
        evaluation_check = verify_artifact(root, evaluation_path, record)
        if not evaluation_check["valid"]:
            failures.append("manifest_evaluation_hash")
        evaluation_generation_links = {}
        for precision in config["precisions"]:
            generation_hash = (
                generation_phase.get("outputs", {}).get(precision, {}).get("sha256")
                if generation_phase is not None
                else None
            )
            evaluation_hash = (
                evaluation_phase.get("inputs", {})
                .get("generations", {})
                .get(precision, {})
                .get("sha256")
                if evaluation_phase is not None
                else None
            )
            evaluation_generation_links[precision] = (
                generation_hash is not None and generation_hash == evaluation_hash
            )
            if not evaluation_generation_links[precision]:
                failures.append(f"evaluation_generation_link:{precision}")
        evaluation_config_matches = (
            evaluation_phase is not None
            and evaluation_phase.get("config_sha256") == config_actual
        )
        phase_links["evaluation_config"] = evaluation_config_matches
        if not evaluation_config_matches:
            failures.append("manifest_phase_link:evaluation_config")

    return (
        {
            "config": config_check,
            "code": code_checks,
            "generations": generation_checks,
            "scoring": scoring_checks,
            "scoring_generation_links": scoring_input_matches,
            "phase_links": phase_links,
            "evaluation": evaluation_check,
            "evaluation_generation_links": evaluation_generation_links,
        },
        failures,
    )


def assess_stability(
    stability: dict[str, Any] | None,
    expected_tasks: int,
) -> dict[str, Any]:
    failures = []
    if stability is None:
        failures.append("stability_report_missing")
    else:
        if stability.get("status") != "PASS":
            failures.append("stability_status")
        if stability.get("expected_tasks") != expected_tasks:
            failures.append("stability_expected_tasks")
        if stability.get("left_tasks") != expected_tasks:
            failures.append("stability_left_completeness")
        if stability.get("right_tasks") != expected_tasks:
            failures.append("stability_right_completeness")
        if stability.get("left_complete") is not True:
            failures.append("stability_left_complete_flag")
        if stability.get("right_complete") is not True:
            failures.append("stability_right_complete_flag")
        if stability.get("missing_from_left"):
            failures.append("stability_missing_from_left")
        if stability.get("missing_from_right"):
            failures.append("stability_missing_from_right")
        if stability.get("semantic_mismatch_count") != 0:
            failures.append("stability_semantic_mismatches")
        left_hash = stability.get("left_semantic_sha256")
        right_hash = stability.get("right_semantic_sha256")
        if (
            not isinstance(left_hash, str)
            or not left_hash
            or left_hash != right_hash
        ):
            failures.append("stability_semantic_hashes")
    return {
        "expected_tasks": expected_tasks,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def assess_stability_score_history(
    manifest: dict[str, Any],
    stability: dict[str, Any] | None,
) -> dict[str, Any]:
    scoring_records = [
        row
        for row in manifest.get("phases", {}).get("scoring", [])
        if row.get("status") == "COMPLETED"
    ]
    recorded_hashes = [
        row.get("outputs", {}).get("scores", {}).get("sha256")
        for row in scoring_records[-2:]
    ]
    reported_hashes = (
        [stability.get("left_file_sha256"), stability.get("right_file_sha256")]
        if stability is not None
        else []
    )
    valid = (
        len(recorded_hashes) == 2
        and all(isinstance(value, str) and value for value in recorded_hashes)
        and all(isinstance(value, str) and value for value in reported_hashes)
        and sorted(recorded_hashes) == sorted(reported_hashes)
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "recorded_score_sha256": recorded_hashes,
        "reported_score_sha256": reported_hashes,
    }


def expected_task_ids(config: dict[str, Any]) -> list[str]:
    task_ids = sorted(load_benchmark(config["dataset"], mini=bool(config["mini"])))
    if config.get("task_limit") is not None:
        random.Random(int(config["task_seed"])).shuffle(task_ids)
        task_ids = task_ids[: int(config["task_limit"])]
    return task_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation")
    parser.add_argument(
        "--register-evaluation",
        action="store_true",
        help="append the supplied evaluation artifact to the manifest before auditing",
    )
    parser.add_argument("--stability")
    parser.add_argument("--require-eligible", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = resolve_path(root, args.run_dir)
    config_path = resolve_path(root, args.config)
    evaluation_path = resolve_path(root, args.evaluation) if args.evaluation else None
    stability_path = resolve_path(root, args.stability) if args.stability else None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = expected_task_ids(config)
    expected_keys = {
        (task, kind, seed)
        for task in tasks
        for kind, seeds in (("greedy", [0]), ("alternative", config["alternative_seeds"]))
        for seed in seeds
    }

    integrity_failures: list[str] = []
    precision_summary = {}
    for precision in config["precisions"]:
        rows = read_jsonl(run_dir / "generations" / f"{precision}.jsonl")
        keys = [(row["task_id"], row["kind"], int(row["seed"])) for row in rows]
        key_set = set(keys)
        duplicate_count = len(keys) - len(key_set)
        hash_mismatches = sum(
            sha256_text(row["solution"]) != row["solution_sha256"] for row in rows
        )
        missing = sorted(expected_keys - key_set)
        unexpected = sorted(key_set - expected_keys)
        precision_summary[precision] = {
            "rows": len(rows),
            "expected_rows": len(expected_keys),
            "duplicate_keys": duplicate_count,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "solution_hash_mismatches": hash_mismatches,
            "complete": not (duplicate_count or missing or unexpected or hash_mismatches),
        }
        if not precision_summary[precision]["complete"]:
            integrity_failures.append(f"{precision}_generation_integrity")

    scores = read_jsonl(run_dir / "scores.jsonl")
    score_tasks = [row["task_id"] for row in scores]
    score_task_set = set(score_tasks)
    score_duplicate_count = len(score_tasks) - len(score_task_set)
    score_missing = set(tasks) - score_task_set
    score_unexpected = score_task_set - set(tasks)
    score_complete = not (score_duplicate_count or score_missing or score_unexpected)
    if not score_complete:
        integrity_failures.append("score_completeness")

    evaluation_summary: dict[str, Any] | None = None
    if evaluation_path is not None:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        rows = evaluation["results"]
        expected_samples = {
            f"{task}|{precision}_greedy" for task in tasks for precision in config["precisions"]
        }
        sample_ids = [row["sample_id"] for row in rows]
        evaluation_summary = {
            "rows": len(rows),
            "expected_rows": len(expected_samples),
            "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
            "missing_sample_ids": len(expected_samples - set(sample_ids)),
            "unexpected_sample_ids": len(set(sample_ids) - expected_samples),
        }
        evaluation_summary["complete"] = not any(
            evaluation_summary[key]
            for key in ("duplicate_sample_ids", "missing_sample_ids", "unexpected_sample_ids")
        )
        if not evaluation_summary["complete"]:
            integrity_failures.append("evaluation_completeness")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.register_evaluation and evaluation_path is None:
        parser.error("--register-evaluation requires --evaluation")
    if (
        args.register_evaluation
        and evaluation_path is not None
        and register_evaluation(manifest, evaluation_path, root)
    ):
        write_json(manifest_path, manifest)
    manifest_summary, manifest_failures = audit_manifest_hashes(
        root=root,
        run_dir=run_dir,
        config_path=config_path,
        config=config,
        expected_tasks=len(tasks),
        manifest=manifest,
        evaluation_path=evaluation_path,
    )
    integrity_failures.extend(manifest_failures)

    stability_payload = (
        json.loads(stability_path.read_text(encoding="utf-8"))
        if stability_path is not None
        else None
    )
    stability_summary = assess_stability(stability_payload, len(tasks))
    stability_history = assess_stability_score_history(manifest, stability_payload)
    eligibility_failures = []
    if integrity_failures:
        eligibility_failures.append("integrity_audit")
    if evaluation_path is None:
        eligibility_failures.append("evaluation_required")
    eligibility_failures.extend(stability_summary["failures"])
    if stability_history["status"] != "PASS":
        eligibility_failures.append("stability_score_history_link")
    eligible = not eligibility_failures

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "config": args.config,
        "tasks": len(tasks),
        "generations": precision_summary,
        "scores": {
            "rows": len(scores),
            "expected_rows": len(tasks),
            "duplicate_task_ids": score_duplicate_count,
            "missing_task_ids": len(score_missing),
            "unexpected_task_ids": len(score_unexpected),
            "complete": score_complete,
        },
        "evaluation": evaluation_summary,
        "manifest_hashes": manifest_summary,
        "stability": {**stability_summary, "score_history": stability_history},
        "failures": integrity_failures,
        "status": "PASS" if not integrity_failures else "FAIL",
        "eligible": eligible,
        "eligibility": {
            "eligible": eligible,
            "failures": eligibility_failures,
            "rule": "integrity PASS + complete evaluation + stability PASS",
        },
    }
    write_json(run_dir / "integrity_audit.json", payload)
    print(json.dumps(payload, indent=2))
    if integrity_failures or (args.require_eligible and not eligible):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
