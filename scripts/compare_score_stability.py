from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml


IGNORED_KEYS = {"q4_execution_seconds", "q8_execution_seconds"}


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keyed = {row["task_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate task_id in {path}")
    return keyed


def semantic_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in IGNORED_KEYS}


def semantic_hash(rows: dict[str, dict[str, Any]]) -> str:
    payload = [semantic_row(rows[task]) for task in sorted(rows)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stable_rows(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    before = file_hash(path)
    rows = read_rows(path)
    after = file_hash(path)
    if before != after:
        raise RuntimeError(f"score file changed while it was being read: {path}")
    return rows, after


def compare(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    expected_tasks: int,
) -> dict[str, Any]:
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")
    left_tasks = set(left)
    right_tasks = set(right)
    common = sorted(left_tasks & right_tasks)
    mismatches = [task for task in common if semantic_row(left[task]) != semantic_row(right[task])]
    payload = {
        "expected_tasks": expected_tasks,
        "left_tasks": len(left),
        "right_tasks": len(right),
        "left_complete": len(left) == expected_tasks,
        "right_complete": len(right) == expected_tasks,
        "missing_from_left": sorted(right_tasks - left_tasks),
        "missing_from_right": sorted(left_tasks - right_tasks),
        "semantic_mismatch_count": len(mismatches),
        "semantic_mismatch_task_ids": mismatches,
        "left_semantic_sha256": semantic_hash(left),
        "right_semantic_sha256": semantic_hash(right),
    }
    payload["status"] = (
        "PASS"
        if payload["left_complete"]
        and payload["right_complete"]
        and not payload["missing_from_left"]
        and not payload["missing_from_right"]
        and not mismatches
        else "FAIL"
    )
    return payload


def expected_tasks_from_config(path: Path) -> int:
    from precisionprobe.data import load_benchmark

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    task_ids = sorted(load_benchmark(config["dataset"], mini=bool(config["mini"])))
    if config.get("task_limit") is not None:
        random.Random(int(config["task_seed"])).shuffle(task_ids)
        task_ids = task_ids[: int(config["task_limit"])]
    return len(task_ids)


def expected_tasks_from_manifest(path: Path) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError(f"manifest has no non-empty task_ids list: {path}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task_id in manifest: {path}")
    return len(task_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--output", required=True)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-tasks", type=int)
    expected.add_argument("--config")
    expected.add_argument("--manifest")
    args = parser.parse_args()
    if args.expected_tasks is not None:
        expected_tasks = args.expected_tasks
    elif args.config is not None:
        expected_tasks = expected_tasks_from_config(Path(args.config))
    else:
        expected_tasks = expected_tasks_from_manifest(Path(args.manifest))
    left_path = Path(args.left)
    right_path = Path(args.right)
    left_rows, left_file_sha256 = read_stable_rows(left_path)
    right_rows, right_file_sha256 = read_stable_rows(right_path)
    payload = compare(
        left_rows,
        right_rows,
        expected_tasks,
    )
    payload.update(
        {
            "left_file": str(left_path.resolve()),
            "right_file": str(right_path.resolve()),
            "left_file_sha256": left_file_sha256,
            "right_file_sha256": right_file_sha256,
        }
    )
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
