from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from precisionprobe.evalplus_labels import evalplus_passed
from precisionprobe.execution import (
    ALLOWED_IMPORT_ROOTS,
    DANGEROUS_CALLS,
    validate_candidate_source,
)


ORIGINAL_ALLOWED_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
}


LEGACY_DANGEROUS_ATTRIBUTES = {
    "chmod",
    "connect",
    "kill",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "socket",
    "spawn",
    "system",
    "unlink",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validation_at_stage(
    source: str,
    *,
    allowed_import_roots: set[str],
    reject_dangerous_attributes: bool,
) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            denied = roots - allowed_import_roots
            if denied:
                return False, f"denied_import:{sorted(denied)[0]}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_import_roots:
                return False, f"denied_import:{root or '<relative>'}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
                return False, f"denied_call:{node.func.id}"
            if (
                reject_dangerous_attributes
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LEGACY_DANGEROUS_ATTRIBUTES
            ):
                return False, f"denied_attribute:{node.func.attr}"
    return True, None


def original_validation(source: str) -> tuple[bool, str | None]:
    return validation_at_stage(
        source,
        allowed_import_roots=ORIGINAL_ALLOWED_IMPORT_ROOTS,
        reject_dangerous_attributes=True,
    )


def attribute_fixed_validation(source: str) -> tuple[bool, str | None]:
    return validation_at_stage(
        source,
        allowed_import_roots=ORIGINAL_ALLOWED_IMPORT_ROOTS,
        reject_dangerous_attributes=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / args.run_dir
    label_rows: dict[str, dict[str, Any]] = {}
    if args.evaluation:
        evaluation = json.loads((root / args.evaluation).read_text(encoding="utf-8"))
        label_rows = {row["sample_id"]: row for row in evaluation["results"]}

    total = 0
    rejected: Counter[str] = Counter()
    rejection_reasons: dict[str, Counter[str]] = {
        "original": Counter(),
        "attribute_fixed": Counter(),
        "final": Counter(),
    }
    recovered_by_transition: Counter[str] = Counter()
    recovered_records: list[dict[str, Any]] = []
    for precision in ("q4", "q8"):
        for row in read_jsonl(run_dir / "generations" / f"{precision}.jsonl"):
            total += 1
            original_safe, original_reason = original_validation(row["solution"])
            attribute_safe, attribute_reason = attribute_fixed_validation(row["solution"])
            final_safe, final_reason = validate_candidate_source(row["solution"])
            states = {
                "original": (original_safe, original_reason),
                "attribute_fixed": (attribute_safe, attribute_reason),
                "final": (final_safe, final_reason),
            }
            for stage, (safe, reason) in states.items():
                if not safe:
                    rejected[stage] += 1
                    rejection_reasons[stage][str(reason)] += 1

            transition = None
            if not original_safe and attribute_safe:
                transition = "attribute_repair"
            elif not attribute_safe and final_safe:
                transition = "import_repair"
            if transition is not None:
                recovered_by_transition[transition] += 1
                item = {
                    "precision": precision,
                    "task_id": row["task_id"],
                    "kind": row["kind"],
                    "seed": int(row["seed"]),
                    "transition": transition,
                    "original_reason": original_reason,
                    "attribute_fixed_reason": attribute_reason,
                }
                if row["kind"] == "greedy" and label_rows:
                    sample_id = f"{row['task_id']}|{precision}_greedy"
                    item["official_plus_pass"] = evalplus_passed(label_rows[sample_id], "plus")
                recovered_records.append(item)

    greedy = [row for row in recovered_records if row["kind"] == "greedy"]
    labeled_greedy = [row for row in greedy if "official_plus_pass" in row]
    payload = {
        "run_dir": args.run_dir,
        "total_candidates": total,
        "stages": {
            stage: {
                "rejected": rejected[stage],
                "accepted": total - rejected[stage],
                "rejection_reasons": dict(sorted(rejection_reasons[stage].items())),
            }
            for stage in ("original", "attribute_fixed", "final")
        },
        "recovered_candidates": len(recovered_records),
        "recovered_by_transition": dict(sorted(recovered_by_transition.items())),
        "affected_tasks": len({row["task_id"] for row in recovered_records}),
        "recovered_greedy": len(greedy),
        "recovered_greedy_with_official_labels": len(labeled_greedy),
        "recovered_greedy_official_plus_pass": sum(
            bool(row["official_plus_pass"]) for row in labeled_greedy
        ),
        "recovered_records": recovered_records,
    }
    output = run_dir / "candidate_validation_audit.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "recovered_records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
