from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_run(run_dir: Path, precisions: list[str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for precision in precisions:
        rows = read_jsonl(run_dir / "generations" / f"{precision}.jsonl")
        syntax_pass = 0
        syntax_errors: list[dict[str, str]] = []
        source_lengths: list[int] = []
        generated_tokens: list[int] = []
        elapsed_seconds: list[float] = []
        finish_reasons: dict[str, int] = {}
        for row in rows:
            source = str(row.get("solution", ""))
            source_lengths.append(len(source))
            usage = row.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                generated_tokens.append(int(usage["completion_tokens"]))
            if row.get("elapsed_seconds") is not None:
                elapsed_seconds.append(float(row["elapsed_seconds"]))
            finish = str(row.get("finish_reason"))
            finish_reasons[finish] = finish_reasons.get(finish, 0) + 1
            try:
                ast.parse(source)
                syntax_pass += 1
            except SyntaxError as exc:
                syntax_errors.append(
                    {
                        "task_id": str(row.get("task_id")),
                        "kind": str(row.get("kind")),
                        "message": exc.msg,
                    }
                )
        results[precision] = {
            "rows": len(rows),
            "expected_rows": None,
            "syntax_pass": syntax_pass,
            "syntax_error": len(syntax_errors),
            "syntax_error_examples": syntax_errors[:10],
            "mean_source_chars": statistics.mean(source_lengths) if source_lengths else None,
            "median_source_chars": statistics.median(source_lengths) if source_lengths else None,
            "mean_completion_tokens": statistics.mean(generated_tokens) if generated_tokens else None,
            "median_completion_tokens": statistics.median(generated_tokens) if generated_tokens else None,
            "mean_elapsed_seconds": statistics.mean(elapsed_seconds) if elapsed_seconds else None,
            "median_elapsed_seconds": statistics.median(elapsed_seconds) if elapsed_seconds else None,
            "finish_reasons": finish_reasons,
        }
    return {
        "status": "STATIC_ONLY",
        "run_dir": str(run_dir),
        "precisions": precisions,
        "results": results,
        "note": "Syntax parsing and generation metadata do not establish functional correctness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precisions", nargs="+", default=["q2", "q4", "q8"])
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    payload = audit_run(run_dir, args.precisions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
