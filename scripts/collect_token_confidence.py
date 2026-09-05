from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from precisionprobe.data import build_generation_messages, extract_python_source, load_benchmark
from precisionprobe.inference import LlamaServer, append_jsonl, read_jsonl

try:
    from run_pilot import find_server, load_model_manifest, sha256_file, sha256_text
except ModuleNotFoundError:
    from scripts.run_pilot import find_server, load_model_manifest, sha256_file, sha256_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_token_rows(logprobs: Any) -> list[dict[str, Any]]:
    if not isinstance(logprobs, dict):
        return []
    if isinstance(logprobs.get("content"), list):
        return [row for row in logprobs["content"] if isinstance(row, dict)]

    tokens = logprobs.get("tokens") or []
    chosen = logprobs.get("token_logprobs") or []
    alternatives = logprobs.get("top_logprobs") or []
    rows = []
    for index, token in enumerate(tokens):
        top = alternatives[index] if index < len(alternatives) else {}
        top_rows = [
            {"token": str(name), "logprob": float(value)}
            for name, value in (top.items() if isinstance(top, dict) else [])
        ]
        rows.append(
            {
                "token": str(token),
                "logprob": float(chosen[index]) if index < len(chosen) else float("nan"),
                "top_logprobs": top_rows,
            }
        )
    return rows


def summarize_logprobs(logprobs: Any) -> dict[str, Any]:
    rows = normalized_token_rows(logprobs)
    chosen = np.asarray(
        [float(row["logprob"]) for row in rows if math.isfinite(float(row["logprob"]))],
        dtype=float,
    )
    if chosen.size == 0:
        raise ValueError("the server response did not contain finite token log-probabilities")

    entropies = []
    margins = []
    for row in rows:
        top = [
            float(item["logprob"])
            for item in row.get("top_logprobs", [])
            if math.isfinite(float(item["logprob"]))
        ]
        if top:
            probabilities = np.exp(np.asarray(top, dtype=float))
            residual = max(0.0, 1.0 - float(probabilities.sum()))
            entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum())
            if residual > 0.0:
                entropy -= residual * math.log(residual)
            entropies.append(entropy)
            ordered = sorted(top, reverse=True)
            margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else float("nan"))

    finite_margins = np.asarray([value for value in margins if math.isfinite(value)], dtype=float)
    return {
        "token_count": int(chosen.size),
        "mean_nll": float(-chosen.mean()),
        "median_nll": float(-np.median(chosen)),
        "worst_decile_nll": float(-np.quantile(chosen, 0.10)),
        "sequence_nll": float(-chosen.sum()),
        "mean_topk_entropy": float(np.mean(entropies)) if entropies else None,
        "mean_top1_margin": float(finite_margins.mean()) if finite_margins.size else None,
        "chosen_logprobs": chosen.tolist(),
        "topk_entropies": entropies,
        "top1_margins": finite_margins.tolist(),
    }


def completion_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    run_dir = root / args.run_dir
    output_path = run_dir / "token_confidence.jsonl"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    problems = load_benchmark(config["dataset"], mini=bool(config["mini"]))
    models = load_model_manifest(root / args.model_manifest)
    completed = {
        (str(row["task_id"]), str(row["precision"]))
        for row in read_jsonl(output_path)
        if row.get("status") == "matched"
    }

    for offset, precision in enumerate(config["precisions"]):
        frozen_rows = {
            str(row["task_id"]): row
            for row in read_jsonl(run_dir / "generations" / f"{precision}.jsonl")
            if row["kind"] == "greedy" and int(row["seed"]) == 0
        }
        if not frozen_rows:
            raise FileNotFoundError(f"no frozen greedy generations for {precision}")
        model = models[config["models"][precision]]
        with LlamaServer(
            find_server(root),
            model["path"],
            port=int(args.port_base) + offset,
            log_path=run_dir / "logs" / f"token-confidence-{precision}.log",
            chat_template=config.get("chat_template"),
            completion_prompt_style=config.get("completion_prompt_style"),
        ) as server:
            for index, task_id in enumerate(sorted(frozen_rows), start=1):
                if (task_id, precision) in completed:
                    continue
                generated = server.generate(
                    build_generation_messages(problems[task_id]),
                    seed=0,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(config["max_tokens"]),
                    return_logprobs=True,
                    top_logprobs=int(args.top_logprobs),
                )
                solution = extract_python_source(generated["text"], problems[task_id])
                expected_hash = str(frozen_rows[task_id]["solution_sha256"])
                actual_hash = sha256_text(solution)
                summary = summarize_logprobs(generated["logprobs"])
                row = {
                    "created_at": utc_now(),
                    "task_id": task_id,
                    "precision": precision,
                    "status": "matched" if actual_hash == expected_hash else "hash_mismatch",
                    "expected_solution_sha256": expected_hash,
                    "generated_solution_sha256": actual_hash,
                    "raw_completion_sha256": completion_hash(generated["text"]),
                    "model_sha256": model["sha256"],
                    "elapsed_seconds": generated["elapsed_seconds"],
                    **summary,
                }
                append_jsonl(output_path, row)
                print(
                    f"[{precision}] {index}/{len(frozen_rows)} {task_id} "
                    f"{row['status']} mean_nll={row['mean_nll']:.4f}"
                )

    rows = read_jsonl(output_path)
    expected_count = sum(
        1
        for precision in config["precisions"]
        for row in read_jsonl(run_dir / "generations" / f"{precision}.jsonl")
        if row["kind"] == "greedy" and int(row["seed"]) == 0
    )
    matched = sum(row.get("status") == "matched" for row in rows)
    manifest = {
        "created_at": utc_now(),
        "config": args.config,
        "config_sha256": sha256_file(config_path),
        "model_manifest": args.model_manifest,
        "run_dir": args.run_dir,
        "top_logprobs": int(args.top_logprobs),
        "expected_rows": expected_count,
        "recorded_rows": len(rows),
        "matched_rows": matched,
        "status": "complete" if len(rows) == expected_count and matched == expected_count else "incomplete",
        "output_sha256": sha256_file(output_path),
    }
    (run_dir / "token_confidence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--port-base", type=int, default=8271)
    parser.add_argument("--top-logprobs", type=int, default=5)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
