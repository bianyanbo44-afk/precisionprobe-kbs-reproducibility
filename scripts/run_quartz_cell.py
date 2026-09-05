from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from precisionprobe.data import build_generation_messages, choose_probe_inputs, extract_python_source, load_benchmark
from precisionprobe.execution import run_candidate, run_candidate_wsl
from precisionprobe.inference import LlamaServer, append_jsonl, read_jsonl
from precisionprobe.scoring import (
    ast_distance,
    dominant_semantic_distance_entropy,
    exact_dominant_semantic_distance_entropy,
    exact_semantic_distance_entropy,
    semantic_distance_entropy,
)
try:
    from run_pilot import find_server, load_model_manifest, sha256_file, sha256_text
except ModuleNotFoundError:  # Imported as scripts.run_quartz_cell in tests.
    from scripts.run_pilot import find_server, load_model_manifest, sha256_file, sha256_text


MANIFEST_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def generation_artifacts(
    root: Path,
    run_dir: Path,
    precisions: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        precision: artifact_record(run_dir / "generations" / f"{precision}.jsonl", root)
        for precision in precisions
    }


def scoring_artifacts(root: Path, run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "scores": artifact_record(run_dir / "scores.jsonl", root),
        "evalplus_samples": artifact_record(run_dir / "evalplus_samples.jsonl", root),
    }


def code_hashes(root: Path) -> dict[str, str]:
    return {
        "run_quartz_cell.py": sha256_file(Path(__file__).resolve()),
        "execution.py": sha256_file(root / "src" / "precisionprobe" / "execution.py"),
        "inference.py": sha256_file(root / "src" / "precisionprobe" / "inference.py"),
        "scoring.py": sha256_file(root / "src" / "precisionprobe" / "scoring.py"),
        "data.py": sha256_file(root / "src" / "precisionprobe" / "data.py"),
    }


def validate_manifest_identity(existing: dict[str, Any], current: dict[str, Any]) -> None:
    mismatches = []
    for key in ("study", "config_sha256", "task_ids"):
        if existing.get(key) != current[key]:
            mismatches.append(key)
    existing_models = existing.get("models", {})
    for precision, model in current["models"].items():
        if existing_models.get(precision, {}).get("sha256") != model.get("sha256"):
            mismatches.append(f"models.{precision}.sha256")
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"existing run manifest is incompatible with this invocation: {joined}")


def prepare_manifest(
    existing: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    mode: str,
    started_at: str,
) -> tuple[dict[str, Any], int]:
    if existing is None:
        manifest = copy.deepcopy(current)
        manifest["created_at"] = started_at
    else:
        validate_manifest_identity(existing, current)
        manifest = copy.deepcopy(existing)
        if int(manifest.get("manifest_schema_version", 1)) < MANIFEST_SCHEMA_VERSION:
            manifest.setdefault("manifest_history", []).append(
                {
                    "event": "legacy_manifest_preserved",
                    "recorded_at": started_at,
                    "snapshot": copy.deepcopy(existing),
                }
            )
        for key, value in current.items():
            manifest[key] = copy.deepcopy(value)

    manifest["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["updated_at"] = started_at
    phases = manifest.setdefault("phases", {})
    phases.setdefault("generation", [])
    phases.setdefault("scoring", [])
    phases.setdefault("evaluation", [])
    invocations = manifest.setdefault("invocations", [])
    invocation_id = max((int(row.get("id", 0)) for row in invocations), default=0) + 1
    invocations.append(
        {
            "id": invocation_id,
            "mode": mode,
            "started_at": started_at,
            "status": "RUNNING",
            "config_sha256": current["config_sha256"],
            "code_sha256": copy.deepcopy(current["code_sha256"]),
        }
    )
    return manifest, invocation_id


def append_phase(
    manifest: dict[str, Any],
    phase: str,
    record: dict[str, Any],
) -> None:
    manifest.setdefault("phases", {}).setdefault(phase, []).append(copy.deepcopy(record))
    manifest["updated_at"] = record["recorded_at"]


def finish_invocation(
    manifest: dict[str, Any],
    invocation_id: int,
    *,
    status: str,
    finished_at: str,
    error_type: str | None = None,
) -> None:
    invocation = next(row for row in manifest["invocations"] if int(row["id"]) == invocation_id)
    invocation["status"] = status
    invocation["finished_at"] = finished_at
    if error_type is not None:
        invocation["error_type"] = error_type
    manifest["updated_at"] = finished_at


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def task_ids_from_config(problems: dict[str, Any], config: dict[str, Any]) -> list[str]:
    task_ids = sorted(problems)
    limit = config.get("task_limit")
    if limit is None:
        return task_ids
    random.Random(int(config["task_seed"])).shuffle(task_ids)
    return task_ids[: int(limit)]


def generate(
    root: Path,
    run_dir: Path,
    problems: dict[str, Any],
    task_ids: list[str],
    config: dict[str, Any],
    models: dict[str, Any],
) -> None:
    executable = find_server(root)
    specifications = [("greedy", 0, 0.0, 1.0)] + [
        ("alternative", int(seed), float(config["temperature"]), float(config["top_p"]))
        for seed in config["alternative_seeds"]
    ]
    for offset, precision in enumerate(config["precisions"]):
        output = run_dir / "generations" / f"{precision}.jsonl"
        completed = {(row["task_id"], row["kind"], int(row["seed"])) for row in read_jsonl(output)}
        model = models[config["models"][precision]]
        with LlamaServer(
            executable,
            model["path"],
            port=8171 + offset,
            log_path=run_dir / "logs" / f"llama-server-{precision}.log",
            chat_template=config.get("chat_template"),
            completion_prompt_style=config.get("completion_prompt_style"),
        ) as server:
            for task_index, task_id in enumerate(task_ids, start=1):
                messages = build_generation_messages(problems[task_id])
                for kind, seed, temperature, top_p in specifications:
                    if (task_id, kind, seed) in completed:
                        continue
                    print(f"[{precision}] {task_index}/{len(task_ids)} {task_id} {kind} seed={seed}")
                    generated = server.generate(
                        messages,
                        seed=seed,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=int(config["max_tokens"]),
                    )
                    solution = extract_python_source(generated["text"], problems[task_id])
                    append_jsonl(
                        output,
                        {
                            "precision": precision,
                            "task_id": task_id,
                            "kind": kind,
                            "seed": seed,
                            "temperature": temperature,
                            "solution": solution,
                            "solution_sha256": sha256_text(solution),
                            "raw_text": generated["text"],
                            "finish_reason": generated["finish_reason"],
                            "usage": generated["usage"],
                            "timings": generated["timings"],
                            "elapsed_seconds": generated["elapsed_seconds"],
                        },
                    )


def score(
    run_dir: Path,
    problems: dict[str, Any],
    task_ids: list[str],
    config: dict[str, Any],
    *,
    execution_mode: str = "windows",
) -> None:
    if execution_mode not in {"windows", "wsl"}:
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    candidate_runner = run_candidate_wsl if execution_mode == "wsl" else run_candidate
    generations = {
        precision: {
            (row["task_id"], row["kind"], int(row["seed"])): row
            for row in read_jsonl(run_dir / "generations" / f"{precision}.jsonl")
        }
        for precision in config["precisions"]
    }
    score_path = run_dir / "scores.jsonl"
    samples_path = run_dir / "evalplus_samples.jsonl"
    for path in (score_path, samples_path):
        if path.exists():
            path.unlink()
    for task_index, task_id in enumerate(task_ids, start=1):
        problem = problems[task_id]
        probes, source = choose_probe_inputs(
            problem,
            max_inputs=int(config["probe_max_inputs"]),
            min_prompt_inputs=int(config["probe_min_prompt_inputs"]),
            allow_base_fallback=bool(config["allow_base_input_fallback"]),
        )
        payload: dict[str, Any] = {
            "task_id": task_id,
            "probe_source": source,
            "probe_count": len(probes),
            "probe_hash": sha256_text(repr(probes)),
        }
        for precision in config["precisions"]:
            execution_started = time.perf_counter()
            candidates = [generations[precision][(task_id, "greedy", 0)]] + [
                generations[precision][(task_id, "alternative", int(seed))]
                for seed in config["alternative_seeds"]
            ]
            observations = [
                candidate_runner(
                    candidate["solution"],
                    problem["entry_point"],
                    probes,
                    timeout_seconds=float(config["execution_timeout_seconds"]),
                )["observations"]
                for candidate in candidates
            ]
            payload[f"{precision}_execution_seconds"] = time.perf_counter() - execution_started
            payload[f"{precision}_sde"] = semantic_distance_entropy(observations)
            payload[f"{precision}_dsde"] = dominant_semantic_distance_entropy(observations[0], observations)
            payload[f"{precision}_exact_sde"] = exact_semantic_distance_entropy(observations)
            payload[f"{precision}_exact_dsde"] = exact_dominant_semantic_distance_entropy(
                observations[0], observations
            )
            payload[f"{precision}_ast_mean"] = sum(
                ast_distance(candidates[0]["solution"], candidate["solution"])
                for candidate in candidates[1:]
            ) / (len(candidates) - 1)
            payload[f"{precision}_distinct_programs"] = len(
                {candidate["solution_sha256"] for candidate in candidates}
            )
            payload[f"{precision}_observations"] = observations
            payload[f"{precision}_generation_seconds"] = [
                candidate["elapsed_seconds"] for candidate in candidates
            ]
            append_jsonl(
                samples_path,
                {
                    "sample_id": f"{task_id}|{precision}_greedy",
                    "task_id": task_id,
                    "solution": candidates[0]["solution"],
                },
            )
        append_jsonl(score_path, payload)
        score_summary = " ".join(
            f"{precision}={payload[f'{precision}_dsde']:.3f}"
            for precision in config["precisions"]
        )
        print(
            f"score {task_index}/{len(task_ids)} {task_id}: "
            f"{score_summary}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quartz_qwen_mbpp.yaml")
    parser.add_argument("--model-manifest", default="runs/model_manifest.json")
    parser.add_argument("--run-dir", default="runs/quartz_qwen_mbpp")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--score-only", action="store_true")
    mode_group.add_argument(
        "--generation-only",
        action="store_true",
        help="generate candidates and record generation without executing them",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("windows", "wsl"),
        default="windows",
        help="where generated candidates are executed during scoring",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    problems = load_benchmark(config["dataset"], mini=bool(config["mini"]))
    task_ids = task_ids_from_config(problems, config)
    models = load_model_manifest(root / args.model_manifest)
    run_dir = root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    started_at = utc_now()
    executable = find_server(root)
    current = {
        "study": config["study"],
        "config": args.config,
        "config_sha256": sha256_file(config_path),
        "model_manifest": args.model_manifest,
        "execution_mode": args.execution_mode,
        "models": {precision: models[config["models"][precision]] for precision in config["precisions"]},
        "task_ids": task_ids,
        "llama_server_sha256": sha256_file(executable),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "base_executable": getattr(sys, "_base_executable", sys.executable),
        },
        "code_sha256": code_hashes(root),
    }
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    manifest, invocation_id = prepare_manifest(
        existing,
        current,
        mode=(
            "score_only"
            if args.score_only
            else "generation_only"
            if args.generation_only
            else "generate_and_score"
        ),
        started_at=started_at,
    )
    write_manifest(manifest_path, manifest)

    try:
        if not args.score_only:
            generate(root, run_dir, problems, task_ids, config, models)
            recorded_at = utc_now()
            append_phase(
                manifest,
                "generation",
                {
                    "recorded_at": recorded_at,
                    "status": "COMPLETED",
                    "invocation_id": invocation_id,
                    "config_sha256": current["config_sha256"],
                    "execution_mode": args.execution_mode,
                    "models": copy.deepcopy(current["models"]),
                    "expected_tasks": len(task_ids),
                    "task_ids_sha256": sha256_text(
                        json.dumps(task_ids, ensure_ascii=False, separators=(",", ":"))
                    ),
                    "llama_server_sha256": current["llama_server_sha256"],
                    "code_sha256": copy.deepcopy(current["code_sha256"]),
                    "outputs": generation_artifacts(root, run_dir, list(config["precisions"])),
                },
            )
            write_manifest(manifest_path, manifest)
        elif not manifest["phases"]["generation"]:
            source = existing if existing is not None else current
            recorded_at = utc_now()
            append_phase(
                manifest,
                "generation",
                {
                    "recorded_at": recorded_at,
                    "status": "CAPTURED_EXISTING",
                    "invocation_id": invocation_id,
                    "source_manifest_created_at": source.get("created_at"),
                    "config_sha256": source["config_sha256"],
                    "models": copy.deepcopy(source["models"]),
                    "expected_tasks": len(source["task_ids"]),
                    "task_ids_sha256": sha256_text(
                        json.dumps(
                            source["task_ids"], ensure_ascii=False, separators=(",", ":")
                        )
                    ),
                    "llama_server_sha256": source["llama_server_sha256"],
                    "code_sha256": copy.deepcopy(source.get("code_sha256", {})),
                    "outputs": generation_artifacts(root, run_dir, list(config["precisions"])),
                },
            )
            write_manifest(manifest_path, manifest)

        recorded_at = utc_now()
        if not args.generation_only:
            score_inputs = generation_artifacts(root, run_dir, list(config["precisions"]))
            score(run_dir, problems, task_ids, config, execution_mode=args.execution_mode)
            recorded_at = utc_now()
            append_phase(
                manifest,
                "scoring",
                {
                    "recorded_at": recorded_at,
                    "status": "COMPLETED",
                    "invocation_id": invocation_id,
                    "config_sha256": current["config_sha256"],
                    "execution_mode": args.execution_mode,
                    "expected_tasks": len(task_ids),
                    "code_sha256": copy.deepcopy(current["code_sha256"]),
                    "inputs": {"generations": score_inputs},
                    "outputs": scoring_artifacts(root, run_dir),
                },
            )
        finish_invocation(
            manifest,
            invocation_id,
            status="COMPLETED",
            finished_at=recorded_at,
        )
        write_manifest(manifest_path, manifest)
    except BaseException as exc:
        finish_invocation(
            manifest,
            invocation_id,
            status="FAILED",
            finished_at=utc_now(),
            error_type=type(exc).__name__,
        )
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
