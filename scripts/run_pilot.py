from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from precisionprobe.data import (
    build_generation_messages,
    choose_probe_inputs,
    extract_python_source,
    load_benchmark,
)
from precisionprobe.execution import run_candidate
from precisionprobe.inference import LlamaServer, append_jsonl, read_jsonl
from precisionprobe.scoring import ast_distance, failure_fraction, xpbd_score


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["label"]: row for row in payload["models"]}


def find_server(project_root: Path) -> Path:
    candidates = list((project_root / ".tools" / "llama.cpp-b9637").rglob("llama-server.exe"))
    if not candidates:
        raise FileNotFoundError("llama-server.exe not found under .tools/llama.cpp-b9637")
    return candidates[0]


def select_tasks(problems: dict[str, dict[str, Any]], limit: int, seed: int) -> list[str]:
    task_ids = sorted(problems)
    random.Random(seed).shuffle(task_ids)
    return task_ids[: min(limit, len(task_ids))]


def collect_hardware() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = subprocess.check_output(command, text=True, timeout=10).strip()
    except Exception as exc:
        gpu = f"unavailable:{type(exc).__name__}"
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": gpu,
    }


def generate_condition(
    *,
    label: str,
    model: dict[str, Any],
    executable: Path,
    problems: dict[str, dict[str, Any]],
    task_ids: list[str],
    config: dict[str, Any],
    run_dir: Path,
    port: int,
) -> None:
    output_path = run_dir / "generations" / f"{label}.jsonl"
    completed = {(row["task_id"], int(row["seed"])) for row in read_jsonl(output_path)}
    server = LlamaServer(
        executable,
        model["path"],
        port=port,
        log_path=run_dir / "logs" / f"llama-server-{label}.log",
    )
    with server:
        for task_index, task_id in enumerate(task_ids, start=1):
            problem = problems[task_id]
            messages = build_generation_messages(problem)
            prompt_hash = sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
            for seed in config["generation_seeds"]:
                if (task_id, int(seed)) in completed:
                    continue
                print(f"[{label}] {task_index}/{len(task_ids)} {task_id} seed={seed}")
                generation = server.generate(
                    messages,
                    seed=int(seed),
                    temperature=float(config["temperature"]),
                    top_p=float(config["top_p"]),
                    max_tokens=int(config["max_tokens"]),
                )
                solution = extract_python_source(generation["text"], problem)
                append_jsonl(
                    output_path,
                    {
                        "condition": label,
                        "task_id": task_id,
                        "seed": int(seed),
                        "prompt_hash": prompt_hash,
                        "raw_text": generation["text"],
                        "solution": solution,
                        "solution_sha256": sha256_text(solution),
                        "finish_reason": generation["finish_reason"],
                        "usage": generation["usage"],
                        "timings": generation["timings"],
                        "elapsed_seconds": generation["elapsed_seconds"],
                    },
                )


def make_scores(
    problems: dict[str, dict[str, Any]],
    task_ids: list[str],
    config: dict[str, Any],
    run_dir: Path,
) -> None:
    generations = {}
    for label in config["models"]:
        for row in read_jsonl(run_dir / "generations" / f"{label}.jsonl"):
            generations[(label, row["task_id"], int(row["seed"]))] = row
    seed_a, seed_b = map(int, config["generation_seeds"][:2])
    score_path = run_dir / "scores.jsonl"
    if score_path.exists():
        score_path.unlink()
    samples_path = run_dir / "evalplus_samples.jsonl"
    if samples_path.exists():
        samples_path.unlink()

    for index, task_id in enumerate(task_ids, start=1):
        problem = problems[task_id]
        probes, probe_source = choose_probe_inputs(
            problem,
            max_inputs=int(config["probe_max_inputs"]),
            min_prompt_inputs=int(config["probe_min_prompt_inputs"]),
            allow_base_fallback=bool(config["allow_base_input_fallback"]),
        )
        rows = {
            "q4_a": generations[("q4", task_id, seed_a)],
            "q4_b": generations[("q4", task_id, seed_b)],
            "q8_a": generations[("q8", task_id, seed_a)],
            "q8_b": generations[("q8", task_id, seed_b)],
        }
        executions = {}
        for name, row in rows.items():
            executions[name] = run_candidate(
                row["solution"],
                problem["entry_point"],
                probes,
                timeout_seconds=float(config["execution_timeout_seconds"]),
            )
            append_jsonl(
                samples_path,
                {
                    "sample_id": f"{task_id}|{name}",
                    "task_id": task_id,
                    "solution": row["solution"],
                },
            )

        cross = xpbd_score(
            executions["q4_a"]["observations"], executions["q8_a"]["observations"]
        )
        same_q4 = xpbd_score(
            executions["q4_a"]["observations"], executions["q4_b"]["observations"]
        )
        same_q8 = xpbd_score(
            executions["q8_a"]["observations"], executions["q8_b"]["observations"]
        )
        append_jsonl(
            score_path,
            {
                "task_id": task_id,
                "probe_source": probe_source,
                "probe_count": len(probes),
                "probe_hash": sha256_text(repr(probes)),
                "xpbd_cross_q4_q8": cross,
                "xpbd_same_q4": same_q4,
                "xpbd_same_q8": same_q8,
                "ast_cross_q4_q8": ast_distance(rows["q4_a"]["solution"], rows["q8_a"]["solution"]),
                "ast_same_q4": ast_distance(rows["q4_a"]["solution"], rows["q4_b"]["solution"]),
                "q4_a_probe_failure_fraction": failure_fraction(executions["q4_a"]["observations"]),
                "q4_cross_code_changed": rows["q4_a"]["solution_sha256"] != rows["q8_a"]["solution_sha256"],
                "q4_same_code_changed": rows["q4_a"]["solution_sha256"] != rows["q4_b"]["solution_sha256"],
                "execution": executions,
                "generation_elapsed": {name: row["elapsed_seconds"] for name, row in rows.items()},
            },
        )
        print(f"score {index}/{len(task_ids)} {task_id}: cross={cross:.3f} same_q4={same_q4:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--model-manifest", default="runs/model_manifest.json")
    parser.add_argument("--run-dir", default="runs/pilot_qwen15b")
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_manifest_path = project_root / args.model_manifest
    models = load_model_manifest(model_manifest_path)
    run_dir = project_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    problems = load_benchmark(config["dataset"], mini=bool(config["mini"]))
    task_ids = select_tasks(problems, int(config["task_limit"]), int(config["task_seed"]))
    executable = find_server(project_root)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_manifest_path": str(model_manifest_path),
        "llama_server": str(executable),
        "llama_server_sha256": sha256_file(executable),
        "task_ids": task_ids,
        "hardware": collect_hardware(),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for offset, label in enumerate(("q4", "q8")):
        generate_condition(
            label=label,
            model=models[label],
            executable=executable,
            problems=problems,
            task_ids=task_ids,
            config=config,
            run_dir=run_dir,
            port=8091 + offset,
        )
    if not args.generation_only:
        make_scores(problems, task_ids, config, run_dir)


if __name__ == "__main__":
    main()

