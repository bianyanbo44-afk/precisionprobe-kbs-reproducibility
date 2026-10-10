"""Run the frozen generator with an exclusive lock and durable record writes.

Only append persistence is changed. Requests, decoding, seeds and candidate
selection still come from the original hashed generator. See Amendment 03.
"""
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import requests

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
RUN = BASE / "runs/livecodebench_v6_increment"
sys.path.insert(0, str(ROOT / "src"))
from precisionprobe import inference

ORIGINAL_GENERATE = inference.LlamaServer.generate


def audited_generate(self, messages, **parameters):
    started = time.perf_counter()
    try:
        return ORIGINAL_GENERATE(self, messages, **parameters)
    except requests.RequestException as exc:
        config = json.loads((BASE / "protocol/validation_config.json").read_text())
        condition = next((m["condition"] for m in config["models"] if Path(m["path"]) == self.model_path), None)
        durable_append(RUN / "logs/generation_request_failures.jsonl", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "condition": condition, "seed": parameters.get("seed"),
            "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": type(exc).__name__ + ": " + str(exc)})
        raise


def durable_append(path, row):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(status, **extra):
    durable_append(RUN / "logs/generation_attempts.jsonl", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "status": status, "wrapper_sha256": digest(Path(__file__)),
        "frozen_generator_sha256": digest(BASE / "scripts/generate_validation.py"), **extra})


def restore_complete_receipts():
    # A crash can occur after the final durable row but before its completion
    # receipt. Validate the complete key set before recreating only that receipt.
    config = json.loads((BASE / "protocol/validation_config.json").read_text())
    ids = {q["task_id"] for q in inference.read_jsonl(BASE / "data/livecodebench/questions.jsonl")}
    expected = {(task, seed) for task in ids for seed in (0, 11, 29, 47, 71)}
    lock_hash = digest(BASE / "protocol/validation_lock.json")
    for spec in config["models"]:
        source = RUN / "generations" / (spec["condition"] + ".jsonl")
        receipt = RUN / (spec["condition"] + "_generation_complete.json")
        if not source.exists() or receipt.exists():
            continue
        rows = inference.read_jsonl(source)
        keys = [(r["task_id"], r["seed"]) for r in rows]
        if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
            raise ValueError("unexpected or duplicate candidate keys")
        if set(keys) != expected:
            continue
        for row in rows:
            if row["condition"] != spec["condition"] or row["model_sha256"] != spec["sha256"] or row["protocol_lock_sha256"] != lock_hash:
                raise ValueError("complete candidate file has wrong provenance")
            if hashlib.sha256(row["solution"].encode()).hexdigest() != row["solution_sha256"]:
                raise ValueError("complete candidate file has wrong solution hash")
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps({"records": len(rows), "sha256": digest(source)}, indent=2), encoding="utf-8")
        temporary.replace(receipt)
        event("restored_completion_receipt", condition=spec["condition"])


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "generation.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        event("started")
        inference.append_jsonl = durable_append
        inference.LlamaServer.generate = audited_generate
        for attempt in range(1, 5):
            # The original server truncates its log when restarting. Preserve
            # every previous log first, including those from failed sessions.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            log_backup = RUN / "logs/server_sessions" / stamp
            for log in (RUN / "logs").glob("*q*.log"):
                if log.is_file():
                    log_backup.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(log, log_backup / log.name)
            try:
                restore_complete_receipts()
                runpy.run_path(str(BASE / "scripts/generate_validation.py"), run_name="__main__")
            except (requests.ConnectionError, requests.Timeout) as exc:
                event("transport_failure", attempt=attempt, error=type(exc).__name__ + ": " + str(exc))
                if attempt == 4:
                    event("retry_limit_reached", attempts=attempt)
                    raise
                event("resuming_missing_records", next_attempt=attempt+1)
            except BaseException as exc:
                event("interrupted_or_error", error=type(exc).__name__ + ": " + str(exc))
                raise
            else:
                event("finished", attempts=attempt)
                break


if __name__ == "__main__":
    main()
