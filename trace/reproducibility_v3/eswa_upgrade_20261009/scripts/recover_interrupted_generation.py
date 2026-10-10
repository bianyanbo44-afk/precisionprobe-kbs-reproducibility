"""Recover only an all-zero trailing extent after a documented system restart.

Every original byte is backed up before replacement. Valid generation records
must have unique expected task/seed keys and matching code/protocol/model hashes.
No correctness labels are read and no intact completion is regenerated.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import generate_validation as generation

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runs/livecodebench_v6_increment"


def main():
    generation.verify_lock()
    config = json.loads((BASE / "protocol/validation_config.json").read_text())
    tasks = {q["task_id"] for q in generation.read_jsonl(BASE / "data/livecodebench/questions.jsonl")}
    expected = {(task, seed) for task in tasks for seed in (0, 11, 29, 47, 71)}
    lock_hash = generation.digest(BASE / "protocol/validation_lock.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = RUN / "recovery" / stamp
    backup.mkdir(parents=True)
    records = []
    for spec in config["models"]:
        path = RUN / "generations" / (spec["condition"] + ".jsonl")
        if not path.exists():
            continue
        original = path.read_bytes()
        good = []
        cutoff = 0
        for line in original.splitlines(keepends=True):
            if b"\x00" in line:
                remainder = original[cutoff:]
                if not remainder or any(remainder):
                    raise ValueError("corruption is not an exclusively zero-filled trailing extent")
                break
            row = json.loads(line)
            if row["condition"] != spec["condition"] or row["model_sha256"] != spec["sha256"]:
                raise ValueError("wrong model identity")
            if row["protocol_lock_sha256"] != lock_hash:
                raise ValueError("wrong locked protocol")
            if hashlib.sha256(row["solution"].encode()).hexdigest() != row["solution_sha256"]:
                raise ValueError("valid-looking record has a code-hash mismatch")
            if row["seed"] == 0 and "token_confidence" not in row:
                raise ValueError("target lacks confidence record")
            good.append(row)
            cutoff += len(line)
        keys = [(row["task_id"], row["seed"]) for row in good]
        if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
            raise ValueError("duplicate or unexpected candidate keys")
        shutil.copy2(path, backup / path.name)
        recovered = original[:cutoff]
        if cutoff != len(original):
            temporary = path.with_suffix(".recovered.tmp")
            with temporary.open("wb") as handle:
                handle.write(recovered)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        complete_tasks = sum(all((task, seed) in set(keys) for seed in (0, 11, 29, 47, 71)) for task in tasks)
        records.append({"condition": spec["condition"], "original_bytes": len(original),
                        "original_sha256": hashlib.sha256(original).hexdigest(),
                        "retained_bytes": cutoff, "unrecoverable_zero_bytes": len(original)-cutoff,
                        "retained_sha256": hashlib.sha256(recovered).hexdigest(),
                        "retained_records": len(good), "complete_tasks": complete_tasks,
                        "last_retained_created_at_utc": good[-1]["created_at_utc"] if good else None})
    for filename in ("generation_status.json", "pipeline_status.json"):
        path = RUN / filename
        if path.exists():
            shutil.copy2(path, backup / filename)
    for path in (RUN / "logs").glob("*"):
        if path.is_file():
            (backup / "logs").mkdir(exist_ok=True)
            shutil.copy2(path, backup / "logs" / path.name)
    receipt = {"recovered_at_utc": datetime.now(timezone.utc).isoformat(),
               "reason": "Windows system restart; stale processes absent; all-zero trailing generation extent",
               "original_backup_directory": str(backup.relative_to(BASE)),
               "script_sha256": generation.digest(__file__), "conditions": records,
               "protocol_unchanged": True, "correctness_outcomes_read": False}
    (backup / "recovery_manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (RUN / "last_recovery.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if records:
        current = records[-1]
        status = {"condition": current["condition"], "completed_tasks": current["complete_tasks"],
                  "total_tasks": len(tasks), "updated_at_utc": receipt["recovered_at_utc"],
                  "stage": "recovered_pending_resume", "retained_records": current["retained_records"]}
        (RUN / "generation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
