"""Retain raw cost and add a labeled machine-suspension sensitivity (Amendment 05).

This does not change outputs, functional labels, primary comparisons or the
original cost frontier. It requires the entire finished validation panel.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from analyze_retrospective import digest, read_jsonl
from analyze_validation import BASE, SCORES, cost_tables, load_complete


def merged_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("nonpositive machine-sleep interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def sleep_overlap(ended_at_utc, elapsed_seconds, intervals):
    end = datetime.fromisoformat(ended_at_utc)
    start = end - timedelta(seconds=elapsed_seconds)
    overlap = sum(max(0., (min(end, b)-max(start, a)).total_seconds()) for a, b in merged_intervals(intervals))
    return min(max(0., elapsed_seconds), overlap)


def main():
    frame, input_manifest = load_complete()
    run = BASE / "runs/livecodebench_v6_increment"
    output = BASE / "analysis/validation"
    intervals_path = run / "system_sleep_intervals.json"
    intervals_data = json.loads(intervals_path.read_text(encoding="utf-8-sig")) if intervals_path.exists() else {"intervals": []}
    intervals = [(datetime.fromisoformat(x["start_utc"]), datetime.fromisoformat(x["end_utc"])) for x in intervals_data["intervals"]]
    config = json.loads((BASE / "protocol/validation_config.json").read_text())
    adjusted = frame.copy(deep=True)
    details, inventory = [], []
    for model in config["models"]:
        condition = model["condition"]
        generations = read_jsonl(run / "generations" / (condition+".jsonl"))
        candidates = {(r["task_id"], r["seed"]): r for r in generations}
        all_sleep = 0.
        affected = 0
        for index, row in adjusted[adjusted.condition == condition].iterrows():
            candidates_for_task = [candidates[(row.task_id, seed)] for seed in (0, 11, 29, 47, 71)]
            times = []
            for candidate in candidates_for_task:
                raw = candidate["elapsed_seconds"]
                overlap = sleep_overlap(candidate["created_at_utc"], raw, intervals)
                times.append(raw - overlap)
                if overlap:
                    all_sleep += overlap
                    affected += 1
                    details.append({"condition": condition, "task_id": row.task_id, "seed": candidate["seed"],
                                    "raw_request_seconds": raw, "documented_sleep_overlap_seconds": overlap,
                                    "available_machine_request_seconds": raw-overlap,
                                    "created_at_utc": candidate["created_at_utc"]})
            adjusted.loc[index, "greedy_generation_seconds"] = times[0]
            for k in (1, 2, 4):
                adjusted.loc[index, f"generation_seconds_{k}"] = sum(times[:k+1])
        inventory.append({"condition": condition, "completed_requests": len(generations),
                          "affected_requests": affected, "documented_sleep_overlap_seconds": all_sleep})
    cost_tables(adjusted).to_csv(output / "cost_frontier_available_machine_sensitivity.csv", index=False)
    pd.DataFrame(details, columns=["condition", "task_id", "seed", "raw_request_seconds",
                                 "documented_sleep_overlap_seconds", "available_machine_request_seconds", "created_at_utc"]).to_csv(
                                     output / "machine_sleep_affected_requests.csv", index=False)
    failures_path = run / "logs/generation_request_failures.jsonl"
    later_failures = read_jsonl(failures_path) if failures_path.exists() else []
    pd.DataFrame(inventory).to_csv(output / "operational_inventory.csv", index=False)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
              "amendment_05_sha256": digest(BASE / "protocol/amendment_05_machine_sleep.md"),
              "sleep_event_source_sha256": digest(intervals_path) if intervals_path.exists() else None,
              "original_cost_frontier_sha256": digest(output / "cost_frontier.csv"),
              "input_manifest": input_manifest, "conditions": inventory,
              "later_recorded_transport_failures": later_failures,
              "exact_costs_missing_for_earlier_failures": True,
              "interpretation": "Secondary sensitivity subtracting overlap with documented machine suspension from completed-request times. Excludes model load, warmup, downtime and failed requests; not a total operating-cost estimate. Raw costs and scientific results remain unchanged."}
    (output / "operational_cost_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2), flush=True)


if __name__ == "__main__":
    main()
