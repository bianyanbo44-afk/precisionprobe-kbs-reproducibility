# Amendment 03: system restart and persistence recovery

Recorded on 2026-10-09, before any prospective program correctness evaluation.
The original protocol, configuration, generator, evaluator and scoring hashes
remain unchanged. This is an operational deviation, not a change of endpoints,
sampling budget, task selection or decoding parameters.

At the 2026-10-09 01:35 UTC continuation check, no generator, llama.cpp server or
evaluation watcher was running. Windows reported its last boot at 2026-10-09
02:56:59 Asia/Shanghai, shortly after the last preserved generation timestamp.
The generation status file contained 137 zero bytes. The Qwen3 Q4 candidate
file contained 359 complete JSON records followed by an exclusively zero-filled
1,861-byte extent. No other validation condition had begun, and no validation
correctness records or completion marker existed.

Every original file, including the damaged candidate file and logs, was copied
before recovery. The recovery receipt is
`runs/livecodebench_v6_increment/recovery/20261009T013659Z/recovery_manifest.json`.
It records original and recovered hashes and the byte cutoff. All 359 intact
records have valid unique task/seed keys, matching solution, model and protocol
hashes, and target confidence where required. They comprise 71 complete tasks
and four candidates of the next task. Those bytes are retained exactly. The
zero-filled extent has no recoverable completion and is not a program outcome
excluded because of performance. The missing candidate is generated using the
same task, prompt, seed and settings. An identical unpreserved output is not
claimed.

The run resumes with the original generator through
`scripts/run_generation_durable.py`. This wrapper serializes candidate records
in the same way as the original append helper, adds an explicit flush/fsync and
an exclusive generation-process lock, and records each generation attempt.
It also validates and restores a missing completion receipt if all 875 records
of a condition are already present after a future interruption. It does not
alter model requests, outputs, scores or correctness labels. The wrapper and
recovery script are retained with their own hashes.

A synthetic warm-up is performed after each server start, including the resumed
Qwen3 Q4 server, and excluded from per-request acquisition costs. Restart
downtime is not part of an individual request's elapsed time. Retained request
timings consequently come from interrupted sessions with potentially different
caching and concurrent activity. They remain descriptive measurements of this
schedule and cannot establish an uninterrupted-service latency or randomized
precision speedup. This limitation is included in the manuscript and supplement.
