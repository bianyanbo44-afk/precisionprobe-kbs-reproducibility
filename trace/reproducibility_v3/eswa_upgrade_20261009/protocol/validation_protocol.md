# Prospective external-validation protocol

Protocol version 1.0. Written on 2026-10-09 (Asia/Shanghai), before any model
generation on this panel. The lock file records the precise UTC time and hashes.
This is a local prospective specification, not a registration with an external
registry.

## Scope and fixed sample

Use all 175 problems in the `test6.jsonl` increment of
`livecodebench/code_generation_lite`, revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`. This is the v6 increment, not the
entire cumulative release_v6 benchmark. The file SHA-256 is
`bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`.
Problems date from 2025-01-04 to 2025-04-06 and include 112 standard-input tasks
and 63 functional tasks. There are 43 easy, 52 medium and 80 hard tasks. No
selection is made using model performance. These tasks are new to this project;
their absence from every model's training data is not claimed.

Evaluate Qwen2.5-Coder-1.5B-Instruct, DeepSeek-Coder-1.3B-Instruct and
Qwen3-4B-Instruct-2507 at Q4_K_M and Q8_0, using the existing pinned artifacts.
Every task is paired across all six conditions. Model and runtime hashes are
recorded in the lock. Use llama.cpp b9637 on the local RTX 4060 Laptop GPU,
one serving slot, context length 6144, and no fine-tuning or repair.

## Generation

One greedy target uses seed 0, temperature 0 and top-p 1. Four alternatives use
seeds 11, 29, 47 and 71, temperature 0.60 and top-p 0.95. All outputs have a
1024-token limit. The problem statement and supplied starter code are shown to
the model. Prompts request a complete Python solution and standard-library
imports. The existing model-specific DeepSeek instruction template is retained.
Extract the first fenced Python code block, if present, otherwise use the raw
response. Do not repair a program after observing tests. Truncations, syntax
errors and runtime failures remain in the results.

Collect target token log probabilities in the same request as the target
program, avoiding deterministic-replay mismatches. Record per-request wall time,
token counts, server timings and finish reason. Warm the server with one
synthetic programming prompt before each condition; the warm-up is excluded.
Use a fixed hashed task order. Timing comparisons are descriptive measurements
of this machine and process schedule, not randomized hardware-speed effects.

## Information available to screening

Execution-disagreement scores use only the public example inputs, in dataset
order, without mutation or hidden-input fallback. Keep both successful values
and failure statuses. Target-anchored DSDE retains the historical bounded
dissimilarity including the 0.15 identical-exception penalty.

The cheap public-example baseline may additionally compare the greedy output
with the expected public outputs, which are available in the problem statement.
Hidden test inputs and outputs are never passed to the model or score builder.

Fixed score families are DSDE with one, two and four alternatives; exact target
execution disagreement; execution-cluster Shannon entropy; target execution
failure fraction; mean token NLL; top-five-plus-residual token entropy;
public-example failure fraction; failure-first then NLL; public-example-failure
first then NLL; and public-example-failure first then DSDE. The guard versions
rank any target that fails a public example before passing targets, then use
the stated uncertainty score. The fixed numeric encoding is `2*guard + x/(1+x)`.
This is an interpretable comparator, not a claimed new learning algorithm.

## Evaluation and execution

Execute generated programs inside a Linux namespace sandbox with no network,
read-only runtime and program mounts, private temporary storage, CPU/memory/file
limits and a wall-clock timeout. Use all supplied public and private tests from
the pinned lite increment. A target is incorrect if any required test fails.
Evaluate greedies only for correctness; alternatives provide the score.

The implementation documents its output comparator and timeout (four seconds per
case). This controlled local evaluation is not an official LiveCodeBench
leaderboard run. Preserve the first failure, number of tests executed, expected
test count, per-probe times and any infrastructure errors. Infrastructure errors
must be fixed or explicitly separated from program failures, never silently
converted to successful cases. Decode dataset pickles with a restricted reader
that rejects class loading and persistent objects.

## Primary comparison

At review capacity 25%, compare residual functional errors per incoming task for
DSDE with four alternatives against public-example-failure-first/NLL. Report the
paired difference in every cell and the equal-weight mean over six cells.
Use 4000 paired task-bootstrap replicates; the same sampled task indices are
used across all models, precisions and comparators. Report two-sided percentile
95% intervals. An interval wholly below zero supports lower residual errors;
an interval crossing zero does not establish that advantage.

Ties at the capacity boundary receive the expected allocation from uniform
sampling inside the equal-score group. Labels never break ties. This estimates
an offline review queue, not human review efficacy or saved human minutes.

## Secondary analyses

- Review fractions 10% and 50%; AUROC and tie-aware AURC.
- Incremental value over target failure and public-example checks.
- Sampling budgets one, two and four alternatives, with measured generation and
  public-execution costs. Report the complete cost/effect frontier.
- Precision-conditioned, pooled and independent calibration rules. Use the
  disjoint 1/4 reference, 1/4 calibration, 1/2 test task split with salt
  `eswa-20261009-split-0`. Alpha values are 0.10, 0.20, 0.30 and 0.40; delta is
  0.10 and the minimum accepted calibration count is 20. Grid and fixed-sequence
  selectors match the archived retrospective specification. Repeat 30 predefined
  split salts descriptively; these do not increase the independent sample size.
- Difficulty and interface-type breakdowns are descriptive and retain all tasks.

Secondary comparisons are estimation exercises, not an unadjusted family of
confirmatory superiority tests. Report all conditions, all infeasible policies
and negative findings. Do not promote a convenient secondary endpoint to primary.
Any post-lock code correction or analysis addition receives a dated amendment
and its effect on the evidence status is stated.

