# Data dictionary and interpretation

## Units, keys and populations

The unit is a programming task with one served greedy target, at a specified
model and precision. Alternatives supply screening information; they are not
independent target observations. `task_id` is paired across precisions and,
in the new panel, across all six model/precision conditions.

The retrospective feature table has 2,288 task/precision records: four full
HumanEval+/MBPP+ model panels and a 60-task Qwen3 MBPP+ subset at each precision.
The subset overlaps the MBPP+ task collection. The new table has 1,050 records:
175 tasks at six conditions. Its 5,250 generation records use seeds 0, 11, 29,
47 and 71. Seed 0 is the target; alternatives have that fixed prefix order.

`panel`, `model`, `dataset`, `precision` and `condition` identify populations.
Q4 and Q8 denote pinned Q4_K_M and Q8_0 artifacts, not controlled isolation of
every numerical-precision effect. `difficulty` and `interface` are task
metadata. The task metadata CSV supplies original platform, title and date.

## Target and screening fields

| Field | Meaning and units |
|---|---|
| `error` | 1 if the target fails any required correctness test; 0 if all pass. |
| `probe_count`, `probe_source` | Number and source of input probes used for disagreement. |
| `token_replay_status` | Archived confidence join status. Two mismatched Qwen3 Q4 targets lack usable token-dependent scores. |
| `mean_nll` | Mean negative log probability of generated response tokens, natural-log units. |
| `mean_topk_entropy` | Top-five entropy with the residual mass in one category; not full-vocabulary entropy. |
| `greedy_failure` | Fraction of probes where target execution status is not successful. A successful wrong output is not an execution failure. |
| `public_failure` | Fraction of supplied public examples where target correctness checking fails. |
| `dsde_1`, `dsde_2`, `dsde_4` | Target-anchored average semantic dissimilarity using the target plus 1, 2 or 4 alternatives. The self-term and fixed exception costs are retained. Larger scores route earlier for review. |
| `exact_4` | Average exact observation disagreement with the target, using all four alternatives. |
| `cluster_entropy` | Natural-log entropy over identical complete observation vectors among all five candidates. |
| `failure_then_nll` | Guard score `2 * execution_failure_present + NLL / (1 + NLL)`. |
| `public_then_nll` | Guard score `2 * public_failure_present + NLL / (1 + NLL)`. Primary prospective comparator. |
| `public_then_dsde` | Public-failure guard followed by normalized DSDE4. Secondary comparator. |
| `solution_sha256` | UTF-8 extracted target solution hash binding confidence and evaluation. |
| `target_finish_reason` | Returned runtime finish reason; length-limited outputs remain included. |
| `created_at_utc` | Initial evaluation creation time; amendment timing is in its separate receipt. |

## Retained execution records

In new evaluation JSONL, `observations` is candidate-by-public-probe order,
and `probe_execution_seconds` has the same shape. Each observation records
status plus output or exception information. `public_test_passes` applies to
the target. `private_test_count` is the required target test count;
`private_tests_checked` is the executed count under private fail-fast checking;
`private_first_failure` is the first failing index, or null. Public failure can
make a target erroneous without running its private tests. Private inputs and
expected outputs are not redistributed in the archive.

The current correctness version is `statement-aware-v2` (Amendment 06). Eight
task-stated output rules were applied to all 48 affected condition/task pairs.
`evaluations_v1_exact_match/` preserves the original records and completion
receipts. `statement_correction_receipt.json` records every before/after
assessment and fresh execution time, excluded from acquisition cost. The
original disagreement observations, generated programs, token scores and
acquisition times are unchanged. Every final error label is preserved; one
public-failure status changes, updating the public-passing diagnostic. Private
evaluation fields reflect the amended execution for the affected targets.

Generation JSONL retains extracted solutions, solution hashes, raw responses,
seed/decoding and model/protocol hashes, same-request token summaries, and
completed-request elapsed seconds. Completion receipts bind row counts and
file SHA-256. Failed transport requests are separate operational records and
are not correctness-guided retries.

## Screening metrics

| Field | Meaning |
|---|---|
| `score` | Screening score family; larger means earlier review. |
| `n`, `errors` | Eligible target count and total observed errors. |
| `excluded_unmatched` | Number excluded only for unavailable hash-matched token confidence. |
| `budget` | Nominal review fraction: 0.10, 0.25 or 0.50. |
| `reviewed`, `review_fraction` | `ceil(budget * n)` slots and actual fraction. |
| `reviewed_errors_expected` | Expected errors captured under uniform allocation at a score tie boundary. |
| `error_capture` | Expected captured errors divided by all errors. |
| `residual_error_per_task` | Errors left outside review divided by all incoming tasks. Multiply by 100 for manuscript tables. |
| `unreviewed_error_rate` | Residual errors divided by targets left outside review. |
| `review_yield` | Expected errors in the queue divided by review slots. |
| `auroc` | Error discrimination, with standard half-credit ties. |
| `aurc` | Average selective error over nonempty acceptance prefixes, using expected tie allocation. Lower is preferable. |

Primary JSON compares DSDE4 minus public-example-failure-first/NLL, then takes
an equal-weight mean across six conditions. Negative differences favor DSDE.
The NumPy array retains all 4,000 paired task-bootstrap draws with the same
resampled task indices across conditions. Retrospective intervals and the
public-passing conditional diagnostic are descriptive 2,000-draw comparisons;
the latter does not replace the primary comparison.

## Costs and operational sensitivity

All timing fields are seconds. `greedy_generation_seconds` is target request
acquisition; `generation_seconds_k` includes the target and first k alternatives.
In the new table, `execution_seconds_k` includes their public probe executions;
`greedy_execution_seconds` includes only target public checks.
`full_generation_seconds` and `full_execution_seconds` use all candidates.
`private_evaluation_seconds` is offline correctness-label acquisition, excluded
from screening cost.

Cost tables contain per-condition means, medians, 95th percentiles and ratios
of means; `alternatives` is the sampled alternative count. Raw completed-request
elapsed time includes a documented machine suspension and excludes failed
requests, warm-up, model loading and restart downtime. It is not end-to-end
operating cost. The separately named available-machine sensitivity subtracts
only the documented union of sleep overlaps, retains every candidate and label,
and does not replace the raw timing analysis.

## Calibration files

`coordinate` is `conditioned`, `pooled`, or `independent`. Conditioned uses
precision-specific empirical-reference CDFs and a shared threshold; pooled uses
one joint reference CDF and a shared threshold. Independent uses a separate CDF
and separately selected threshold for each precision. `selector` identifies Bonferroni fixed-family calibration or the exploratory
count-screened sequential heuristic. `alpha` is the selective-error budget;
`salt` and `repetition` identify one of 30 fixed task partitions.
`reference_n`, `calibration_n`, `test_n` are disjoint task counts.
`selected` indicates whether a feasible threshold exists; `threshold` is
empty for infeasible policies. `accepted`, `errors`, `coverage`, `risk` and
`test_target_met` describe the held-out accepted subset.

Infeasible policies have zero acceptance/coverage and undefined (blank) risk;
they are not treated as zero-error successful policies. Summary tables retain
them in unconditional coverage. Repeated partitions are sensitivity analyses,
not independent task replications. The conditional risk statement applies to
each prespecified fixed score family and independent reference map under the
stated sampling assumptions; it does not justify choosing a score after seeing
the results or asserting safety on a shifted task stream.

## Formats, missingness and provenance

CSV uses UTF-8 with a header, decimal fractions and blank undefined values.
JSON/JSONL uses UTF-8 with JSON null for undefined quantities. `.npy` stores the
bootstrap array and can be read with NumPy without executing generated code.
`artifact_manifest.json` inventories bytes and SHA-256 for every packaged file.
Original machine paths in locked provenance are historical context; the
portable `replay_analysis.py` resolves its inputs relative to the archive.
README commands, source scripts and protocol amendments define exact rules.

## Decision reporting added in statement-aware-v3

`same_stage_capacity_metrics.csv` compares DSDE4, Exec/NLL, Public/NLL and
Public/DSDE4 on the same 175 tasks at 10%, 25% and 50% review. It records the
number of known execution failures, public failures, positions and errors
captured. The 25% comparison retains its original primary role.

`public_pass_decision_counts_and_costs.csv` reports public-passing subset sizes,
actual positions, hidden errors and expected captured counts for DSDE4 and NLL.
`mean_extra_sampling_seconds` sums four alternative generations and their public
executions after target generation/checks, averaged only over that passing subset.
It excludes host-side matching, score construction and sorting.

`guarded_capacity_curve.csv` describes every queue size from 1 to 175. Its
`additional_errors_captured` is Public/DSDE4 minus Public/NLL. The shaded region
in the plot corresponds to `review_slots <= public_failures`. These retained-
record additions are descriptive, recorded in Amendment 07.

`functional_oracle_inventory.json` lists all 63 functional tasks, expected JSON
output types and the task-specific finite numerical tolerance. It includes no
private test inputs or expected values. `functional_correction_receipt.json`
records comparator-only rejudging of the six Separate Squares I targets, with
no changes to their labels or public-failure statuses.
