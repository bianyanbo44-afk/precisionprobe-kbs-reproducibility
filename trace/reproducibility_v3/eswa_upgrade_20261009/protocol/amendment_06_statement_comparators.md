# Amendment 06: task-stated output rules

Date: 2026-10-09. Added after the completed panel and manuscript review.

The original locked evaluator compared standard output by stripped lines and
exact decimal equality. Review identified a conflict with the numerical error
tolerance in `atcoder/abc392_d`. Inspection of the retained task statements also
identified three case-insensitive answers and four tasks permitting nonunique
valid solutions. This amendment corrects those eight output rules in every
condition; it is not part of the initial pre-generation specification.

## Rules

| Task | Statement-aware comparison |
|---|---|
| `atcoder/abc392_d` | One finite probability; absolute or relative error at most `1e-8` |
| `atcoder/abc395_a` | Case-insensitive Yes/No |
| `atcoder/arc192_a` | Case-insensitive Yes/No |
| `atcoder/arc192_b` | Case-insensitive Fennec/Snuke |
| `atcoder/abc396_e` | Nonnegative XOR-constrained assignment with minimum sum; tied optima accepted |
| `atcoder/abc397_d` | Positive integers satisfying the cubic difference; `-1` follows the supplied no-solution reference |
| `atcoder/arc190_a` | Valid interval operations covering every point with the minimum operation count |
| `atcoder/arc191_c` | A bounded integer pair with the required exact multiplicative order, checked by gcd and modular powers |

The pure comparator module records statement locations, bounds and algorithms.
Every other task and functional-interface comparison retains the original rule.
Comparator tests include valid alternative answers, boundary cases and rejection
of malformed or invalid witnesses. No new model responses are generated.

## Complete and separate reevaluation

All 48 affected condition/task pairs are reevaluated using their retained seed-0
solutions and the original Linux bubblewrap sandbox and four-second execution
limit. Every public case is executed. Targets passing the public cases continue
through the private cases until the first failure or completion. A corrected
passing label requires execution of every private case; a tolerated public
rounding difference alone is insufficient.

The original six evaluation files and completion receipts are preserved under
`runs/livecodebench_v6_increment/evaluations_v1_exact_match/`. The unchanged
protocol lock, generation files, target hashes, token-confidence measurements,
public execution observations used to construct disagreement scores, score
values and original acquisition times remain the initial evidence. Only
correctness fields and public-failure-guarded ranking scores are amended.
Fresh correctness-execution times are recorded separately and are excluded from
the measured acquisition frontier. The machine-readable
`statement_correction_receipt.json` identifies inputs, scripts, all before/after
assessments, completed private tests and preservation checks. It contains no
private input or expected-output payloads.

All label-dependent validation analyses are then recomputed: the joint-task
bootstrap, all ranking and subgroup tables, public-passing diagnostics, cost
frontiers, operating-time sensitivity, and disjoint calibration policies. The
retrospective EvalPlus panel is unaffected. Manuscript and artifact versions
identify this amended evaluation explicitly; the initial public release remains
a separate earlier version.
