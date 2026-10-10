# Amendment 07, functional tolerance and decision reporting

Recorded 2026-10-10 Asia/Shanghai after the independent pre-submission assessment
of statement-aware-v2 and the author's instruction to implement its corrections.
This amendment is retrospective. It does not change the original local protocol
lock, generation requests, task membership, sampling budget, primary endpoint,
bootstrap seeds, reference/calibration/test partitions or acquisition timings.

The Separate Squares I task, `leetcode/3763`, explicitly accepts answers within
10^-5. The current functional literal-equality branch omitted this convention.
The revised comparator uses an absolute tolerance of 10^-5 for a single finite
numeric return on this task; Boolean, missing, nonnumeric and nonfinite returns
are rejected. Other functional tasks keep their documented literal/JSON rules.
The precise v2 comparator is preserved as `statement_comparators_v2.py` so the
Amendment 06 receipt remains bound to the implementation that produced it.

All 63 functional tasks in the pinned increment are inventoried by task ID,
function and expected output types. The retained statements are screened for
explicit tolerance, ordering and alternative-output clauses. Separate Squares I
is the only functional task with an explicit numeric tolerance in this increment.
The inventory reports its scope and does not claim equivalence to the official
LiveCodeBench evaluator or establish the semantics of every private test.

The six retained greedy targets for the affected task are rejudged using their
existing public execution observations and pinned public references. If any
becomes publicly passing, analysis stops until its required private tests have
been evaluated in the original sandbox. An unchanged public-failing target
requires no new private execution. The receipt preserves input hashes, before
and after judgements, final labels and whether downstream records changed.
No new candidate or favorable task is selected, and score observations remain
separate from correctness comparison.

The main text adds the already retained same-stage Exec/NLL and Public/NLL
comparisons at 10%, 25% and 50%, keeping 25% as the prospective primary endpoint.
A decision table connects known failures, remaining capacity, incremental cost
and independent calibration evidence. Public-passing absolute counts and costs,
and an all-capacity curve, are explicitly descriptive additions based on retained
records. They do not replace the unfavorable primary result or become prospective
evidence after inspection.

Source for the functional rule:
https://leetcode.com/problems/separate-squares-i/description/
