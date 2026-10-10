# Retrospective analysis specification

Date: 2026-10-09 (Asia/Shanghai).

All four Qwen2.5/DeepSeek x MBPP+/HumanEval+ panels and the 60-task Qwen3
MBPP+ panel have already been inspected. Every additional comparison on them is
retrospective, regardless of how tasks are resplit. The original locked
alpha=0.30 results remain part of the historical record.

## Questions

1. How much does execution disagreement improve the concentration of functional
   errors in a fixed review queue compared with token confidence, a single
   candidate's execution failures, and exact execution disagreement?
2. Does increasing stochastic alternatives from one to two to four justify its
   measured generation cost? Archived per-probe execution times are unavailable,
   so prefix execution wall times will not be presented as measured quantities.
3. Does precision-specific rank normalization improve a shared threshold over
   pooled normalization or independent thresholds when all policies use the
   same reference, calibration and test tasks?
4. How often can a policy meet its evidence requirements, and what happens when
   it cannot? Zero-acceptance outcomes will remain in unconditional summaries.

## Units and comparisons

The independent resampling unit is the programming task, paired across
precisions and methods. Model/benchmark cells are reported separately. Neither
repeated splits nor Q4/Q8 outputs are counted as extra independent tasks.

Review fractions are 0.10, 0.25 and 0.50; the primary descriptive review
comparison is 0.25. Equal-score ties receive the expected allocation under
uniform selection within the tied group, without using labels to break ties.
Report error capture, residual errors per incoming task, and error prevalence
among tasks outside the review queue. These are offline allocation metrics,
not the effectiveness of an actual human reviewer.

Fixed score families include original DSDE with one/two/four alternatives,
exact target disagreement, exact execution-cluster entropy, greedy execution
failure fraction, mean token negative log likelihood, top-k token entropy and
a fixed failure-first/token-confidence ordering. No baseline is dropped for
outperforming the original method.

Reference/calibration/test splits use sorted SHA-256 task hashes in proportions
1/4, 1/4, 1/2. Reference scores define the transformations before independent
calibration labels are used. Risk budgets are 0.10, 0.20, 0.30 and 0.40, delta
is 0.10, the grid is 0, 0.025, ..., 1, and minimum calibration acceptance is 20.
Compare a Bonferroni grid search and a prespecified fixed sequence that stops
at the first failed eligible test. All inference remains conditional on the
sampling assumptions, model artifact and task distribution.

## Known implementation corrections

- The archived behavioral score assigns 0.15 to the same exception type, so
  its self-comparison can be positive. Preserve the original numeric score and
  describe it as a bounded dissimilarity with an exception penalty.
- Separate reference and calibration tasks for any new confidence claim.
- The old data-adaptive CDF/calibration reuse is reported empirically and will
  not be justified by a theorem requiring an independently fixed rule family.
- Do not select a convenient ordering of tied zero scores to claim deployable
  low-risk coverage.

