# Locked Extension Protocol: Precision-Conditioned Quantile Risk Control

Protocol version: 1.0  
Lock date: 2026-08-26 (Asia/Shanghai)  
Status at lock: neither validation panel has been generated, scored, or labeled.

## Objective

The extension tests whether precision-conditioned empirical-quantile mapping can
turn an execution-semantic uncertainty ranking into one operational acceptance
coordinate across Q4 and Q8 without assuming that raw score scales transfer.
The method is named Precision-Conditioned Quantile Risk Control (PCQRC).

## Development and validation separation

The completed Qwen2.5-Coder-1.5B-Instruct/MBPP+ and
DeepSeek-Coder-1.3B-Instruct/HumanEval+ panels are development evidence. They
motivated PCQRC and may be used for implementation checks, but they do not count
as locked validation.

The two untouched crossed panels are confirmatory validation evidence:

1. Qwen2.5-Coder-1.5B-Instruct on HumanEval+ at Q4_K_M and Q8_0.
2. DeepSeek-Coder-1.3B-Instruct on MBPP+ at Q4_K_M and Q8_0.

Every task in each EvalPlus benchmark is included. Generation uses one greedy
target and four stochastic alternatives with seeds 11, 29, 47, and 71,
temperature 0.60, top-p 0.95, and a 512-token cap. Correctness is the conjunction
of the official EvalPlus base and augmented suites and remains independent of
score construction.

## Primary method

For each model-benchmark panel, task identifiers are sorted by
SHA256("pcqrc-primary-v1|" + task_id). The first ceiling(n/2) tasks form the
calibration set and the remainder form the held-out test set. For each precision
separately, DSDE is mapped through the right-continuous empirical CDF of its
calibration scores. This label-independent mapping preserves within-precision
rank while placing both precisions on a common percentile coordinate.

One common percentile threshold is chosen from the fixed grid 0.00, 0.025, ...,
1.00. For every threshold and both precisions, a one-sided Clopper-Pearson upper
bound is computed. Bonferroni correction uses delta divided by 82 (41 thresholds
times two precisions). A candidate is feasible only if both upper bounds are at
most alpha=0.30 and each precision accepts at least 20 calibration tasks. The
selected candidate maximizes the smaller accepted count, with the larger
percentile threshold as the deterministic tie break. If no candidate is
feasible, the panel is reported as infeasible.

The selected mapping and threshold are applied once to the held-out tasks.
Primary reported quantities are accepted count, coverage, error count, and
empirical selective risk for each precision.

## Confirmatory criteria

The study-level primary promotion criterion is met when at least one untouched
validation panel selects a PCQRC threshold with at least 10% held-out coverage
and empirical risk at most 0.30 at both precisions, and no validation panel with
a selected threshold has empirical risk above 0.30 at either precision. The
strong criterion requires both validation panels to meet the coverage and risk
conditions. Both criteria and every panel outcome will be reported regardless of
direction.

The confirmatory discrimination criterion requires DSDE AUROC above chance in
all four untouched model-benchmark-precision deployments, operationalized as a
95% stratified-bootstrap lower confidence limit above 0.50 using 4,000 valid
replicates.

## Baselines and secondary analyses

The required baselines are greedy mean token negative log-likelihood, worst-
decile token negative log-likelihood, pairwise semantic-distance entropy, exact
execution disagreement, AST dispersion, and distinct-program ratio. Token
statistics are obtained by deterministic greedy replay and are valid only when
the regenerated solution hash matches the frozen greedy solution hash.

AUROC and AURC are reported for every baseline. Paired task bootstrap intervals
compare DSDE with token confidence and the strongest non-DSDE baseline. A fixed
five-view logistic fusion (DSDE, pairwise semantic entropy, AST dispersion,
distinct-program ratio, and mean token NLL) is secondary and must use strictly
out-of-fold predictions from repeated stratified five-fold cross-fitting. It
cannot replace the primary PCQRC analysis.

Robustness analyses repeat the task-hash calibration/test split with 100 fixed
seeds and report threshold-selection frequency, risk-target success frequency,
and the distribution of held-out coverage. Alpha values 0.20 and 0.40 are
sensitivity analyses; alpha=0.30 is the only primary operating point.

## Integrity rules

All missing, duplicate, hash-mismatched, nonfinite, and failed-evaluation records
are counted and disclosed. No validation task may be removed because of its
result. The protocol file is hashed before validation generation begins and is
not edited afterward. Any later method change is labeled exploratory and written
to a separate file.
