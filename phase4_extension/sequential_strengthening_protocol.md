# Method-Development Protocol: Fixed-Sequence Split PCQRC

Protocol version: 1.0  
Development date: 2026-08-26 (Asia/Shanghai)  
Status: separately locked method-development analysis; it does not modify the
confirmatory protocol or the first exploratory strengthening protocol.

## Disclosure boundary

This protocol was written after the independently split Bonferroni procedure
was implemented and checked on the two development panels. The
Qwen/HumanEval+ generations and labels already existed, although no extension
analysis had been run on that panel. DeepSeek/MBPP+ generation was in progress
and its official labels did not exist. Results under this protocol are therefore
method-development evidence for the development panels, exploratory evidence
for Qwen/HumanEval+, and prospectively specified secondary evidence for
DeepSeek/MBPP+. They cannot replace the original confirmatory endpoint.

## Fixed-sequence split PCQRC

Use the same outer calibration/test split and inner score-reference/risk-
calibration split specified in `exploratory_strengthening_protocol.md`. The
score-reference set defines an empirical CDF for each precision. Conditional
on those reference sets, the 41 common percentile rules are fixed before the
independent risk-calibration labels are examined.

The eligible percentile thresholds are ordered from 0.00 to 1.00. Thresholds
that accept fewer than 20 risk-calibration tasks at either precision are not
eligible. Starting with the smallest eligible threshold, test whether the
selective risk is at most alpha=0.30 at both precisions. At a threshold, compute
a one-sided Clopper-Pearson upper bound at delta=0.10 for each precision and
pass the intersection-union test only when both upper bounds are at most alpha.
Continue only after a pass and stop at the first failure. Select the largest
consecutively passing threshold; return infeasible if the first eligible
threshold fails or no threshold is eligible.

No division by the number of thresholds is used. Under IID sampling and
conditional on the score-reference set, any unsafe selected threshold requires
rejection of the first true null in the fixed sequence. That test has size at
most delta. Requiring both precision-specific nulls to be rejected is an
intersection-union test for the composite null that at least one precision is
unsafe, so no across-precision division is required at a single common
threshold. The guarantee is restricted to the fixed sequence, fixed precision
family, and stated sampling assumptions; it is not valid under arbitrary
distribution shift.

## Comparisons and robustness

Using exactly the same inner risk-calibration and outer test tasks, report:

1. fixed-sequence split PCQRC on the precision-conditioned percentile scale;
2. the same fixed-sequence procedure on the raw DSDE scale;
3. the Bonferroni split-PCQRC result;
4. the original locked single-calibration PCQRC result.

Repeat the fixed-sequence percentile and raw procedures over 100 fixed outer
and inner split salts. Report selection, target-qualification, contradiction,
and held-out coverage rates. All panels and all infeasible outcomes remain in
the record. Method comparisons are descriptive and do not create a post-hoc
superiority test.
