# Exploratory Strengthening Protocol: Split-Sample PCQRC

Protocol version: 1.0  
Amendment date: 2026-08-26 (Asia/Shanghai)  
Status: separately locked methodological strengthening; it does not modify or
replace `locked_protocol.md`.

## Disclosure boundary

This amendment was written after the Qwen/HumanEval+ generation and official
labels had been produced, but before its PCQRC or baseline results were run by
the extension analysis. DeepSeek/MBPP+ generation was still in progress and no
official DeepSeek/MBPP+ labels existed. Consequently, all analyses in this file
are exploratory for Qwen/HumanEval+ and prospectively specified with respect to
the still-unlabeled DeepSeek/MBPP+ panel. The original locked confirmatory
analysis remains authoritative and will be reported regardless of direction.

## Motivation

The original PCQRC protocol estimates a precision-specific empirical CDF and
checks risk on the same outer calibration tasks. Although the CDF is
label-independent, its raw-score cutoffs are data-dependent. To support a clean
finite-grid confidence statement conditional on the estimated score map, the
strengthened analysis separates the observations used to estimate each CDF
from the observations used to certify selective risk.

## Split-sample PCQRC

The outer calibration/test split, primary alpha=0.30, delta=0.10, 41-point
percentile grid, and minimum accepted count of 20 remain exactly as in the
original protocol. Within the outer calibration half, task identifiers are
sorted by SHA256("split-pcqrc-inner-v1|" + outer_salt + "|" + task_id). The
first ceiling(n/2) tasks form a score-reference set and the remainder form an
independent risk-calibration set.

For each precision, the score-reference empirical CDF maps risk-calibration
scores to percentiles. A common percentile threshold is selected using only
risk-calibration labels and the same simultaneous Clopper-Pearson/Bonferroni
rule over 41 thresholds and two precisions. The reference map and selected
threshold are then applied once to the untouched outer test set. Infeasibility
is retained as a result.

Conditional on the score-reference set, all candidate acceptance rules are
fixed before risk-calibration labels are inspected. Under IID sampling, a union
bound over the fixed precision-threshold family therefore gives the selected
rule the stated simultaneous finite-grid confidence interpretation. This is
not a distribution-shift or unseen-deployment guarantee.

## Required comparisons

Using the same independent risk-calibration subset, report:

1. split-sample PCQRC;
2. a shared raw-DSDE threshold on the fixed 0.00, 0.025, ..., 1.00 grid;
3. separate precision-conditioned percentile thresholds, with delta divided
   across the two precision-specific policies;
4. the original locked single-calibration PCQRC result.

For the first three policies report calibration feasibility and outer-test
accepted count, coverage, error count, and empirical selective risk. Repeat the
outer and inner deterministic procedure for 100 fixed salts. Comparisons are
descriptive; no post-hoc superiority hypothesis will be introduced.

## Reporting rule

The manuscript must distinguish the original confirmatory endpoint from this
strengthening analysis, disclose this amendment's timing, and report every
method/panel outcome. A positive result may be claimed only for panels and
operating points that actually satisfy their stated criteria.
