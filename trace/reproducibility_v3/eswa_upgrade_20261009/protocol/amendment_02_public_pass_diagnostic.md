# Amendment 02: specify the public-passing diagnostic

Recorded on 2026-10-09 while candidate generation was running, before any new
validation programs had been evaluated. No validation correctness labels or
screening outcomes had been inspected. Preserve protocol version 1.0 and its
hash lock unchanged.

The fixed primary comparison remains DSDE4 versus public-example-failure-first
NLL at 25% review, using all 175 tasks in each of the six conditions.

Specify an additional secondary diagnostic of the protocol's planned
"incremental value over public-example checks": restrict each condition to
targets that pass all public examples, then compare DSDE4 and NLL at 25% review
within that subset. Report every condition, the number of public-passing tasks,
its hidden-test error count, and the paired residual-error difference with a
2000-replicate task-bootstrap interval (seed 20261009). Also report point metrics
for all fixed score families on this subset. These conditional subsets can differ
across model/precision conditions, so no confirmatory cross-condition macro claim
is made for them. Their intervals are descriptive and not multiplicity adjusted.

Reason: public-example failures necessarily count as functional errors under
the evaluation criterion. When their count is at least the review capacity,
any guard that places them first already achieves the minimum possible residual
error at that capacity, because every review slot contains a known error. Pure
DSDE cannot improve that lower bound. The primary comparison remains a valid
application comparison, while the conditional diagnostic asks whether execution
sampling provides information where public checks have not settled correctness.

The interpretation of this additional analysis is secondary regardless of its
outcome. It does not alter generation, tests, score computation, primary sample,
primary endpoint, or the frozen bootstrap pairing across six conditions.
