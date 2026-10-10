# Amendment 01: scope of the calibration claims

Recorded on 2026-10-09, before the first prospective validation generation.
The original version 1.0 protocol and lock are retained unchanged.

The implementation called `fixed_sequence` skips thresholds with fewer than
20 accepted calibration observations before applying its stopping rule. This
calibration-dependent eligibility step does not inherit the usual fixed-order
family-wise error guarantee without an additional argument. We have not
established such an argument.

Keep its specified computations and all outcomes for transparency, but label
them as a count-screened sequential heuristic. They are exploratory sensitivity
results and must not support a finite-sample risk-control claim. The independently
fixed reference maps and the Bonferroni-adjusted finite threshold family are the
only calibration procedure for which the manuscript states a formal guarantee,
subject to the explicitly stated sampling assumptions. The guarantee applies to
each specified score family separately, not to choosing a favorable score after
looking at these results. Repeated splits are descriptive, not independent
replications or a simultaneous guarantee over split choices.

This amendment changes the interpretation and reporting, not generated data,
score definitions, the primary endpoint, the bootstrap, or policy selection.
No prospective outcomes were available when this decision was made.
