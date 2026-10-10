# TRACE: reproducibility data, analysis and figures

This repository supports **TRACE: When high discrimination fails to improve
quantized code screening**. TRACE connects target checks, remaining review
capacity, the incremental value of sampled execution, and acceptance calibration.

## Current versions

| Component | Version | Location |
| --- | --- | --- |
| Generated programs, observations, labels and statistical analysis | statement-aware-v3 | [Versioned data release](https://github.com/bianyanbo44-afk/precisionprobe-kbs-reproducibility/releases/tag/eswa-trace-v3-20261010); `trace/reproducibility_v3/` |
| Four manuscript figures and drawing code | presentation-v6 | [Figure release](https://github.com/bianyanbo44-afk/precisionprobe-kbs-reproducibility/releases/tag/eswa-trace-figures-v6-20261010); `trace/figures_v6/` |

The data release is the archive cited by the manuscript. Its ZIP is identical to
the anonymous Supplementary Data supplied for review:

`SHA-256: 392861dbfe7767e768f9202c688020906607cdfd66f27c799ed764a542ef596a`

The v6 update improves figure typography, panel layout and the cost comparison
display. Experimental records, numerical results and bootstrap intervals remain
the same. The data release and earlier versions remain available.

## Reproduce the current analysis

Use Python 3.12. From the repository root:

```sh
cd trace/reproducibility_v3
python -m pip install -r requirements-analysis.txt
python -B eswa_upgrade_20261009/scripts/replay_analysis.py --verify-only
python -B eswa_upgrade_20261009/scripts/replay_analysis.py
```

The verification command checks retained inputs, task/candidate completeness,
confidence joins, execution scores, costs and evaluation labels. The replay
reconstructs the statistical comparisons, including all 4,000 joint primary
bootstrap draws. Both use retained observations. See the artifact's README,
DATA_DICTIONARY.md and THIRD_PARTY.md for file definitions and upstream inputs.

## Reproduce the current figures

From the repository root, with Matplotlib, NumPy and pandas installed:

```sh
python trace/figures_v6/figure_source/build_visual_figures.py --out-dir reproduced/trace_figures_v6
```

Each figure has editable PDF/SVG and PNG/TIFF exports. Figure 2 shows excess
residual errors over Public/NLL, with unchanged absolute errors in the data
tables. Figure 3 retains every integer queue size; Figure S1 retains all ten
original estimates and intervals. See [the figure README](trace/figures_v6/README.md).

## Citation and license

For the empirical evidence, cite: Bian, Y., & Wang, Z. (2026). *TRACE screening
data and decision evidence* (Version statement-aware-v3) [Data set and analysis
code]. [GitHub release](https://github.com/bianyanbo44-afk/precisionprobe-kbs-reproducibility/releases/tag/eswa-trace-v3-20261010).

Project code uses MIT; derived tables and figures use CC BY 4.0. Models,
benchmarks and their outputs retain applicable upstream terms, documented in
the data artifact. Model weights, runtime binaries and private-test payloads
are obtained separately from their upstream sources.

The earlier PrecisionProbe/PCQRC files remain in the original root directories.
Their documentation is [preserved here](docs/legacy_precisionprobe.md).
