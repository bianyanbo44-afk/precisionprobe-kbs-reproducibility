# PrecisionProbe: reproducibility release

This repository contains the code, frozen configurations, derived result tables,
and figure files for:

> **Precision-Conditioned Risk Control for Execution-Semantic Uncertainty in Quantized Code Generation**

The release studies whether disagreement in observable program behaviour across
quantized versions of the same code model can support selective code generation.
It includes two frozen cross-model/benchmark panels and an independent Qwen3
confirmation run. The repository is intended to make the reported analyses and
figures inspectable without distributing model weights or private submission
files.

## What is included

- `src/precisionprobe/`: scoring, execution, ranking, inference, and risk-control code.
- `configs/`: frozen panel and analysis configurations.
- `runs/`: derived generations, scores, EvalPlus labels, audits, and stability records.
- `results/`: compact summary tables and source-manifest metadata.
- `figures/`: publication figures exported as PDF.
- `scripts/`: analysis and audit entry points.
- `tests/`: unit and integrity tests for the released implementation.

The release does **not** include model weights, API credentials, private author
information, manuscript files, or runtime binaries. Benchmark task definitions
and EvalPlus evaluation code remain governed by their upstream licenses. The
bundled result records are derived artifacts; rerunning model generation is not
required to inspect the reported tables and figures.

## Reproduce the released analyses

Python 3.11 or newer is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

The commands below regenerate analyses from the bundled derived tables:

```powershell
python scripts/analyze_rapid_qwen3.py --run-dir runs/rapid_qwen3_confirm60
python scripts/analyze_pcqrc_sensitivity_robustness.py
python scripts/generate_pcqrc_figures.py --root . --output figures/regenerated
```

The model-generation pipeline requires separately obtained model weights and an
isolated execution environment. See the configuration files and each run
manifest for the recorded model identifiers and hashes. The bundled runner is a
convenience guard against accidental unsafe execution, not a security sandbox
for adversarial programs; use WSL2 or Docker with resource and network
isolation for full generation/evaluation runs.

The public release is deliberately platform-neutral: run manifests and
evaluation records use repository-relative paths, so the analyses can be
checked after cloning to any local directory.

## Released evidence

The frozen validation panels contain 164 HumanEval+ tasks for Qwen2.5-Coder and
378 MBPP+ tasks for DeepSeek-Coder, each evaluated at Q4 and Q8. The primary
behavioral-disagreement AUROC values are available in
`results/validation_summary.csv`; the independent Qwen3 confirmation is in
`results/qwen3_replication_summary.csv`. Risk-control sensitivity and stability
records are provided in `results/pcqrc_sensitivity_robustness_summary.csv` and
the corresponding run directories.

All reported quantities are derived from the files in this repository. A missing
or infeasible risk-control threshold is represented explicitly in the released
records rather than replaced with an imputed value.

## License and citation

Source code is released under the MIT License. Derived tables and figures are
released under Creative Commons Attribution 4.0 International (CC BY 4.0).
See `LICENSE` and `CITATION.cff`.

When using this release, please cite the repository and the accompanying
manuscript.
