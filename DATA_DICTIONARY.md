# Data dictionary and evidence map

## Evidence map

| Evidence | Public source | Analysis entry point |
| --- | --- | --- |
| Crossed-panel discrimination and baselines | `runs/*/joined.csv`, `runs/*/token_confidence.jsonl` | `scripts/analyze_pcqrc_extension.py` |
| Frozen policies and held-out screening | `phase4_extension/results/extension_analysis.json` | `scripts/analyze_pcqrc_extension.py` |
| Repeated-split robustness | `phase4_extension/results/pcqrc_sensitivity_robustness.json` | `scripts/analyze_pcqrc_sensitivity_robustness.py` |
| Qwen3 confirmation | `runs/rapid_qwen3_confirm60/analysis/analysis.json` | `scripts/analyze_rapid_qwen3.py` |
| KBS figures 1-5 and compact tables | `figures/`, `results/*summary.csv` | `scripts/build_kbs_assets.py` |
| Protocol definitions | `phase4_extension/*protocol*.md`, `phase4_extension/*lock.json` | `scripts/verify_public_release.py` |

## Record conventions

- `task_id`: upstream benchmark task identifier; join records by task ID and precision.
- `precision`: `q2`, `q4`, or `q8`; exact GGUF format and model hash are in the run manifest.
- `generations/q*.jsonl`: generated candidate source and generation metadata. Treat generated code as untrusted data; reading it does not require executing it.
- `scores.jsonl`: task-level execution-semantic and structural uncertainty scores.
- `joined.csv`: wide task table; the precision prefix identifies the served artifact. `*_error` is the binary EvalPlus error label, with 1 meaning functional error.
- `evalplus_results.json`: evaluation outcomes; benchmark test suites themselves are obtained upstream.
- `sha256` and `*_sha256`: SHA-256 identifiers of the named artifacts or content.
- `elapsed_seconds`: wall-clock duration in seconds, where recorded.

## Token-confidence summaries

One row represents one task at one precision. Every row is retained, including
non-matched Qwen3 records; analyses use the recorded match status.

| Column | Meaning |
| --- | --- |
| `token_count` | Number of scored completion tokens |
| `mean_nll`, `median_nll` | Mean/median negative token log-probability, in nats |
| `mean_top1_margin` | Mean gap between the largest and second-largest returned token log-probabilities |
| `mean_topk_entropy` | Mean entropy of returned top-k probabilities plus one residual-probability bucket, in nats |
| `worst_decile_nll` | Negative 10th percentile of chosen-token log-probabilities (90th percentile of token NLL), in nats |
| `status` | Whether replayed solution content matches the expected generated solution |
| `expected_solution_sha256`, `generated_solution_sha256` | Expected and replayed solution-content hashes |
| `model_sha256` | Hash of the served model artifact |

Empty CSV entries and JSON `null` indicate unavailable values, not zero. A policy
with `no_feasible_threshold` must not be interpreted as a selected threshold.
Coverage, error rate, and risk are fractions in [0, 1] unless a figure axis states
percent. `ci_low`/`ci_high` are bootstrap interval endpoints as defined in the
corresponding analysis. AURC is area under the risk-coverage curve; AUPRC is area
under the precision-recall curve. The Qwen3 summary's final column is `auprc`
(the initial repository used the incorrect header `aurc`; values are unchanged).

## Sources and reuse

Task identifiers refer to [HumanEval](https://github.com/openai/human-eval),
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp), and
their [EvalPlus](https://github.com/evalplus/evalplus) evaluation variants.
Upstream dataset and evaluation-code licenses continue to apply. Models are
identified by Hugging Face repository, filename, revision where recorded, byte
count and digest in `runs/*model_manifest.json`; weights are not redistributed.
The repository's MIT/CC BY license grant does not replace third-party rights in
upstream benchmark or model content.
