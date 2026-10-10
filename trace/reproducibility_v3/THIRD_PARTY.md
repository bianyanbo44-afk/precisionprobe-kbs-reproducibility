# Upstream inputs and attribution

This artifact does not redistribute model weights, inference binaries or
benchmark private tests. Retrieval identifiers are:

- LiveCodeBench lite: https://huggingface.co/datasets/livecodebench/code_generation_lite
  revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, file `test6.jsonl`,
  SHA-256 `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`.
  This is the 175-task increment, not cumulative release_v6.
- EvalPlus 0.3.1: https://github.com/evalplus/evalplus
- Qwen2.5-Coder: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
- DeepSeek-Coder: https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF
- Qwen3: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
- llama.cpp build b9637: https://github.com/ggml-org/llama.cpp

The configuration identifies the six model files and inference executable by
SHA-256. Model license and dataset terms should be checked at the upstream
locations before a new generation or redistribution. Public problem metadata
identify the evaluated tasks; no assertion of training-data absence is made.

The semantic-distance approach and DSDE are attributed to He, Sharma and David
(2026), https://doi.org/10.48550/arXiv.2605.09023. The retained implementation
has documented type-aware output costs and a 0.15 matching-exception penalty.
It is not an exact reproduction of their published score.

The experiment-design checklist used during preparation is attributed to
Kassis et al. (2026), https://doi.org/10.48550/arXiv.2609.00065. This is procedural
assistance, not evidence for the findings. OpenAI Codex assisted in scripting,
analysis, writing and plotting code as disclosed in the manuscript.
