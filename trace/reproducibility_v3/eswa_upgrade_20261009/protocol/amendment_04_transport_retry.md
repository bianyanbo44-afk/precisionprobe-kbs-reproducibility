# Amendment 04: GPU service interruption and bounded transport retry

Recorded on 2026-10-09 after Qwen3 Q4 evaluation, before inspecting prospective
correctness comparisons or selecting any reported findings. The model files,
runtime, prompts, candidate seeds, decoding parameters, score definitions,
functional criterion and inferential endpoints remain unchanged.

The first condition completed all 875 generations and all 175 evaluations.
Qwen3 Q8 then completed the five records for its first task. At approximately
11:20 Asia/Shanghai the sixth request lost its local connection and the
llama.cpp server exited. Windows Application events identify a llama-server
failure and a LiveKernelEvent 117 associated with the NVIDIA display driver.
The server uses the Vulkan backend. This is a service/driver failure with no
completed response, rather than an incorrect generated program. No change to
GPU drivers, system watchdogs, power settings or decoding settings is made.

The five intact Qwen3 Q8 records have unique expected keys and valid solution
hashes. They are retained exactly. The original records, stderr, stdout, server
log and previous persistence wrapper are backed up in
`runs/livecodebench_v6_increment/recovery/20261009T032026Z_service_failure`.
The missing request resumes with the same task and seed; an identical
unpreserved response is not claimed. All first-condition generation and
evaluation files remain intact.

The persistence wrapper now allows at most four generation attempts within
one invocation after connection resets or request timeouts. Each restart uses
the original generator, which skips every committed task/seed record and
restarts the server before continuing. Other exceptions stop the run. Previous
server logs are preserved, and later failed requests record their condition,
seed, prompt hash, client elapsed time and error in
`logs/generation_request_failures.jsonl`. They do not receive correctness labels
or synthetic candidate outputs. Retrying depends on transport failure, not on
the contents or correctness of a completed program.

The earlier failed sixth request has no retained exact elapsed-time field;
its event time and server/client logs are preserved. Per-candidate acquisition
costs concern completed requests and public program executions, excluding
warm-ups, model loading, restart downtime and unsuccessful service requests.
They should not be interpreted as the total operating cost or uninterrupted
latency of this run. Operational interruptions and measurable later failed
request costs are reported separately. The original first-condition CPU
evaluation is retained; there is no evidence of a sandbox infrastructure
failure and no selection of reruns using correctness outcomes.
