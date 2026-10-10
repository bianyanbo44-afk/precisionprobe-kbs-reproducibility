# Anonymous analysis artifact

This archive supports *TRACE: When high discrimination fails to improve quantized
code screening*. Its evaluation version is `statement-aware-v3`. It is a research artifact,
not a submission receipt or evidence of journal acceptance.

## Replay without inference

Use Python 3.12 and install `requirements-analysis.txt`. From this directory:

```sh
python -B eswa_upgrade_20261009/scripts/replay_analysis.py --verify-only
python -B eswa_upgrade_20261009/scripts/replay_analysis.py
```

The first command checks the artifact manifest, all retrospective input hashes,
candidate/task completeness, stored solution hashes, same-request confidence
joins, reconstructed execution scores, screening costs, functional labels and
derived feature tables. It does not execute generated code.

The second command reconstructs the statistical comparisons from those records
using the original task-bootstrap seeds and calibration partitions. It writes
to `eswa_upgrade_20261009/replay_results/`, leaving the reference analyses intact.
It verifies the prospective primary result and all 4000 joint bootstrap draws,
and compares ranking and policy tables to the retained reference output.
Recomputing all retrospective bootstrap and calibration tables may take several
minutes. `--stage retrospective` or `--stage validation` selects one stage.

To regenerate manuscript figures and tables from the retained reference data:

```sh
python -B eswa_upgrade_20261009/scripts/build_paper_assets.py --require-validation
```

## Contents and scope

- `runs/`: archived observation, confidence and label inputs identified by the
  original input manifest, plus retained target solutions and label provenance.
- `eswa_upgrade_20261009/runs/`: all new generated responses, execution records
  and their completion receipts.
- `eswa_upgrade_20261009/analysis/`: reference results, including unfavorable
  comparisons, every score and risk budget, and empty acceptance policies.
- `eswa_upgrade_20261009/protocol/`: the local specification, unchanged lock,
  and dated analysis amendments. The protocol was not externally registered.
- `eswa_upgrade_20261009/data/livecodebench/task_metadata.csv`: all 175 task
  identifiers and public metadata. It contains no private test payloads.
- `src/precisionprobe/`: exact archived score implementation and its imports.
- `eswa_upgrade_20261009/scripts/`: analysis and plotting entry points, and
  original generation/evaluation code for method inspection.

Model weights, runtime binaries, original benchmark questions and private test
payloads are not included. Their official locations and hashes are documented
in `THIRD_PARTY.md` and the locked configuration. Stored observations reproduce
the analysis; they cannot establish that model inference or program execution
will repeat bit for bit on different hardware.

The original generation scripts retain the original machine paths and lock.
They are provenance, not a portable generation command. A new inference run
requires upstream inputs, a compatible sandbox/runtime, locally configured
paths and a separately recorded replication configuration. Do not silently
rewrite the original lock. Statistical replay does not require those inputs.

The sequential calibration selector is exploratory because of its
calibration-dependent count screening (Amendment 01). The public-passing
diagnostic is secondary (Amendment 02). Neither replaces the all-task primary
comparison. Repeated partitions do not constitute independent new tasks.

Amendment 06 corrects eight task-stated output conventions after review. The
48 affected target assessments preserve every final error label, while one
public-failure status changes. Original exact-match evaluations remain under
`evaluations_v1_exact_match/`. The correction receipt identifies every assessment,
executed private-test counts and fresh evaluation time, excluded from acquisition
cost. Generation, disagreement inputs, token scores and the original lock are
unchanged. The new analysis and its public-passing diagnostic use the amended
correctness records. Amendment 07 adds the functional tolerance and retains all six target labels.
Its receipt and the exact v2 comparator preserve the correction chain. The
public-passing counts/costs and guarded capacity curves provide descriptive
decision evidence. This archive is the authoritative statement-aware-v3 dataset;
its deposit record is supplied separately from the anonymous review files.

## License

The project code retains its MIT license. Derived tables and figures use
Creative Commons Attribution 4.0, following the existing project release.
Third-party benchmarks, models and their outputs retain applicable upstream
terms; this archive does not relicense those materials. Cite the accompanying
article and the final versioned archive when using the results.
