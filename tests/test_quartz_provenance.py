import copy
from pathlib import Path

import scripts.audit_quartz_run as audit
from scripts.audit_quartz_run import artifact_record, assess_stability, verify_artifact
from scripts.run_quartz_cell import prepare_manifest


def manifest_context(code_hash: str) -> dict:
    return {
        "study": "quartz_test",
        "config": "configs/test.yaml",
        "config_sha256": "config-hash",
        "model_manifest": "runs/model_manifest.json",
        "models": {"q4": {"sha256": "model-q4"}, "q8": {"sha256": "model-q8"}},
        "task_ids": ["task-1", "task-2"],
        "llama_server_sha256": "server-hash",
        "python": {"version": "test"},
        "code_sha256": {"run_quartz_cell.py": code_hash},
    }


def test_score_only_manifest_keeps_generation_phase_and_created_at():
    existing = manifest_context("generation-code")
    existing.update(
        {
            "manifest_schema_version": 2,
            "created_at": "2026-01-01T00:00:00+00:00",
            "phases": {
                "generation": [{"recorded_at": "then", "outputs": {"q4": {"sha256": "g"}}}],
                "scoring": [],
                "evaluation": [],
            },
            "invocations": [],
        }
    )
    untouched = copy.deepcopy(existing)

    resumed, invocation_id = prepare_manifest(
        existing,
        manifest_context("scoring-code"),
        mode="score_only",
        started_at="2026-01-02T00:00:00+00:00",
    )

    assert existing == untouched
    assert resumed["created_at"] == existing["created_at"]
    assert resumed["phases"]["generation"] == existing["phases"]["generation"]
    assert resumed["code_sha256"]["run_quartz_cell.py"] == "scoring-code"
    assert resumed["invocations"][-1] == {
        "id": invocation_id,
        "mode": "score_only",
        "started_at": "2026-01-02T00:00:00+00:00",
        "status": "RUNNING",
        "config_sha256": "config-hash",
        "code_sha256": {"run_quartz_cell.py": "scoring-code"},
    }


def test_legacy_manifest_is_snapshotted_before_score_only_update():
    existing = manifest_context("generation-code")
    existing["created_at"] = "2026-01-01T00:00:00+00:00"
    resumed, _ = prepare_manifest(
        existing,
        manifest_context("scoring-code"),
        mode="score_only",
        started_at="2026-01-02T00:00:00+00:00",
    )

    snapshot = resumed["manifest_history"][0]["snapshot"]
    assert snapshot["code_sha256"]["run_quartz_cell.py"] == "generation-code"
    assert resumed["code_sha256"]["run_quartz_cell.py"] == "scoring-code"


def complete_stability(expected_tasks: int) -> dict:
    return {
        "status": "PASS",
        "expected_tasks": expected_tasks,
        "left_tasks": expected_tasks,
        "right_tasks": expected_tasks,
        "left_complete": True,
        "right_complete": True,
        "missing_from_left": [],
        "missing_from_right": [],
        "semantic_mismatch_count": 0,
        "left_semantic_sha256": "same-hash",
        "right_semantic_sha256": "same-hash",
    }


def test_eligibility_stability_requires_complete_expected_panel():
    assert assess_stability(complete_stability(2), expected_tasks=2)["status"] == "PASS"

    partial = complete_stability(1)
    partial["expected_tasks"] = 2
    result = assess_stability(partial, expected_tasks=2)
    assert result["status"] == "FAIL"
    assert "stability_left_completeness" in result["failures"]
    assert "stability_right_completeness" in result["failures"]


def test_stability_report_must_match_two_recorded_score_runs():
    manifest = {
        "phases": {
            "scoring": [
                {"status": "COMPLETED", "outputs": {"scores": {"sha256": "first"}}},
                {"status": "COMPLETED", "outputs": {"scores": {"sha256": "second"}}},
            ]
        }
    }
    stability = {"left_file_sha256": "first", "right_file_sha256": "second"}
    assert audit.assess_stability_score_history(manifest, stability)["status"] == "PASS"

    stability["left_file_sha256"] = "stale"
    assert audit.assess_stability_score_history(manifest, stability)["status"] == "FAIL"


def test_manifest_artifact_hash_detects_file_change(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text('{"task_id":"a"}\n', encoding="utf-8")
    record = artifact_record(path, tmp_path)
    assert verify_artifact(tmp_path, path, record)["valid"] is True

    path.write_text('{"task_id":"b"}\n', encoding="utf-8")
    result = verify_artifact(tmp_path, path, record)
    assert result["valid"] is False
    assert result["sha256_match"] is False


def test_manifest_audit_links_generation_scoring_and_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REQUIRED_CODE_PATHS", {"pipeline.py": Path("pipeline.py")})
    (tmp_path / "pipeline.py").write_text("# fixed code\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("precisions: [q4, q8]\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    (run_dir / "generations").mkdir(parents=True)
    q4 = run_dir / "generations" / "q4.jsonl"
    q8 = run_dir / "generations" / "q8.jsonl"
    scores = run_dir / "scores.jsonl"
    samples = run_dir / "evalplus_samples.jsonl"
    evaluation = run_dir / "evalplus_results.json"
    for path, value in (
        (q4, "q4\n"),
        (q8, "q8\n"),
        (scores, "scores\n"),
        (samples, "samples\n"),
        (evaluation, "evaluation\n"),
    ):
        path.write_text(value, encoding="utf-8")

    generation_outputs = {
        "q4": artifact_record(q4, tmp_path),
        "q8": artifact_record(q8, tmp_path),
    }
    code_hash = audit.sha256_file(tmp_path / "pipeline.py")
    config_hash = audit.sha256_file(config_path)
    manifest = {
        "config_sha256": config_hash,
        "code_sha256": {"pipeline.py": code_hash},
        "phases": {
            "generation": [
                {
                    "config_sha256": config_hash,
                    "expected_tasks": 2,
                    "outputs": generation_outputs,
                }
            ],
            "scoring": [
                {
                    "config_sha256": config_hash,
                    "expected_tasks": 2,
                    "code_sha256": {"pipeline.py": code_hash},
                    "inputs": {"generations": generation_outputs},
                    "outputs": {
                        "scores": artifact_record(scores, tmp_path),
                        "evalplus_samples": artifact_record(samples, tmp_path),
                    },
                }
            ],
            "evaluation": [
                {
                    "config_sha256": config_hash,
                    "inputs": {"generations": generation_outputs},
                    "outputs": {"evaluation": artifact_record(evaluation, tmp_path)},
                }
            ],
        },
    }
    _, failures = audit.audit_manifest_hashes(
        root=tmp_path,
        run_dir=run_dir,
        config_path=config_path,
        config={"precisions": ["q4", "q8"]},
        expected_tasks=2,
        manifest=manifest,
        evaluation_path=evaluation,
    )
    assert failures == []

    scores.write_text("changed\n", encoding="utf-8")
    _, failures = audit.audit_manifest_hashes(
        root=tmp_path,
        run_dir=run_dir,
        config_path=config_path,
        config={"precisions": ["q4", "q8"]},
        expected_tasks=2,
        manifest=manifest,
        evaluation_path=evaluation,
    )
    assert "manifest_scoring_hash:scores" in failures


def test_inference_code_is_required_provenance():
    assert "inference.py" in audit.REQUIRED_CODE_PATHS
