import hashlib
import json

import pytest

from scripts.verify_public_release import contained_file, verify


def test_release_inventory_and_frozen_locks():
    assert verify()["status"] == "PASS"


def test_checksum_rejects_changed_record(tmp_path):
    (tmp_path / "results").mkdir()
    record = tmp_path / "record.json"
    record.write_text('{}', encoding="utf-8")
    manifest = {"source_hashes": {"record.json": hashlib.sha256(record.read_bytes()).hexdigest()}}
    (tmp_path / "results/source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path)


def test_inventory_cannot_escape_repository(tmp_path):
    with pytest.raises(ValueError, match="outside release"):
        contained_file(tmp_path, "../outside.json")
