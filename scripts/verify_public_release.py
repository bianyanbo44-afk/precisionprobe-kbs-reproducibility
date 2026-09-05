"""Verify the public byte-level inventory, JSON records and protocol locks.

This check reads data only; it never executes bundled model-generated code.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside release: {name}")
    if not path.is_file():
        raise FileNotFoundError(name)
    return path


def verify(root=ROOT):
    manifest = json.loads((root / "results/source_manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["source_hashes"]
    if not hashes:
        raise ValueError("Empty inventory")
    jsonl_rows = 0
    for name, expected in hashes.items():
        path = contained_file(root, name)
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
                    jsonl_rows += 1

    for lock in (root / "phase4_extension").glob("*lock.json"):
        record = json.loads(lock.read_text(encoding="utf-8"))
        protocol = contained_file(root, record["protocol"])
        if sha256(protocol) != record["sha256"] or protocol.stat().st_size != record["bytes"]:
            raise ValueError(f"Protocol lock mismatch: {lock.name}")

    for path in (root / "runs").glob("*/token_confidence_manifest.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        raw = path.parent / "token_confidence.jsonl"
        if sha256(raw) != record["output_sha256"]:
            raise ValueError(f"Confidence record mismatch: {path.parent.name}")

    transforms = json.loads((root / "results/publication_transforms.json").read_text(encoding="utf-8"))
    for name, record in transforms["files"].items():
        if sha256(contained_file(root, name)) != record["public_sha256"]:
            raise ValueError(f"Public transformation mismatch: {name}")
    return {"status": "PASS", "hashed_files": len(hashes), "jsonl_rows": jsonl_rows}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
