import os
import subprocess
import sys
from pathlib import Path

import pytest

import precisionprobe.execution as execution
from precisionprobe.execution import (
    is_windows_commitment_limit_returncode,
    project_venv_site_packages,
    run_candidate,
    validate_candidate_source,
    worker_python_executable,
)


def test_rejects_os_import():
    safe, reason = validate_candidate_source("import os\ndef f(x): return x")
    assert not safe
    assert reason == "denied_import:os"


def test_allows_common_pure_replace_and_remove_methods():
    source = (
        "def f(text, values):\n"
        "    cleaned = text.replace('x', 'y')\n"
        "    copied = list(values)\n"
        "    if 0 in copied:\n"
        "        copied.remove(0)\n"
        "    return cleaned, copied\n"
    )
    safe, reason = validate_candidate_source(source)
    assert safe
    assert reason is None


def test_all_frozen_generation_import_roots_are_allowed():
    roots = (
        "calendar",
        "cmath",
        "collections",
        "datetime",
        "doctest",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "itertools",
        "math",
        "numpy",
        "operator",
        "queue",
        "re",
        "sympy",
        "sys",
        "typing",
    )
    for root in roots:
        safe, reason = validate_candidate_source(f"import {root}\ndef f(): return None\n")
        assert safe, (root, reason)
        assert reason is None


def test_allows_and_executes_cmath_polar():
    source = "import cmath\ndef f():\n    return cmath.polar(3 + 4j)\n"
    safe, reason = validate_candidate_source(source)
    assert safe
    assert reason is None

    result = run_candidate(source, "f", [[]])
    assert result["runner_status"] == "ok"
    assert result["observations"][0]["status"] == "ok"
    radius, phase = result["observations"][0]["value"]
    assert radius == 5.0
    assert 0.9 < phase < 1.0


def test_allows_and_executes_sys_getsizeof():
    source = "import sys\ndef f():\n    return sys.getsizeof([]) > 0\n"
    safe, reason = validate_candidate_source(source)
    assert safe
    assert reason is None

    result = run_candidate(source, "f", [[]])
    assert result["runner_status"] == "ok"
    assert result["observations"] == [{"status": "ok", "value": True}]


def test_allows_and_executes_hashlib_md5():
    source = "import hashlib\ndef f():\n    return hashlib.md5(b'abc').hexdigest()\n"
    safe, reason = validate_candidate_source(source)
    assert safe
    assert reason is None

    result = run_candidate(source, "f", [[]])
    assert result["runner_status"] == "ok"
    assert result["observations"] == [
        {"status": "ok", "value": "900150983cd24fb0d6963f7d28e17f72"}
    ]


def test_allows_and_executes_numpy_and_sympy_imports():
    source = (
        "import numpy\n"
        "import sympy\n"
        "def f():\n"
        "    total = int(numpy.array([1, 2]).sum() + sympy.Integer(3))\n"
        "    return total, numpy.__version__, sympy.__version__, numpy.__file__, sympy.__file__\n"
    )
    safe, reason = validate_candidate_source(source)
    assert safe
    assert reason is None

    result = run_candidate(source, "f", [[]])
    assert result["runner_status"] == "ok"
    value = result["observations"][0]["value"]
    assert value[:3] == [6, "2.5.2", "1.13.1"]
    site_packages = project_venv_site_packages().resolve()
    assert all(Path(module_path).resolve().is_relative_to(site_packages) for module_path in value[3:])


def test_dangerous_builtin_call_remains_rejected():
    safe, reason = validate_candidate_source("def f(path):\n    return open(path).read()\n")
    assert not safe
    assert reason == "denied_call:open"


def test_executes_simple_function():
    result = run_candidate("def f(x):\n    return x + 1\n", "f", [[1], [4]])
    assert result["runner_status"] == "ok"
    assert [item["value"] for item in result["observations"]] == [2, 5]


def test_infinite_candidate_returns_a_timeout():
    result = run_candidate(
        "def f(x):\n    while True:\n        pass\n",
        "f",
        [[1]],
        timeout_seconds=0.2,
    )
    assert result["runner_status"] == "timeout"
    assert result["observations"] == [{"status": "timeout"}]


def test_random_module_is_seeded_for_repeatable_probes():
    source = "import random\ndef f():\n    return random.random()\n"
    first = run_candidate(source, "f", [[]])
    second = run_candidate(source, "f", [[]])
    assert first["observations"] == second["observations"]


def test_generator_outputs_are_materialized_without_process_addresses():
    source = (
        "def f(limit):\n"
        "    for value in range(limit):\n"
        "        yield value * value\n"
    )
    first = run_candidate(source, "f", [[3]])
    second = run_candidate(source, "f", [[3]])
    expected = {
        "status": "ok",
        "value": {"__iterator__": [0, 1, 4], "__truncated__": False},
    }
    assert first["runner_status"] == "ok"
    assert first["observations"] == [expected]
    assert second["observations"] == [expected]


def test_generator_normalization_has_a_fixed_bound():
    source = "def f():\n    return (value for value in range(1000))\n"
    result = run_candidate(source, "f", [[]])
    assert result["runner_status"] == "ok"
    value = result["observations"][0]["value"]
    assert value["__truncated__"] is True
    assert value["__iterator__"] == list(range(execution.MAX_NORMALIZED_ITERATOR_ITEMS))


def test_opaque_return_reprs_do_not_leak_process_addresses():
    source = "def f():\n    return object()\n"
    first = run_candidate(source, "f", [[]])
    second = run_candidate(source, "f", [[]])
    assert first["runner_status"] == "ok"
    assert first["observations"] == second["observations"]
    assert "0x<address>" in first["observations"][0]["value"]["repr"]


def test_python_hash_seed_is_repeatable_across_worker_processes():
    source = "def f(value):\n    return hash(value)\n"
    observations = [run_candidate(source, "f", [["precisionprobe"]]) for _ in range(3)]
    assert all(result["runner_status"] == "ok" for result in observations)
    assert observations[0]["observations"] == observations[1]["observations"]
    assert observations[1]["observations"] == observations[2]["observations"]


def test_retries_windows_commitment_limit_from_process_creation(monkeypatch):
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            error = OSError("commitment limit")
            error.winerror = execution.WINDOWS_COMMITMENT_LIMIT_ERROR
            raise error
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout='__PRECISIONPROBE__[{"status":"ok","value":7}]\n',
            stderr="",
        )

    monkeypatch.setattr(execution.os, "name", "nt")
    monkeypatch.setattr(execution, "worker_python_executable", lambda: "python")
    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = run_candidate("def f():\n    return 7\n", "f", [[]])
    assert calls == 2
    assert result == {
        "observations": [{"status": "ok", "value": 7}],
        "runner_status": "ok",
        "reason": None,
    }


def test_retries_windows_commitment_limit_exit_status(monkeypatch):
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(args[0], execution.WINDOWS_STATUS_COMMITMENT_LIMIT, "", "")
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout='__PRECISIONPROBE__[{"status":"ok","value":8}]\n',
            stderr="",
        )

    monkeypatch.setattr(execution.os, "name", "nt")
    monkeypatch.setattr(execution, "worker_python_executable", lambda: "python")
    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = run_candidate("def f():\n    return 8\n", "f", [[]])
    assert calls == 2
    assert result["observations"] == [{"status": "ok", "value": 8}]


def test_commitment_limit_recognizes_signed_and_unsigned_ntstatus():
    assert is_windows_commitment_limit_returncode(execution.WINDOWS_STATUS_COMMITMENT_LIMIT)
    assert is_windows_commitment_limit_returncode(-execution.WINDOWS_STATUS_COMMITMENT_LIMIT)
    assert not is_windows_commitment_limit_returncode(0)


def test_unrelated_process_creation_error_is_not_retried(monkeypatch):
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        error = OSError("not commitment")
        error.winerror = 5
        raise error

    monkeypatch.setattr(execution.os, "name", "nt")
    monkeypatch.setattr(execution, "worker_python_executable", lambda: "python")
    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = run_candidate("def f():\n    return 8\n", "f", [[]])
    assert calls == 1
    assert result["reason"] == "worker_create_error:5"


def test_windows_worker_bypasses_the_virtual_environment_launcher():
    if os.name == "nt":
        assert worker_python_executable() == sys._base_executable
