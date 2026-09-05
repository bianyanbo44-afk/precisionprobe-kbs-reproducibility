from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ALLOWED_IMPORT_ROOTS = {
    "bisect",
    "calendar",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "decimal",
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
    "random",
    "re",
    "statistics",
    "string",
    "sympy",
    "sys",
    "typing",
}

DANGEROUS_CALLS = {"breakpoint", "compile", "eval", "exec", "input", "open", "__import__"}
WINDOWS_COMMITMENT_LIMIT_ERROR = 1455
WINDOWS_STATUS_COMMITMENT_LIMIT = 0xC000012D
RESOURCE_RETRY_ATTEMPTS = 2
MAX_NORMALIZED_ITERATOR_ITEMS = 256
WSL_DISTRIBUTION = "Ubuntu"


def worker_python_executable() -> str:
    """Return the directly owned interpreter used for a restricted worker.

    Bypassing the Windows virtual-environment launcher removes an unnecessary
    process layer from timeout handling.
    """

    if os.name == "nt":
        return str(getattr(sys, "_base_executable", sys.executable))
    return sys.executable


def project_venv_site_packages() -> Path:
    """Return the sole third-party package root exposed to workers."""

    project_root = Path(__file__).resolve().parents[2]
    venv_root = project_root / ".venv"
    if os.name == "nt":
        site_packages = venv_root / "Lib" / "site-packages"
    else:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = venv_root / "lib" / version / "site-packages"
    if not site_packages.is_dir():
        raise RuntimeError(f"project virtual-environment site-packages not found: {site_packages}")
    return site_packages


def is_windows_commitment_limit_returncode(returncode: int) -> bool:
    """Accept the signed and unsigned Python representations of NTSTATUS."""

    return returncode in {
        WINDOWS_STATUS_COMMITMENT_LIMIT,
        -WINDOWS_STATUS_COMMITMENT_LIMIT,
    }


def validate_candidate_source(source: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            denied = roots - ALLOWED_IMPORT_ROOTS
            if denied:
                return False, f"denied_import:{sorted(denied)[0]}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                return False, f"denied_import:{root or '<relative>'}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
                return False, f"denied_call:{node.func.id}"
    return True, None


def _worker_source(
    candidate: str,
    entry_point: str,
    inputs: Iterable[Any],
    site_packages: Path,
) -> str:
    return f'''\
import sys as __pp_sys

__pp_sys.path.insert(0, {str(site_packages)!r})

import copy as __pp_copy
import json as __pp_json
import random as __pp_random
import re as __pp_re
from collections.abc import Iterator as __pp_Iterator

__pp_random.seed(0)

{candidate}

def __pp_stable_repr(value):
    try:
        text = repr(value)
    except BaseException:
        return "<unrepresentable>"
    return __pp_re.sub(r"0x[0-9A-Fa-f]+", "0x<address>", text)

def __pp_normalize(value, depth=0):
    if depth > 12:
        return {{"__type__": type(value).__name__, "repr": __pp_stable_repr(value)}}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [__pp_normalize(item, depth + 1) for item in value]
    if isinstance(value, set):
        return {{"__set__": sorted((__pp_normalize(item, depth + 1) for item in value), key=repr)}}
    if isinstance(value, dict):
        items = [(__pp_normalize(key, depth + 1), __pp_normalize(item, depth + 1)) for key, item in value.items()]
        return {{"__dict__": sorted(items, key=repr)}}
    if isinstance(value, __pp_Iterator):
        items = []
        for _ in range({MAX_NORMALIZED_ITERATOR_ITEMS}):
            try:
                item = next(value)
            except StopIteration:
                return {{"__iterator__": items, "__truncated__": False}}
            except BaseException as exc:
                return {{
                    "__iterator__": items,
                    "__iterator_exception__": type(exc).__name__,
                }}
            items.append(__pp_normalize(item, depth + 1))
        return {{"__iterator__": items, "__truncated__": True}}
    return {{"__type__": type(value).__name__, "repr": __pp_stable_repr(value)}}

__pp_inputs = {repr(list(inputs))}
__pp_results = []
__pp_fn = globals()[{entry_point!r}]
for __pp_args in __pp_inputs:
    try:
        if not isinstance(__pp_args, (list, tuple)):
            __pp_args = [__pp_args]
        __pp_value = __pp_fn(*__pp_copy.deepcopy(__pp_args))
        __pp_results.append({{"status": "ok", "value": __pp_normalize(__pp_value)}})
    except BaseException as __pp_exc:
        __pp_results.append({{"status": "exception", "exception": type(__pp_exc).__name__}})
print("__PRECISIONPROBE__" + __pp_json.dumps(__pp_results, ensure_ascii=False, sort_keys=True))
'''


def run_candidate(
    source: str,
    entry_point: str,
    inputs: Iterable[Any],
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    """Execute a benchmark candidate with coarse restrictions.

    This protects the experiment from accidental imports and hangs. It is not a
    security boundary against adversarial Python and must be replaced by OS-level
    isolation before executing untrusted real-world code.
    """

    safe, reason = validate_candidate_source(source)
    input_list = list(inputs)
    if not safe:
        status = "parse_error" if reason and reason.startswith("syntax_error") else "rejected"
        return {
            "observations": [{"status": status, "exception": reason} for _ in input_list],
            "runner_status": status,
            "reason": reason,
        }

    with tempfile.TemporaryDirectory(prefix="precisionprobe-") as temporary:
        worker_path = Path(temporary) / "worker.py"
        worker_path.write_text(
            _worker_source(source, entry_point, input_list, project_venv_site_packages()),
            encoding="utf-8",
        )
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
        }
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(RESOURCE_RETRY_ATTEMPTS):
            try:
                completed = subprocess.run(
                    [worker_python_executable(), "-s", "-S", "-P", str(worker_path)],
                    cwd=temporary,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                    creationflags=creation_flags,
                )
            except subprocess.TimeoutExpired:
                return {
                    "observations": [{"status": "timeout"} for _ in input_list],
                    "runner_status": "timeout",
                    "reason": "process_timeout",
                }
            except OSError as exc:
                if os.name == "nt" and exc.winerror == WINDOWS_COMMITMENT_LIMIT_ERROR and attempt + 1 < RESOURCE_RETRY_ATTEMPTS:
                    continue
                return {
                    "observations": [
                        {"status": "runner_error", "exception": f"worker_create_error:{exc.winerror}"}
                        for _ in input_list
                    ],
                    "runner_status": "runner_error",
                    "reason": f"worker_create_error:{exc.winerror}",
                }
            if (
                os.name == "nt"
                and is_windows_commitment_limit_returncode(completed.returncode)
                and attempt + 1 < RESOURCE_RETRY_ATTEMPTS
            ):
                continue
            break

        assert completed is not None

        marker = "__PRECISIONPROBE__"
        payload_line = next(
            (line[len(marker) :] for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
            None,
        )
        if payload_line is None:
            reason = f"worker_exit_{completed.returncode}"
            return {
                "observations": [{"status": "runner_error", "exception": reason} for _ in input_list],
                "runner_status": "runner_error",
                "reason": reason,
                "stderr_tail": completed.stderr[-1000:],
            }
        try:
            observations = json.loads(payload_line)
        except json.JSONDecodeError:
            return {
                "observations": [{"status": "runner_error", "exception": "invalid_json"} for _ in input_list],
                "runner_status": "runner_error",
                "reason": "invalid_worker_json",
            }
        return {"observations": observations, "runner_status": "ok", "reason": None}


def _wsl_project_paths() -> tuple[str, str]:
    """Return the WSL Python and site-package paths for this checkout."""

    project_root = Path(__file__).resolve().parents[2]
    if not project_root.drive:
        raise RuntimeError(f"WSL execution requires a Windows checkout path: {project_root}")
    drive = project_root.drive.rstrip(":").lower()
    relative_parts = project_root.parts[1:]
    relative = "/".join(relative_parts)
    python = f"/mnt/{drive}/{relative}/.venv-wsl/bin/python"
    site_packages = (
        f"/mnt/{drive}/{relative}/.venv-wsl/lib/"
        f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    return python, site_packages


def run_candidate_wsl(
    source: str,
    entry_point: str,
    inputs: Iterable[Any],
    *,
    timeout_seconds: float = 4.0,
    distribution: str = WSL_DISTRIBUTION,
) -> dict[str, Any]:
    """Execute a candidate through the project's Linux WSL environment.

    The existing source validation remains in force. WSL provides process and
    filesystem separation from the Windows host, but this is still not a
    hardened adversarial sandbox. It is intended for benchmark completions
    generated by this study, not arbitrary user-supplied code.
    """

    safe, reason = validate_candidate_source(source)
    input_list = list(inputs)
    if not safe:
        status = "parse_error" if reason and reason.startswith("syntax_error") else "rejected"
        return {
            "observations": [{"status": status, "exception": reason} for _ in input_list],
            "runner_status": status,
            "reason": reason,
        }

    worker_python, site_packages = _wsl_project_paths()
    worker = _worker_source(source, entry_point, input_list, Path(site_packages))
    command = [
        "wsl.exe",
        "-d",
        distribution,
        "--",
        worker_python,
        "-s",
        "-S",
        "-P",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=worker,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        return {
            "observations": [{"status": "timeout"} for _ in input_list],
            "runner_status": "timeout",
            "reason": "wsl_process_timeout",
        }
    except OSError as exc:
        return {
            "observations": [
                {"status": "runner_error", "exception": f"wsl_worker_create_error:{exc}"}
                for _ in input_list
            ],
            "runner_status": "runner_error",
            "reason": f"wsl_worker_create_error:{exc}",
        }

    marker = "__PRECISIONPROBE__"
    payload_line = next(
        (
            line[len(marker) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(marker)
        ),
        None,
    )
    if payload_line is None:
        reason = f"wsl_worker_exit_{completed.returncode}"
        return {
            "observations": [
                {"status": "runner_error", "exception": reason} for _ in input_list
            ],
            "runner_status": "runner_error",
            "reason": reason,
            "stderr_tail": completed.stderr[-1000:],
        }
    try:
        observations = json.loads(payload_line)
    except json.JSONDecodeError:
        return {
            "observations": [
                {"status": "runner_error", "exception": "invalid_wsl_worker_json"}
                for _ in input_list
            ],
            "runner_status": "runner_error",
            "reason": "invalid_wsl_worker_json",
        }
    return {"observations": observations, "runner_status": "ok", "reason": None}
