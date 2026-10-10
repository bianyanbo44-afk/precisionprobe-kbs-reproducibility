from __future__ import annotations

import ast
import re
from copy import deepcopy
from typing import Any

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus


def load_benchmark(dataset: str, *, mini: bool = False) -> dict[str, dict[str, Any]]:
    if dataset == "humaneval":
        return get_human_eval_plus(mini=mini)
    if dataset == "mbpp":
        return get_mbpp_plus(mini=mini)
    raise ValueError(f"unsupported dataset: {dataset}")


def build_generation_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    prompt = problem["prompt"].rstrip()
    return [
        {
            "role": "system",
            "content": (
                "You are a careful Python programmer. Return only one complete Python program "
                "that defines the requested function. Do not include Markdown fences, tests, or explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Implement the function below. Preserve its name and signature. Include any required "
                f"standard-library imports.\n\n{prompt}"
            ),
        },
    ]


def extract_python_source(text: str, problem: dict[str, Any]) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    source = fenced.group(1) if fenced else text
    source = source.strip()
    entry_point = problem["entry_point"]
    if re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", source, re.MULTILINE):
        return source + "\n"
    return problem["prompt"].rstrip() + "\n" + source + "\n"


def _literal_call_arguments(expression: str, entry_point: str) -> list[Any] | None:
    try:
        parsed = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(parsed, ast.Call):
        return None
    function_name = parsed.func.id if isinstance(parsed.func, ast.Name) else None
    if function_name != entry_point or parsed.keywords:
        return None
    try:
        return [ast.literal_eval(argument) for argument in parsed.args]
    except (ValueError, TypeError, SyntaxError):
        return None


def _mutations(value: Any) -> list[Any]:
    candidates: list[Any] = []
    if isinstance(value, bool):
        candidates.append(not value)
    elif isinstance(value, int):
        candidates.extend([0, value + 1, value - 1])
    elif isinstance(value, float):
        candidates.extend([0.0, value * 0.5, value + 1.0])
    elif isinstance(value, str):
        candidates.extend(["", value[::-1], value + value[:1]])
    elif isinstance(value, list):
        candidates.extend([[], list(reversed(value)), value[: max(1, len(value) // 2)]])
    elif isinstance(value, tuple):
        candidates.extend([tuple(), tuple(reversed(value))])
    unique = []
    for candidate in candidates:
        if candidate != value and candidate not in unique:
            unique.append(candidate)
    return unique[:2]


def prompt_derived_probe_inputs(problem: dict[str, Any], *, max_inputs: int = 8) -> list[list[Any]]:
    """Extract doctest calls and create deterministic one-argument mutations.

    Only the public prompt is read. No benchmark test input or expected output is
    consulted, which enables a strict leakage-free analysis subset.
    """

    entry_point = problem["entry_point"]
    calls: list[list[Any]] = []
    for line in problem["prompt"].splitlines():
        if ">>>" not in line:
            continue
        arguments = _literal_call_arguments(line.split(">>>", 1)[1], entry_point)
        if arguments is not None and arguments not in calls:
            calls.append(arguments)

    probes = [deepcopy(arguments) for arguments in calls]
    for arguments in calls:
        for index, value in enumerate(arguments):
            for mutation in _mutations(value):
                changed = deepcopy(arguments)
                changed[index] = mutation
                if changed not in probes:
                    probes.append(changed)
                if len(probes) >= max_inputs:
                    return probes
    return probes[:max_inputs]


def choose_probe_inputs(
    problem: dict[str, Any],
    *,
    max_inputs: int = 8,
    min_prompt_inputs: int = 3,
    allow_base_fallback: bool = True,
) -> tuple[list[list[Any]], str]:
    prompt_inputs = prompt_derived_probe_inputs(problem, max_inputs=max_inputs)
    if len(prompt_inputs) >= min_prompt_inputs:
        return prompt_inputs, "prompt_only"
    if not allow_base_fallback:
        return prompt_inputs, "prompt_insufficient"
    combined = list(prompt_inputs)
    for item in problem.get("base_input", []):
        normalized = list(item) if isinstance(item, tuple) else item
        if normalized not in combined:
            combined.append(normalized)
        if len(combined) >= max_inputs:
            break
    return combined, "prompt_plus_base_inputs"
