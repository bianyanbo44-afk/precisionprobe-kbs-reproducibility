from __future__ import annotations

from typing import Any, Mapping


def first_status(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("EvalPlus status sequence is empty")
        value = value[0]
    if not isinstance(value, str):
        raise TypeError(f"EvalPlus status must be a string or sequence, got {type(value).__name__}")
    return value


def evalplus_passed(row: Mapping[str, Any], suite: str = "plus") -> bool:
    """Apply official or diagnostic EvalPlus correctness semantics.

    ``plus`` is the official EvalPlus definition and requires both suites.
    ``augmented`` is a sensitivity label based only on the result stored in the
    evaluator's ``plus`` field; it must not be reported as official Plus
    correctness.
    """

    base_passed = first_status(row["base"]) == "pass"
    if suite == "base":
        return base_passed
    if suite == "augmented":
        return first_status(row["plus"]) == "pass"
    if suite == "plus":
        return base_passed and first_status(row["plus"]) == "pass"
    raise ValueError(f"unsupported EvalPlus suite: {suite}")


def evalplus_error(row: Mapping[str, Any], suite: str = "plus") -> int:
    return int(not evalplus_passed(row, suite))
