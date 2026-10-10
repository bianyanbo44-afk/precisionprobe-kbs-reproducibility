from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Observation:
    status: str
    value: Any = None
    exception: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Observation":
        return cls(
            status=str(value.get("status", "unknown")),
            value=value.get("value"),
            exception=value.get("exception"),
        )


def _text_distance(left: Any, right: Any) -> float:
    left_text = json.dumps(left, ensure_ascii=False, sort_keys=True, default=repr)
    right_text = json.dumps(right, ensure_ascii=False, sort_keys=True, default=repr)
    if left_text == right_text:
        return 0.0
    return 1.0 - SequenceMatcher(None, left_text, right_text).ratio()


def _numeric_distance(left: float, right: float) -> float:
    if math.isnan(left) and math.isnan(right):
        return 0.0
    if not math.isfinite(left) or not math.isfinite(right):
        return 0.0 if left == right else 1.0
    scale = max(1.0, abs(left), abs(right))
    return min(1.0, abs(left - right) / scale)


def value_distance(left: Any, right: Any) -> float:
    """A bounded, type-aware distance between two observable return values."""

    if left == right:
        return 0.0
    if isinstance(left, bool) or isinstance(right, bool):
        return 1.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _numeric_distance(float(left), float(right))
    if isinstance(left, Sequence) and isinstance(right, Sequence) and not isinstance(
        left, (str, bytes)
    ) and not isinstance(right, (str, bytes)):
        if not left and not right:
            return 0.0
        shared = min(len(left), len(right))
        element_distance = (
            mean(value_distance(left[i], right[i]) for i in range(shared))
            if shared
            else 0.0
        )
        length_distance = abs(len(left) - len(right)) / max(len(left), len(right), 1)
        return min(1.0, 0.75 * element_distance + 0.25 * length_distance)
    return _text_distance(left, right)


def behavioral_distance(left: Observation, right: Observation) -> float:
    """Distance in [0, 1] over success, exception, timeout, and rejection behavior."""

    if left.status == "ok" and right.status == "ok":
        return value_distance(left.value, right.value)
    if left.status == right.status:
        if left.status == "exception":
            return 0.15 if left.exception == right.exception else 0.60
        return 0.0
    if "ok" in {left.status, right.status}:
        return 1.0
    if "timeout" in {left.status, right.status}:
        return 0.85
    return 0.70


def xpbd_score(
    left: Iterable[Mapping[str, Any] | Observation],
    right: Iterable[Mapping[str, Any] | Observation],
) -> float:
    """Mean paired behavioral distance; missing observations are maximal disagreement."""

    left_obs = [x if isinstance(x, Observation) else Observation.from_mapping(x) for x in left]
    right_obs = [x if isinstance(x, Observation) else Observation.from_mapping(x) for x in right]
    count = max(len(left_obs), len(right_obs))
    if count == 0:
        return 0.0
    distances = []
    for index in range(count):
        if index >= len(left_obs) or index >= len(right_obs):
            distances.append(1.0)
        else:
            distances.append(behavioral_distance(left_obs[index], right_obs[index]))
    return float(mean(distances))


def semantic_distance_entropy(
    executions: Sequence[Iterable[Mapping[str, Any] | Observation]],
) -> float:
    """Rao-style pairwise semantic-distance entropy over sampled programs."""

    count = len(executions)
    if count < 2:
        return 0.0
    total = 0.0
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            total += xpbd_score(executions[left_index], executions[right_index])
    return float(total / (count * count))


def dominant_semantic_distance_entropy(
    target: Iterable[Mapping[str, Any] | Observation],
    executions: Sequence[Iterable[Mapping[str, Any] | Observation]],
) -> float:
    """Target-anchored semantic-distance entropy for the served program."""

    if not executions:
        return 0.0
    return float(mean(xpbd_score(target, candidate) for candidate in executions))


def exact_disagreement_score(
    left: Iterable[Mapping[str, Any] | Observation],
    right: Iterable[Mapping[str, Any] | Observation],
) -> float:
    """Fraction of probes with non-identical observable behavior."""

    left_obs = [x if isinstance(x, Observation) else Observation.from_mapping(x) for x in left]
    right_obs = [x if isinstance(x, Observation) else Observation.from_mapping(x) for x in right]
    count = max(len(left_obs), len(right_obs))
    if count == 0:
        return 0.0
    disagreement = 0
    for index in range(count):
        if index >= len(left_obs) or index >= len(right_obs) or left_obs[index] != right_obs[index]:
            disagreement += 1
    return disagreement / count


def exact_semantic_distance_entropy(
    executions: Sequence[Iterable[Mapping[str, Any] | Observation]],
) -> float:
    """Rao-style entropy using exact execution disagreement as distance."""

    count = len(executions)
    if count < 2:
        return 0.0
    total = 0.0
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            total += exact_disagreement_score(executions[left_index], executions[right_index])
    return float(total / (count * count))


def exact_dominant_semantic_distance_entropy(
    target: Iterable[Mapping[str, Any] | Observation],
    executions: Sequence[Iterable[Mapping[str, Any] | Observation]],
) -> float:
    """Target-anchored entropy using exact execution disagreement."""

    if not executions:
        return 0.0
    return float(mean(exact_disagreement_score(target, candidate) for candidate in executions))


def ast_distance(left_source: str, right_source: str) -> float:
    """Normalized AST-text distance, with parse failure represented explicitly."""

    try:
        left_ast = ast.dump(ast.parse(left_source), annotate_fields=False)
    except SyntaxError:
        left_ast = "<PARSE_ERROR>"
    try:
        right_ast = ast.dump(ast.parse(right_source), annotate_fields=False)
    except SyntaxError:
        right_ast = "<PARSE_ERROR>"
    if left_ast == right_ast:
        return 0.0
    return 1.0 - SequenceMatcher(None, left_ast, right_ast).ratio()


def failure_fraction(observations: Iterable[Mapping[str, Any] | Observation]) -> float:
    values = [x if isinstance(x, Observation) else Observation.from_mapping(x) for x in observations]
    if not values:
        return 1.0
    return sum(item.status != "ok" for item in values) / len(values)
