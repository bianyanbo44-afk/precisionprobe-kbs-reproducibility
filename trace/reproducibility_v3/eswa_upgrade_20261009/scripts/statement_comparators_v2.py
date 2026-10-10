"""Bounded, deterministic comparisons for explicit task-statement output rules.

The original locked evaluator is not modified.  Only the eight rules in
TASK_POLICIES differ from its line/Decimal comparison.  Functional observations
retain the original literal/JSON convention.  No candidate code or model is
executed here.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
import json
from math import gcd, isqrt
import re
from typing import Any


MAX_INPUT_CHARS = 8 * 1024 * 1024
MAX_OUTPUT_CHARS = 4 * 1024 * 1024
MAX_INTEGER_TOKEN_CHARS = 32
MAX_DECIMAL_TOKEN_CHARS = 512
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_XOR_MAX = (1 << 30) - 1

TASK_POLICIES: dict[str, dict[str, Any]] = {
    "atcoder/abc392_d": {
        "kind": "finite_numeric_tolerance",
        "source_id": "questions.jsonl:34/atcoder/abc392_d/Output",
        "absolute_tolerance": "1e-8",
        "relative_tolerance": "1e-8",
        "reference": "supplied expected probability",
        "finite_only": True,
        "output_values": 1,
    },
    "atcoder/abc395_a": {
        "kind": "case_insensitive_answer",
        "source_id": "questions.jsonl:52/atcoder/abc395_a/Output",
        "answers": ["yes", "no"],
    },
    "atcoder/arc192_a": {
        "kind": "case_insensitive_answer",
        "source_id": "questions.jsonl:95/atcoder/arc192_a/Output",
        "answers": ["yes", "no"],
    },
    "atcoder/arc192_b": {
        "kind": "case_insensitive_answer",
        "source_id": "questions.jsonl:93/atcoder/arc192_b/Output",
        "answers": ["fennec", "snuke"],
    },
    "atcoder/abc396_e": {
        "kind": "minimum_sum_xor_assignment",
        "source_id": "questions.jsonl:58/atcoder/abc396_e/Output+Constraints",
        "input_required": True,
        "maximum_n": 200000,
        "maximum_m": 100000,
        "maximum_z": 1000000000,
        "optimality": "exact component-wise bit minimum derived from input",
        "maximum_optimal_output_integer": _XOR_MAX,
        "nonunique_optima_allowed": True,
    },
    "atcoder/abc397_d": {
        "kind": "positive_cubic_difference",
        "source_id": "questions.jsonl:61/atcoder/abc397_d/Output+Constraints",
        "input_required": True,
        "maximum_n": 1000000000000000000,
        "positive_witness": "x^3-y^3=N with positive x and y",
        "no_solution_answer": "-1 requires the supplied no-solution reference",
        "derived_output_integer_bound": 1000000000,
        "nonunique_solutions_allowed": True,
    },
    "atcoder/arc190_a": {
        "kind": "minimum_cost_interval_cover",
        "source_id": "questions.jsonl:87/atcoder/arc190_a/Output+Constraints",
        "input_required": True,
        "maximum_n": 1000000,
        "maximum_m": 200000,
        "optimality": "exact minimum of one, two, three, or infeasible",
        "operation_values": [0, 1, 2],
        "nonunique_optima_allowed": True,
    },
    "atcoder/arc191_c": {
        "kind": "exact_multiplicative_order",
        "source_id": "questions.jsonl:90/atcoder/arc191_c/Output+Constraints",
        "input_required": True,
        "maximum_cases": 10000,
        "maximum_n": 1000000000,
        "maximum_a_and_m": 1000000000000000000,
        "verification": "gcd and modular powers for distinct prime divisors of N",
        "nonunique_solutions_allowed": True,
    },
}


def _text(value: Any, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("missing or oversized text")
    return value


def _tokens(value: str, maximum: int) -> list[str]:
    # Limited splitting also bounds allocation for many tiny output tokens.
    words = value.split(None, maximum)
    if len(words) > maximum:
        raise ValueError("too many tokens")
    return words


def _integer(token: str, lower: int, upper: int) -> int:
    if len(token) > MAX_INTEGER_TOKEN_CHARS or not _INTEGER.fullmatch(token):
        raise ValueError("invalid integer token")
    value = int(token)
    if not lower <= value <= upper:
        raise ValueError("integer outside task bounds")
    return value


def _integers(value: str, count: int, lower: int, upper: int) -> list[int]:
    words = _tokens(value, count)
    if len(words) != count:
        raise ValueError("wrong integer count")
    return [_integer(word, lower, upper) for word in words]


def _header(value: str) -> tuple[int, int]:
    words = value.split(None, 2)
    if len(words) < 2:
        raise ValueError("missing header")
    return _integer(words[0], 1, 1000000), _integer(words[1], 0, 200000)


def _legacy_stdio(raw: str, expected: str) -> bool:
    actual_lines = [line.strip() for line in raw.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(actual_lines) != len(expected_lines):
        return False
    for actual, wanted in zip(actual_lines, expected_lines):
        if actual == wanted:
            continue
        left, right = actual.split(), wanted.split()
        if len(left) != len(right):
            return False
        if any(len(token) > MAX_DECIMAL_TOKEN_CHARS for token in left) or any(
            len(token) > MAX_DECIMAL_TOKEN_CHARS for token in right
        ):
            return False
        try:
            if any(Decimal(a) != Decimal(b) for a, b in zip(left, right)):
                return False
        except InvalidOperation:
            return False
    return True


def _probability(raw: str, expected: str) -> bool:
    actual_words, expected_words = _tokens(raw, 1), _tokens(expected, 1)
    if len(actual_words) != 1 or len(expected_words) != 1:
        return False
    if max(len(actual_words[0]), len(expected_words[0])) > 128:
        return False
    actual, wanted = Decimal(actual_words[0]), Decimal(expected_words[0])
    if not actual.is_finite() or not wanted.is_finite():
        return False
    if not Decimal(0) <= wanted <= Decimal(1):
        return False
    # This preliminary bound rejects enormous exponents before arithmetic, but
    # permits small negative/above-one approximations within the stated tolerance.
    if not Decimal(-1) <= actual <= Decimal(2):
        return False
    with localcontext() as context:
        context.prec = 80
        tolerance = Decimal("1e-8")
        difference = abs(actual - wanted)
        return difference <= max(tolerance, tolerance * abs(wanted))


def _case_insensitive(raw: str, expected: str, policy: dict[str, Any]) -> bool:
    actual_words, expected_words = _tokens(raw, 1), _tokens(expected, 1)
    if len(actual_words) != 1 or len(expected_words) != 1:
        return False
    actual, wanted = actual_words[0].casefold(), expected_words[0].casefold()
    return wanted in policy["answers"] and actual == wanted


def _xor_minimum(raw: str, input_text: str) -> bool:
    n, m = _header(input_text)
    if n > 200000 or m > 100000:
        return False
    output_words = _tokens(raw, n)
    no_solution = output_words == ["-1"]
    if not no_solution and len(output_words) != n:
        return False
    actual = None if no_solution else [_integer(word, 0, _XOR_MAX) for word in output_words]
    fields = _integers(input_text, 2 + 3 * m, 0, 1000000000)
    parent = list(range(n))
    size = [1] * n
    xor_parent = [0] * n

    def find(node: int) -> tuple[int, int]:
        root, difference = node, 0
        while parent[root] != root:
            difference ^= xor_parent[root]
            root = parent[root]
        remaining = difference
        while parent[node] != node:
            previous, edge = parent[node], xor_parent[node]
            parent[node], xor_parent[node] = root, remaining
            remaining ^= edge
            node = previous
        return root, difference

    inconsistent = False
    for index in range(m):
        left, right, value = fields[2 + 3 * index:5 + 3 * index]
        if not 1 <= left <= n or not 1 <= right <= n:
            return False
        if inconsistent:
            continue
        root_l, offset_l = find(left - 1)
        root_r, offset_r = find(right - 1)
        difference = offset_l ^ offset_r ^ value
        if root_l == root_r:
            inconsistent = difference != 0
        else:
            if size[root_l] < size[root_r]:
                root_l, root_r = root_r, root_l
            parent[root_r], xor_parent[root_r] = root_l, difference
            size[root_l] += size[root_r]
    if inconsistent:
        return no_solution
    if no_solution:
        return False

    # Every component has one common XOR offset.  Each bit of that offset is
    # chosen independently to minimize the number of ones, allowing tied optima.
    component_offsets: dict[int, int] = {}
    bit_counts: dict[int, list[int]] = {}
    for node in range(n):
        root, offset = find(node)
        shift = actual[node] ^ offset
        if root in component_offsets and component_offsets[root] != shift:
            return False
        component_offsets[root] = shift
        if size[root] == 1:
            if actual[node] != 0:
                return False
            continue
        counts = bit_counts.get(root)
        if counts is None:
            counts = [0] * 30
            bit_counts[root] = counts
        while offset:
            bit = offset & -offset
            counts[bit.bit_length() - 1] += 1
            offset ^= bit
    minimum = sum(
        min(ones, size[root] - ones) << bit
        for root, counts in bit_counts.items()
        for bit, ones in enumerate(counts)
    )
    return sum(actual) == minimum


def _cubic_difference(raw: str, expected: str, input_text: str) -> bool:
    n = _integers(input_text, 1, 1, 10**18)[0]
    words = _tokens(raw, 2)
    if words == ["-1"]:
        # A no-solution answer uses the existing benchmark oracle; enumerating
        # up to a million divisors on each failed candidate is unnecessary.
        return _tokens(expected, 2) == ["-1"]
    if len(words) != 2:
        return False
    x, y = (_integer(word, 1, 10**9) for word in words)
    return x > y and x * x * x - y * y * y == n


def _minimum_cover_cost(n: int, intervals: list[tuple[int, int]]) -> int | None:
    if any(left == 1 and right == n for left, right in intervals):
        return 1
    if len(intervals) == 1:
        return None
    # Two outside operations cover all points when their original intervals
    # are disjoint.  One inside plus one outside works for a containing pair.
    if max(left for left, _ in intervals) > min(right for _, right in intervals):
        return 2
    max_right = 0
    for left, right in sorted(intervals, key=lambda pair: (pair[0], -pair[1])):
        if right <= max_right:
            return 2
        max_right = right
    # Two inside operations must connect an interval starting at 1 to one
    # ending at N, with overlap or adjacency on the integer grid.
    from_one = max((right for left, right in intervals if left == 1), default=0)
    to_n = min((left for left, right in intervals if right == n), default=n + 2)
    if from_one and from_one + 1 >= to_n:
        return 2
    if len(intervals) == 2:
        return None
    # With no containment, endpoints of three L-sorted intervals both strictly
    # increase.  The intersection of the outer two lies inside the middle one.
    # Outside, inside, outside therefore covers all points in three operations.
    return 3


def _interval_cover(raw: str, input_text: str) -> bool:
    n, m = _header(input_text)
    if not 1 <= m <= 200000:
        return False
    fields = _integers(input_text, 2 + 2 * m, 1, 1000000)
    intervals = []
    for index in range(m):
        left, right = fields[2 + 2 * index:4 + 2 * index]
        if not 1 <= left <= right <= n:
            return False
        intervals.append((left, right))
    optimum = _minimum_cover_cost(n, intervals)
    words = _tokens(raw, m + 1)
    if words == ["-1"]:
        return optimum is None
    if optimum is None or len(words) != m + 1:
        return False
    cost = _integer(words[0], 1, m)
    operations = [_integer(word, 0, 2) for word in words[1:]]
    if cost != optimum or sum(operation != 0 for operation in operations) != cost:
        return False
    covered_intervals = []
    for (left, right), operation in zip(intervals, operations):
        if operation == 1:
            covered_intervals.append((left, right))
        elif operation == 2:
            if left > 1:
                covered_intervals.append((1, left - 1))
            if right < n:
                covered_intervals.append((right + 1, n))
    covered = 0
    for left, right in sorted(covered_intervals):
        if left > covered + 1:
            return False
        covered = max(covered, right)
    return covered == n


@lru_cache(maxsize=1)
def _factor_primes() -> tuple[int, ...]:
    limit = isqrt(10**9)
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for prime in range(2, isqrt(limit) + 1):
        if sieve[prime]:
            start = prime * prime
            sieve[start:limit + 1:prime] = b"\x00" * ((limit - start) // prime + 1)
    return tuple(index for index, flag in enumerate(sieve) if flag)


@lru_cache(maxsize=10000)
def _distinct_prime_factors(n: int) -> tuple[int, ...]:
    factors = []
    for prime in _factor_primes():
        if prime * prime > n:
            break
        if n % prime == 0:
            factors.append(prime)
            while n % prime == 0:
                n //= prime
    if n > 1:
        factors.append(n)
    return tuple(factors)


def _multiplicative_order(raw: str, input_text: str) -> bool:
    first = input_text.split(None, 1)
    if not first:
        return False
    count = _integer(first[0], 1, 10000)
    fields = _integers(input_text, count + 1, 1, 10**9)
    # The statement specifies one pair per test-case line, so don't accept
    # missing/extra case lines or merge several pairs into a single line.
    lines = raw.strip().splitlines()
    if len(lines) != count:
        return False
    for n, line in zip(fields[1:], lines):
        a, modulus = _integers(line, 2, 1, 10**18)
        if gcd(a, modulus) != 1:
            return False
        identity = 1 % modulus
        if pow(a, n, modulus) != identity:
            return False
        if any(pow(a, n // prime, modulus) == identity for prime in _distinct_prime_factors(n)):
            return False
    return True


def matches_statement(
    observation: Mapping[str, Any],
    expected: str,
    testtype: str,
    task_id: str,
    input_text: str = "",
) -> bool:
    """Compare one observation using an explicit statement policy or old rules.

    Constructive tasks require their original input.  The dispatcher fails
    closed for malformed, out-of-range, non-finite or oversized observations.
    Stdio comparisons use raw_output, preserving precision and formatting;
    normalized value is deliberately not reconstructed into purported raw text.
    """
    if not isinstance(observation, Mapping) or observation.get("status") != "ok":
        return False
    try:
        expected = _text(expected, MAX_OUTPUT_CHARS)
        if testtype == "functional":
            literal = _text(observation.get("literal"), MAX_OUTPUT_CHARS)
            return ast.literal_eval(literal) == json.loads(expected)
        raw = _text(observation.get("raw_output"), MAX_OUTPUT_CHARS)
        policy = TASK_POLICIES.get(task_id)
        if policy is None:
            return _legacy_stdio(raw, expected)
        kind = policy["kind"]
        if kind == "finite_numeric_tolerance":
            return _probability(raw, expected)
        if kind == "case_insensitive_answer":
            return _case_insensitive(raw, expected, policy)
        input_text = _text(input_text, MAX_INPUT_CHARS)
        if kind == "minimum_sum_xor_assignment":
            return _xor_minimum(raw, input_text)
        if kind == "positive_cubic_difference":
            return _cubic_difference(raw, expected, input_text)
        if kind == "minimum_cost_interval_cover":
            return _interval_cover(raw, input_text)
        if kind == "exact_multiplicative_order":
            return _multiplicative_order(raw, input_text)
        return _legacy_stdio(raw, expected)
    except (ValueError, TypeError, KeyError, InvalidOperation, OverflowError, RecursionError, MemoryError):
        return False
