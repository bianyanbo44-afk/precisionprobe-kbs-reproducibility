"""Statement-rule regression checks with published examples and hand-checkable cases.

Additional fixtures prove nonunique acceptance where the published examples
themselves have a unique optimum.  They are unit cases, not experimental data.
"""
from __future__ import annotations

from itertools import product
import json
import unittest

from statement_comparators import (
    FUNCTIONAL_TASK_POLICIES,
    MAX_OUTPUT_CHARS,
    TASK_POLICIES,
    matches_statement,
)


PUBLIC_EXAMPLES = {
    "atcoder/abc392_d": [
        ("3\n3 1 2 3\n4 1 2 2 1\n6 1 2 3 4 5 6", "0.333333333333333"),
        ("3\n5 1 1 1 1 1\n4 2 2 2 2\n3 1 1 2", "0.666666666666667"),
    ],
    "atcoder/abc395_a": [
        ("3\n1 2 5", "Yes"),
        ("3\n3 9 5", "No"),
        ("10\n1 1 2 3 5 8 13 21 34 55", "No"),
    ],
    "atcoder/arc192_a": [
        ("12\n0 1 0 1 1 1 1 0 1 1 1 0", "Yes"),
        ("3\n0 0 0", "No"),
        ("29\n" + " ".join(["1"] * 29), "Yes"),
    ],
    "atcoder/arc192_b": [
        ("3\n1 9 2", "Fennec"),
        ("2\n25 29", "Snuke"),
        ("6\n1 9 2 25 2 9", "Snuke"),
    ],
    "atcoder/abc396_e": [
        ("3 2\n1 3 4\n1 2 3", "0 3 4"),
        ("3 3\n1 3 4\n1 2 3\n2 3 5", "-1"),
        ("5 8\n4 2 4\n2 3 11\n3 4 15\n4 5 6\n3 2 11\n3 3 0\n3 1 9\n3 4 15", "0 2 9 6 0"),
    ],
    "atcoder/abc397_d": [
        ("397", "12 11"),
        ("1", "-1"),
        ("39977273855577088", "342756 66212"),
    ],
    "atcoder/arc190_a": [
        ("5 4\n2 4\n3 5\n1 4\n2 5", "2\n2 0 1 0"),
        ("5 4\n1 3\n1 5\n2 4\n3 5", "1\n0 1 0 0"),
        ("5 2\n1 3\n2 5", "2\n1 1"),
        ("5 2\n1 3\n2 4", "-1"),
    ],
    "atcoder/arc191_c": [
        ("4\n3\n16\n1\n55", "2 7\n11 68\n20250126 1\n33 662"),
    ],
}


def compare(task: str, output: str, expected: str, input_text: str = "") -> bool:
    return matches_statement(
        {"status": "ok", "raw_output": output, "value": None},
        expected, "stdin", task, input_text,
    )


class StatementComparatorTests(unittest.TestCase):
    def test_functional_probability_accepts_statement_tolerance(self):
        task = "leetcode/3763"
        self.assertEqual(set(FUNCTIONAL_TASK_POLICIES), {task})
        self.assertTrue(matches_statement({"status": "ok", "literal": repr(7/6)},
                                         "1.16667", "functional", task))
        self.assertTrue(matches_statement({"status": "ok", "literal": "1"},
                                         "1.00000", "functional", task))

    def test_functional_probability_rejects_outside_tolerance(self):
        task = "leetcode/3763"
        self.assertTrue(matches_statement({"status": "ok", "literal": "1.00001"},
                                         "1.0", "functional", task))
        self.assertFalse(matches_statement({"status": "ok", "literal": "1.00001001"},
                                          "1.0", "functional", task))
        for literal in ["1e999", "True", "None", "[1.0]", "'1.0'"]:
            self.assertFalse(matches_statement({"status": "ok", "literal": literal},
                                              "1.0", "functional", task))

    def test_functional_tolerance_does_not_relax_other_tasks(self):
        self.assertFalse(matches_statement({"status": "ok", "literal": repr(7/6)},
                                          "1.16667", "functional", "unlisted/task"))

    def test_policy_inventory_and_json_serialization(self):
        self.assertEqual(set(TASK_POLICIES), set(PUBLIC_EXAMPLES))
        self.assertEqual(json.loads(json.dumps(TASK_POLICIES)), TASK_POLICIES)

    def test_all_published_examples(self):
        for task, examples in PUBLIC_EXAMPLES.items():
            for index, (input_text, expected) in enumerate(examples, 1):
                with self.subTest(task=task, sample=index):
                    self.assertTrue(compare(task, expected + "\n", expected, input_text))

    def test_recorded_float_outputs_and_true_incorrect_values(self):
        task = "atcoder/abc392_d"
        for observed, (input_text, expected) in zip(
            ("0.3333333333333333", "0.6666666666666666"), PUBLIC_EXAMPLES[task]
        ):
            self.assertTrue(compare(task, observed, expected, input_text))
        expected = PUBLIC_EXAMPLES[task][0][1]
        self.assertTrue(compare(task, "0.33333334", expected))
        for incorrect in ("0.33333335", "0.16666666666666666", "1.0", "-0.333333333"):
            with self.subTest(output=incorrect):
                self.assertFalse(compare(task, incorrect, expected))
        self.assertTrue(compare(task, "-0.000000005", "0"))
        self.assertTrue(compare(task, "1.000000005", "1"))

    def test_float_nonfinite_and_malformed_outputs_fail(self):
        task = "atcoder/abc392_d"
        for output in ("NaN", "sNaN", "Inf", "-Infinity", "1e99999999", "0.3 0.3", "0.3\n0.3", ""):
            with self.subTest(output=output):
                self.assertFalse(compare(task, output, "0.333333333333333"))
        self.assertFalse(compare(task, "NaN", "NaN"))
        self.assertFalse(compare(task, "0.3", "Infinity"))

    def test_explicit_case_insensitivity_only(self):
        for task in ("atcoder/abc395_a", "atcoder/arc192_a", "atcoder/arc192_b"):
            for input_text, expected in PUBLIC_EXAMPLES[task]:
                for alternate in (expected.lower(), expected.upper(), expected.swapcase()):
                    self.assertTrue(compare(task, alternate, expected, input_text))
                self.assertFalse(compare(task, "YES NO", expected, input_text))
        self.assertFalse(compare("atcoder/abc395_a", "NO", "Yes"))
        self.assertFalse(compare("atcoder/arc192_b", "fennec", "Snuke"))
        self.assertFalse(compare("unlisted/task", "YES", "Yes"))

    def test_xor_valid_but_nonoptimal_published_sequences_fail(self):
        task = "atcoder/abc396_e"
        input_text, expected = PUBLIC_EXAMPLES[task][0]
        # The statement itself lists these as feasible but nonoptimal.
        for output in ("1 2 5", "7 4 3", "0 2 4", "-1", "0 3 4 0"):
            self.assertFalse(compare(task, output, expected, input_text))
        input_text, expected = PUBLIC_EXAMPLES[task][1]
        self.assertFalse(compare(task, "0 3 4", expected, input_text))

    def test_xor_tied_optima_and_isolated_vertices(self):
        # Restrict published sample 1 to its edge 1 2 3.  Both endpoints can
        # exchange tied bits while preserving the minimum sum of three.
        task = "atcoder/abc396_e"
        input_text = "2 1\n1 2 3"
        for output in ("0 3", "1 2", "2 1", "3 0"):
            self.assertTrue(compare(task, output, "0 3", input_text))
        self.assertFalse(compare(task, "4 7", "0 3", input_text))
        self.assertFalse(compare(task, "1 3", "0 3", input_text))
        self.assertTrue(compare(task, "1 2 0 0", "0 3 0 0", "4 1\n1 2 3"))
        self.assertFalse(compare(task, "1 2 1 0", "0 3 0 0", "4 1\n1 2 3"))
        self.assertTrue(compare(task, "0 0 0", "0 0 0", "3 0"))
        self.assertFalse(compare(task, "0 0 1", "0 0 0", "3 0"))
        self.assertTrue(compare(task, "-1", "-1", "1 1\n1 1 1"))

    def test_cubes_accept_distinct_positive_witnesses(self):
        task = "atcoder/abc397_d"
        # 9^3-1^3 and 12^3-10^3 are both 728; no canonical witness is required.
        self.assertTrue(compare(task, "9 1", "12 10", "728"))
        self.assertTrue(compare(task, "12 10", "9 1", "728"))
        for output in ("9 0", "10 12", "12 11", "9.0 1", "0 0", "-1"):
            self.assertFalse(compare(task, output, "12 10", "728"))
        self.assertFalse(compare(task, "-1", "12 11", "397"))
        self.assertFalse(compare(task, "1 0", "-1", "1"))

    def test_interval_cover_public_alternatives_and_nonoptimal_results(self):
        task = "atcoder/arc190_a"
        input_text, expected = PUBLIC_EXAMPLES[task][0]
        self.assertTrue(compare(task, "2\n2 0 0 1", expected, input_text))
        self.assertTrue(compare(task, "2\n0 1 1 0", expected, input_text))
        for output in ("3\n2 0 1 1", "2\n2 0 1 1", "2\n1 0 0 1", "2\n2 0 3 0", "-1"):
            self.assertFalse(compare(task, output, expected, input_text))
        input_text, expected = PUBLIC_EXAMPLES[task][1]
        self.assertFalse(compare(task, "2\n1 1 0 0", expected, input_text))

    def test_interval_cover_three_operation_and_duplicate_interval_cases(self):
        task = "atcoder/arc190_a"
        self.assertTrue(compare(task, "3\n2 1 2", "3\n2 1 2", "6 3\n2 4\n3 5\n4 6"))
        self.assertFalse(compare(task, "2\n2 1 0", "3\n2 1 2", "6 3\n2 4\n3 5\n4 6"))
        self.assertTrue(compare(task, "2\n1 2", "2\n2 1", "5 2\n2 4\n2 4"))
        self.assertTrue(compare(task, "1\n0 1", "1\n1 0", "1 2\n1 1\n1 1"))

    def test_interval_minimum_against_independent_small_enumeration(self):
        # The oracle enumerates all operation assignments on three physical
        # points, independently of the comparator's interval-pair argument.
        task, n = "atcoder/arc190_a", 3
        possible = [(left, right) for left in range(1, n + 1) for right in range(left, n + 1)]
        full = (1 << n) - 1
        for count in (1, 2, 3):
            for intervals in product(possible, repeat=count):
                inside_masks = [sum(1 << (point - 1) for point in range(left, right + 1)) for left, right in intervals]
                feasible = []
                for operations in product((0, 1, 2), repeat=count):
                    covered = 0
                    for inside, operation in zip(inside_masks, operations):
                        covered |= inside if operation == 1 else (full ^ inside if operation == 2 else 0)
                    if covered == full:
                        feasible.append((sum(operation != 0 for operation in operations), operations))
                input_text = f"{n} {count}\n" + "\n".join(f"{left} {right}" for left, right in intervals)
                if not feasible:
                    self.assertTrue(compare(task, "-1", "-1", input_text))
                    continue
                optimum, witness = min(feasible)
                expected = str(optimum) + "\n" + " ".join(map(str, witness))
                self.assertTrue(compare(task, expected, expected, input_text), (input_text, expected))
                nonoptimal = next((pair for pair in feasible if pair[0] > optimum), None)
                if nonoptimal:
                    cost, operations = nonoptimal
                    output = str(cost) + "\n" + " ".join(map(str, operations))
                    self.assertFalse(compare(task, output, expected, input_text), (input_text, output))

    def test_modular_order_published_and_congruent_alternatives(self):
        task = "atcoder/arc191_c"
        input_text, expected = PUBLIC_EXAMPLES[task][0]
        # (100,777) is explicitly supplied as another valid N=3 answer.
        self.assertTrue(compare(task, "100 777\n79 68\n1 1\n695 662", expected, input_text))
        self.assertTrue(compare(task, "100 777", "2 7", "1\n3"))
        self.assertTrue(compare(task, "5 1", "20250126 1", "1\n1"))

    def test_modular_order_requires_the_smallest_exponent(self):
        task = "atcoder/arc191_c"
        for output in ("1 7", "7 7", "0 7", "2 0", "2 7\n2 7", "NaN 7"):
            self.assertFalse(compare(task, output, "2 7", "1\n3"))
        self.assertFalse(compare(task, "3 4", "11 68", "1\n16"))
        self.assertFalse(compare(task, "2 1", "2 7", "1\n3"))

    def test_modular_order_largest_statement_bounds_without_giant_powers(self):
        task = "atcoder/arc191_c"
        # (1+N)^k = 1+k*N modulo N^2; the least positive exponent is N.
        self.assertTrue(compare(task, "1000000001 1000000000000000000", "unused", "1\n1000000000"))
        self.assertFalse(compare(task, "1000000001 1000000000000000001", "unused", "1\n1000000000"))

    def test_missing_input_invalid_ranges_and_oversized_values_fail(self):
        for task in ("atcoder/abc396_e", "atcoder/abc397_d", "atcoder/arc190_a", "atcoder/arc191_c"):
            self.assertFalse(compare(task, "-1", "-1"))
        self.assertFalse(compare("atcoder/abc396_e", "0", "0", "200001 0"))
        self.assertFalse(compare("atcoder/abc396_e", "0 0", "0 0", "2 1\n1 3 0"))
        self.assertFalse(compare("atcoder/abc397_d", "12 11", "12 11", "1000000000000000001"))
        self.assertFalse(compare("atcoder/abc397_d", "9" * 100000 + " 1", "12 11", "397"))
        self.assertFalse(compare("atcoder/arc190_a", "1\n1", "1\n1", "5 1\n4 2"))
        self.assertFalse(compare("atcoder/arc191_c", "1 1", "1 1", "10001\n1"))
        self.assertFalse(compare("atcoder/abc392_d", "0" * (MAX_OUTPUT_CHARS + 1), "0"))

    def test_unknown_tasks_keep_line_and_decimal_conventions(self):
        task = "unlisted/task"
        self.assertTrue(compare(task, " 1.000  2e1 \n 3 ", "1 20\n3"))
        self.assertFalse(compare(task, "1 20 3", "1 20\n3"))
        self.assertTrue(compare(task, "word\nother", "word\nother"))
        self.assertFalse(compare(task, "word\nother", "WORD\nother"))

    def test_shortest_prefix_palindrome_has_one_answer_and_uses_legacy_rule(self):
        # abc398_f has generic nonunique wording, but a shortest fixed-prefix
        # palindrome is unique.  These are its three published examples.
        task = "atcoder/abc398_f"
        self.assertNotIn(task, TASK_POLICIES)
        for input_text, expected in (("ABC", "ABCBA"), ("Z", "Z"), ("TREE", "TREERT")):
            self.assertTrue(compare(task, expected, expected, input_text))
        self.assertFalse(compare(task, "ABCCBA", "ABCBA", "ABC"))

    def test_functional_literal_json_convention_and_status(self):
        self.assertTrue(matches_statement({"status": "ok", "literal": "[1, 2]"}, "[1,2]", "functional", "unlisted/task"))
        self.assertTrue(matches_statement({"status": "ok", "literal": "{'a': True}"}, '{"a":true}', "functional", "atcoder/abc392_d"))
        self.assertFalse(matches_statement({"status": "ok", "literal": "(1, 2)"}, "[1,2]", "functional", "unlisted/task"))
        self.assertFalse(matches_statement({"status": "ok", "literal": "f()"}, "0", "functional", "unlisted/task"))
        for status in ("exception", "timeout", "unknown"):
            self.assertFalse(matches_statement({"status": status, "raw_output": "Yes"}, "Yes", "stdin", "atcoder/abc395_a"))
        self.assertFalse(matches_statement({"status": "ok", "value": [[1]]}, "1", "stdin", "unlisted/task"))


if __name__ == "__main__":
    unittest.main()
