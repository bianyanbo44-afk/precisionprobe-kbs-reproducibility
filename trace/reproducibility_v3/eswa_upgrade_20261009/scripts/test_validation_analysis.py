import unittest

import numpy as np
import pandas as pd

from analyze_validation import SCORES, primary_comparison, public_pass_diagnostic
from analyze_retrospective import expected_review


class PairedBootstrapChecks(unittest.TestCase):
    def make_frame(self, reversed_second=False):
        rows = []
        for condition in range(6):
            for task, error in enumerate([0, 1, 0, 1, 0, 1, 0, 1]):
                left, right = error, 1 - error
                if reversed_second and condition >= 3:
                    left, right = right, left
                rows.append({"condition": str(condition), "task_id": str(task),
                             "error": error, "dsde_4": left, "public_then_nll": right})
        return pd.DataFrame(rows)

    def test_identical_cells_have_identical_resampled_differences(self):
        result, draws = primary_comparison(self.make_frame(), repeats=50)
        np.testing.assert_array_equal(draws, np.repeat(draws[:, :1], 6, axis=1))
        self.assertEqual(result["macro"]["difference"], -.25)
        self.assertEqual(result["macro"]["ci95"], result["cells"][0]["ci95"])

    def test_paired_opposites_cancel_in_every_replicate(self):
        result, draws = primary_comparison(self.make_frame(True), repeats=50)
        np.testing.assert_allclose(draws.mean(axis=1), 0, atol=1e-15)
        self.assertEqual(result["macro"]["difference"], 0)

    def test_task_order_does_not_change_inference(self):
        frame = self.make_frame(True)
        first, draws1 = primary_comparison(frame, repeats=50)
        second, draws2 = primary_comparison(frame.sample(frac=1, random_state=19), repeats=50)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(draws1, draws2)


class PublicExampleChecks(unittest.TestCase):
    def test_known_failures_fill_queue_and_reach_error_lower_bound(self):
        errors = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        public_failures = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        token = np.array([.3, .2, .1, .8, .7, .9, .6, .5])
        score = 2*public_failures + token/(1+token)
        result = expected_review(errors, score, .25)
        lower_bound = (errors.sum() - result["reviewed"]) / len(errors)
        self.assertEqual(result["residual_error_per_task"], lower_bound)

    def test_no_public_passing_targets_is_explicit_not_dropped(self):
        frame = pd.DataFrame([{"condition": "example_q4", "error": 1,
                               **{score: 1.0 for score in SCORES}} for _ in range(4)])
        metrics, contrasts = public_pass_diagnostic(frame)
        self.assertTrue(metrics.empty)
        self.assertEqual(len(contrasts), 1)
        self.assertEqual(contrasts[0]["public_passing_tasks"], 0)
        self.assertTrue(contrasts[0]["public_failures_fill_primary_queue"])
        self.assertNotIn("residual_difference", contrasts[0])


if __name__ == "__main__":
    unittest.main()
