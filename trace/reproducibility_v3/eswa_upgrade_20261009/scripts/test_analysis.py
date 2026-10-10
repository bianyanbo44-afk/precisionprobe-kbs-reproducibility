import itertools
import unittest

import numpy as np

from analyze_retrospective import expected_review, percentile, split_ids, tie_aware_aurc, upper_cp


class AllocationChecks(unittest.TestCase):
    def test_uninformative_score_has_random_expected_capture(self):
        y = [1, 0, 1, 0]
        result = expected_review(y, [0, 0, 0, 0], .25)
        self.assertAlmostEqual(result["error_capture"], .25)
        self.assertAlmostEqual(result["unreviewed_error_rate"], .5)

    def test_tie_aurc_matches_exhaustive_permutations(self):
        y = [1, 0, 1, 0]
        expected = np.mean([np.mean(np.cumsum(p) / np.arange(1, 5)) for p in itertools.permutations(y)])
        self.assertAlmostEqual(tie_aware_aurc(y, [0, 0, 0, 0]), expected)

    def test_perfect_ranking_removes_errors_first(self):
        result = expected_review([0, 1, 0, 1], [.1, .9, .2, .8], .5)
        self.assertEqual(result["residual_error_per_task"], 0)
        self.assertEqual(result["error_capture"], 1)

    def test_reference_calibration_test_are_disjoint(self):
        ids = [str(i) for i in range(164)]
        ref, cal, test = split_ids(ids, "fixed")
        self.assertEqual((len(ref), len(cal), len(test)), (41, 41, 82))
        self.assertFalse(ref & cal or cal & test or ref & test)
        self.assertEqual(ref | cal | test, set(ids))

    def test_monotone_transform_keeps_percentiles(self):
        x = np.array([0., .1, .1, .4, 1.])
        q = np.array([0., .1, .3, 1.])
        np.testing.assert_array_equal(percentile(x, q), percentile(np.exp(x), np.exp(q)))

    def test_zero_error_binomial_bound(self):
        self.assertAlmostEqual(upper_cp(0, 20, .1), 1 - .1 ** (1/20))


if __name__ == "__main__":
    unittest.main()
