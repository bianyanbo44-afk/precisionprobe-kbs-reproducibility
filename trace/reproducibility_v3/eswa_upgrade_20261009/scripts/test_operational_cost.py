from datetime import datetime, timezone
import unittest

from analyze_operational_cost import sleep_overlap


class SleepCostChecks(unittest.TestCase):
    def at(self, minute, second=0):
        return datetime(2026, 10, 9, 4, minute, second, tzinfo=timezone.utc)

    def test_overlapping_intervals_are_not_subtracted_twice(self):
        intervals = [(self.at(1), self.at(2)), (self.at(1, 30), self.at(3))]
        self.assertEqual(sleep_overlap(self.at(4).isoformat(), 240, intervals), 120)

    def test_only_request_overlap_is_removed(self):
        intervals = [(self.at(1), self.at(5))]
        self.assertEqual(sleep_overlap(self.at(2).isoformat(), 90, intervals), 60)
        self.assertEqual(sleep_overlap(self.at(8).isoformat(), 60, intervals), 0)


if __name__ == "__main__":
    unittest.main()
