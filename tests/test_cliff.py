import unittest
import numpy as np
import pandas as pd
from f1tyre import cliff


def stint(slope, knee=None, extra=0.0, laps=35, seed=0):
    rng = np.random.RandomState(seed)
    age = np.arange(1, laps + 1, dtype=float)
    y = slope * age + (extra * np.maximum(age - knee, 0.0) if knee else 0.0) + rng.normal(0, 0.05, laps)
    return age, y


def as_panel(stints):
    rows = []
    for i, (age, y) in enumerate(stints):
        for a, v in zip(age, y):
            rows.append({"race_id": "r0", "Driver": f"d{i}", "Stint": 1, "Compound": "HARD", "Year": 2024, "age": a, "y": v})
    return pd.DataFrame(rows)


class CliffTests(unittest.TestCase):
    def test_finds_the_knee_and_the_extra_slope(self):
        age, y = stint(0.03, knee=22, extra=0.25)
        found = cliff.best_break(age, y)
        self.assertAlmostEqual(found["knee"], 22, delta=2)
        self.assertAlmostEqual(found["extra_slope"], 0.25, delta=0.05)
        self.assertGreater(found["bic_gain"], cliff.MIN_BIC_GAIN)

    def test_a_straight_stint_is_not_flagged(self):
        table = cliff.find_cliffs(as_panel([stint(0.04, seed=s) for s in range(20)]))
        self.assertLessEqual(table["cliff"].sum(), 1)
        self.assertLessEqual(table["relief"].sum(), 1)

    def test_a_late_slowdown_is_a_cliff_and_a_late_recovery_is_not(self):
        table = cliff.find_cliffs(as_panel([stint(0.03, knee=22, extra=0.25), stint(0.12, knee=20, extra=-0.2, seed=1)]))
        self.assertEqual(table["cliff"].tolist(), [True, False])
        self.assertEqual(table["relief"].tolist(), [False, True])

    def test_short_stints_are_skipped_and_summary_counts_match(self):
        panel = as_panel([stint(0.03, knee=22, extra=0.25), stint(0.04, laps=12, seed=2)])
        table = cliff.find_cliffs(panel)
        self.assertEqual(len(table), 1)
        summary = cliff.summarise_cliffs(table)
        self.assertEqual(summary["ALL"]["cliff_stints"], 1)
        self.assertEqual(summary["HARD"]["stints"], 1)
        self.assertIsNone(summary["SOFT"]["median_onset_age"])


if __name__ == "__main__":
    unittest.main()
