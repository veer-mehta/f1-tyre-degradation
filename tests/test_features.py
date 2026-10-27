import unittest
import numpy as np
import pandas as pd
from f1tyre import config, data, features


class FeatureTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            raw = data.load_laps()
        except ValueError:
            raise unittest.SkipTest("no data in data/")
        cls.raw = raw[(raw["Year"] == 2023) & (raw["Round"] <= 4)].reset_index(drop=True)
        cls.table = features.build_lap_table(cls.raw)

    def test_clean_laps_exclude_pit_laps_and_lap_one(self):
        cleaned = data.clean_laps(self.raw)
        self.assertFalse(cleaned["PitInTime"].any() or cleaned["PitOutTime"].any())
        self.assertGreater(cleaned["LapNumber"].min(), 1)

    def test_target_is_relative_to_field_median(self):
        medians = self.table.groupby("race_id")["delta"].median()
        np.testing.assert_allclose(medians.values, 0.0, atol=1e-9)

    def test_tier_features_are_present_and_finite(self):
        for column in config.TIERS["live"]:
            self.assertIn(column, self.table.columns)
        pre_lap_complete = [c for c in config.TIERS["pre_lap"] if c != "field_vs_cum"]
        self.assertFalse(self.table[pre_lap_complete].isna().any().any())
        self.assertTrue(np.isfinite(self.table[pre_lap_complete].values).all())

    def test_no_future_information_leaks_into_earlier_races(self):
        last_round = self.raw["Round"].max()
        changed = self.raw.copy()
        in_last = changed["Round"] == last_round
        noise = np.random.RandomState(0).uniform(0.9, 1.4, in_last.sum())
        changed.loc[in_last, "LapTime"] = changed.loc[in_last, "LapTime"].values * noise
        changed.loc[in_last, "TrackStatus"] = 4
        rebuilt = features.build_lap_table(changed)
        columns = ["race_id", "Driver", "LapNumber", "delta"] + config.TIERS["live"]
        key = ["race_id", "Driver", "LapNumber"]
        before = self.table[self.table["Round"] < last_round][columns].sort_values(key).reset_index(drop=True)
        after = rebuilt[rebuilt["Round"] < last_round][columns].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_stint_history_uses_only_earlier_stints(self):
        rows = []
        for stint, deltas in enumerate([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0], [100.0, 200.0, 300.0]], start=1):
            for lap, delta in enumerate(deltas, start=1):
                rows.append({"race_id": "r", "Driver": "d", "Stint": stint, "LapNumber": lap + 10 * stint, "delta": delta})
        laps = pd.DataFrame(rows)
        history = features.add_stint_history(laps)
        first, second, third = (history[history["Stint"] == s].iloc[0] for s in (1, 2, 3))
        self.assertTrue(np.isnan(first["prev_stint_mean_delta"]) and np.isnan(first["race_mean_delta_before"]))
        self.assertAlmostEqual(second["prev_stint_mean_delta"], 2.5)
        self.assertAlmostEqual(second["prev_stint_last3_delta"], 3.0)
        self.assertEqual(second["prev_stint_laps"], 4)
        self.assertAlmostEqual(third["prev_stint_mean_delta"], 15.0)
        self.assertAlmostEqual(third["race_mean_delta_before"], (1 + 2 + 3 + 4 + 10 + 20) / 6)
        changed = laps.copy()
        changed.loc[changed["Stint"] >= 2, "delta"] = changed.loc[changed["Stint"] >= 2, "delta"] * 7
        after = features.add_stint_history(changed)
        for column in config.PRE_STINT_FEATURES:
            pd.testing.assert_series_equal(history.loc[history["Stint"] == 2, column].reset_index(drop=True), after.loc[after["Stint"] == 2, column].reset_index(drop=True))
        self.assertNotAlmostEqual(history.loc[history["Stint"] == 3, "prev_stint_mean_delta"].iloc[0], after.loc[after["Stint"] == 3, "prev_stint_mean_delta"].iloc[0])

    def test_every_event_has_circuit_features(self):
        self.assertFalse(self.table[config.CIRCUIT_FEATURES].isna().any().any())
        self.assertGreater(self.table["climb_m_per_km"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
