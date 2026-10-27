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

    def test_what_if_rows_cover_every_pre_lap_feature(self):
        event = self.table["EventName"].iloc[0]
        rows, total_laps = features.what_if_rows(self.table, event, "SOFT", 35.0, 25.0, start_lap=10)
        self.assertEqual(len(rows), 30)
        self.assertEqual(set(config.TIERS["pre_lap"]) - set(rows.columns), set())
        self.assertLessEqual(rows["race_fraction"].max(), 1.0)
        self.assertFalse(rows[config.TIERS["pre_lap"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
