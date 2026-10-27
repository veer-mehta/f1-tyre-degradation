import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from f1tyre import config, data, features, lapmodel, models


class LapModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            raw = data.load_laps()
        except ValueError:
            raise unittest.SkipTest("no data in data/")
        cls.raw = raw[(raw["Year"] == 2023) & (raw["Round"] <= 3)].reset_index(drop=True)
        cls.table = features.build_lap_table(cls.raw)
        cls.circuits = data.load_circuits()
        cls.defaults = lapmodel.reference_defaults(cls.table, cls.circuits)

    def stint_rows(self):
        counts = self.table.groupby(["race_id", "Driver", "Stint"]).size()
        race_id, driver, stint = counts[counts >= 15].index[0]
        stint_laps = self.table[(self.table["race_id"] == race_id) & (self.table["Driver"] == driver) & (self.table["Stint"] == stint)].reset_index(drop=True)
        rows = stint_laps[["LapNumber", "LapTime", "Compound", "TyreLife", "TrackTemp"]]
        settings = lapmodel.default_settings(self.defaults)
        settings.update({"median_s": float(stint_laps["field_med"].iloc[0]), "total_laps": float(stint_laps["TotalLaps"].iloc[0]), "stint": int(stint)})
        return stint_laps, rows, settings

    def test_entered_laps_rebuild_the_training_features(self):
        stint_laps, rows, settings = self.stint_rows()
        _, built = lapmodel.build_features(rows, settings, stint_laps["climb_m_per_km"].iloc[0], self.defaults)
        for column in ["tyre_life", "tyre_life_sq", "tyre_life_log", "age_le2", "race_fraction", "compound_soft", "compound_medium", "compound_hard", "age_x_soft", "age_x_medium", "age_x_hard", "climb_m_per_km", "prev_delta", "prev3_delta", "prev_lap_gap"]:
            np.testing.assert_allclose(built[column].to_numpy(dtype=float), stint_laps[column].to_numpy(dtype=float), rtol=1e-9, atol=1e-9, err_msg=column)

    def test_live_features_use_only_earlier_laps(self):
        _, rows, settings = self.stint_rows()
        _, built = lapmodel.build_features(rows, settings, 10.0, self.defaults)
        changed = rows.copy()
        changed.loc[changed.index[-1], "LapTime"] += 5.0
        _, rebuilt = lapmodel.build_features(changed, settings, 10.0, self.defaults)
        pd.testing.assert_frame_equal(built, rebuilt)

    def test_blank_lap_times_are_skipped_and_invalid_rows_dropped(self):
        rows = pd.DataFrame({"LapNumber": [10, 11, 12, 13], "LapTime": [95.0, None, 94.5, None], "Compound": ["medium", "MEDIUM", "wrong", "MEDIUM"], "TyreLife": [5, 6, 7, 8], "TrackTemp": [35, 35, 35, 35]})
        kept, built = lapmodel.build_features(rows, lapmodel.default_settings(self.defaults) | {"median_s": 93.0}, 10.0, self.defaults)
        self.assertEqual(kept["LapNumber"].tolist(), [10, 11, 13])
        self.assertEqual(built["prev_delta"].tolist()[1:], [2.0, 2.0])
        self.assertTrue(np.isnan(built["prev_delta"].iloc[0]))
        self.assertEqual(built["prev_lap_gap"].tolist()[1:], [1.0, 3.0])

    def test_saved_models_predict_the_same_after_loading(self):
        train = self.table[self.table["Round"] <= 2]
        original = models.fit_gb(train, config.TIERS["pre_lap"], "pre_lap")
        stint_laps, rows, settings = self.stint_rows()
        _, built = lapmodel.build_features(rows, settings, stint_laps["climb_m_per_km"].iloc[0], self.defaults)
        with tempfile.TemporaryDirectory() as folder:
            old_dir = lapmodel.MODEL_DIR
            lapmodel.MODEL_DIR = Path(folder)
            try:
                import pickle
                with open(lapmodel.model_path("pre_lap"), "wb") as f:
                    pickle.dump(original, f)
                with open(lapmodel.model_path("pre_lap"), "rb") as f:
                    loaded = pickle.load(f)
            finally:
                lapmodel.MODEL_DIR = old_dir
        columns = config.TIERS["pre_lap"]
        np.testing.assert_array_equal(original.predict(built[columns].to_numpy()), loaded.predict(built[columns].to_numpy()))


if __name__ == "__main__":
    unittest.main()
