import unittest
import numpy as np
import pandas as pd
from f1tyre import degradation


def synthetic_laps(slopes, circuit_effect=None, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for race in range(12):
        event = f"circuit_{race % 4}"
        for driver in range(8):
            for stint in (1, 2):
                compound = ["SOFT", "MEDIUM", "HARD"][(driver + stint + race) % 3]
                slope = slopes[compound] + (circuit_effect[event] if circuit_effect else 0.0)
                offset = rng.normal(0, 2.0)
                for age in range(1, 21):
                    rows.append({"race_id": f"r{race}", "Driver": f"d{driver}", "Stint": stint, "Year": 2024, "EventName": event, "Compound": compound, "TyreLife": age, "LapNumber": age + 20 * (stint - 1), "TotalLaps": 40, "TrackTemp": 30.0, "delta": offset + slope * age + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


class DegradationModelTests(unittest.TestCase):
    slopes = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}

    def test_recovers_compound_slopes_despite_large_driver_offsets(self):
        panel = degradation.build_panel(synthetic_laps(self.slopes), fuel_total_s=0.0)
        model = degradation.DegradationModel(temperature=False, curvature=False).fit(panel)
        for compound, slope in self.slopes.items():
            self.assertAlmostEqual(model.coefficients()[f"age[{compound}]"], slope, delta=0.003)

    def test_curve_is_ordered_by_compound_and_starts_at_zero(self):
        panel = degradation.build_panel(synthetic_laps(self.slopes), fuel_total_s=0.0)
        model = degradation.DegradationModel(temperature=False, curvature=False).fit(panel)
        ages = np.arange(1, 21)
        curves = {c: model.curve(c, pd.Series(dtype=float), 30.0, ages) for c in self.slopes}
        self.assertTrue(all(curve[0] == 0.0 for curve in curves.values()))
        self.assertTrue(curves["SOFT"][-1] > curves["MEDIUM"][-1] > curves["HARD"][-1])

    def test_circuit_modifier_is_recovered(self):
        circuits = pd.DataFrame({"severity": [-1.0, 0.0, 1.0, 2.0]}, index=[f"circuit_{i}" for i in range(4)])
        z = (circuits["severity"] - circuits["severity"].mean()) / circuits["severity"].std()
        effect = {event: 0.02 * z[event] for event in circuits.index}
        panel = degradation.build_panel(synthetic_laps(self.slopes, effect), fuel_total_s=0.0)
        model = degradation.DegradationModel(circuit_columns=["severity"], temperature=False, curvature=False, alpha=0.0).fit(panel, circuits)
        self.assertAlmostEqual(model.coefficients()["age x severity"], 0.02, delta=0.003)

    def test_hinge_shape_captures_a_kink_that_a_straight_line_misses(self):
        rng = np.random.RandomState(3)
        rows = []
        for stint in range(120):
            offset = rng.normal(0, 2.0)
            for age in range(1, 26):
                rows.append({"race_id": f"r{stint % 6}", "Driver": f"d{stint}", "Stint": 1, "Year": 2024, "EventName": "c", "Compound": "SOFT", "TyreLife": age, "LapNumber": age, "TotalLaps": 50, "TrackTemp": 30.0, "delta": offset + 0.06 * max(age - 8, 0) + rng.normal(0, 0.05)})
        panel = degradation.build_panel(pd.DataFrame(rows), fuel_total_s=0.0)
        ages = np.arange(1, 26)
        hinge = degradation.DegradationModel(temperature=False, curvature=False, knots=[8], shape_alpha=0.0).fit(panel).curve("SOFT", pd.Series(dtype=float), 30.0, ages)
        line = degradation.DegradationModel(temperature=False, curvature=False).fit(panel).curve("SOFT", pd.Series(dtype=float), 30.0, ages)
        truth = 0.06 * np.maximum(ages - 8, 0) - 0.06 * max(1 - 8, 0)
        self.assertLess(np.abs(hinge - truth).max(), 0.05)
        self.assertGreater(np.abs(line - truth).max(), 0.15)

    def test_curve_band_contains_the_curve(self):
        panel = degradation.build_panel(synthetic_laps(self.slopes), fuel_total_s=0.0)
        model = degradation.DegradationModel(temperature=False, curvature=False).fit(panel)
        draws = pd.DataFrame([model.coef_ * f for f in (0.9, 1.0, 1.1)], columns=model.names_)
        ages = np.arange(1, 21)
        low, high = model.curve_band("SOFT", pd.Series(dtype=float), 30.0, ages, draws)
        curve = model.curve("SOFT", pd.Series(dtype=float), 30.0, ages)
        self.assertTrue((low <= curve + 1e-9).all() and (curve <= high + 1e-9).all())

    def test_rank_descriptors_puts_the_driving_descriptor_first(self):
        circuits = pd.DataFrame({"noise": [0.3, -1.2, 0.9, 0.1], "severity": [-1.0, 0.0, 1.0, 2.0]}, index=[f"circuit_{i}" for i in range(4)])
        z = (circuits["severity"] - circuits["severity"].mean()) / circuits["severity"].std()
        panel = degradation.build_panel(synthetic_laps(self.slopes, {event: 0.02 * z[event] for event in circuits.index}), fuel_total_s=0.0)
        self.assertEqual(degradation.rank_descriptors(panel, circuits, ["noise", "severity"])[0], "severity")

    def test_held_out_stints_score_well(self):
        panel = degradation.build_panel(synthetic_laps(self.slopes, seed=1), fuel_total_s=0.0)
        train_mask = panel["race_id"].isin([f"r{i}" for i in range(9)])
        model = degradation.DegradationModel(temperature=False, curvature=False).fit(panel[train_mask].reset_index(drop=True))
        test = panel[~train_mask].reset_index(drop=True)
        scores = degradation.evaluate(test, model.predict_within(test))
        # noise-limited ceiling with these slopes and 0.1 s noise is about 0.91 (hard-tyre stints carry little signal)
        self.assertGreater(scores["within_r2"], 0.88)
        self.assertGreater(scores["slope_corr"], 0.9)


if __name__ == "__main__":
    unittest.main()
