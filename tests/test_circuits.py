import unittest
import numpy as np
import pandas as pd
from f1tyre import circuits


def circle_telemetry(radius_m, speed_kmh, n=2000):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    speed_ms = speed_kmh / 3.6
    return pd.DataFrame({"Distance": theta * radius_m, "Speed": speed_kmh, "Throttle": 100.0, "Brake": False, "X": radius_m * np.cos(theta) * 10, "Y": radius_m * np.sin(theta) * 10, "Z": 0.0, "Time": pd.to_timedelta(theta * radius_m / speed_ms, unit="s")})


class CircuitProfileTests(unittest.TestCase):
    def test_circle_matches_analytic_values(self):
        radius, speed = 200.0, 144.0
        tel = circle_telemetry(radius, speed)
        lap_time = 2 * np.pi * radius / (speed / 3.6)
        profile = circuits.profile_from_telemetry(tel, lap_time)
        expected_g = (speed / 3.6) ** 2 / radius / 9.81
        self.assertAlmostEqual(profile["length_km"], 2 * np.pi * radius / 1000, delta=0.01)
        self.assertAlmostEqual(profile["avg_speed_kmh"], speed, delta=0.5)
        self.assertAlmostEqual(profile["mean_lat_g"], expected_g, delta=0.1 * expected_g)
        self.assertAlmostEqual(profile["full_throttle_pct"], 100.0, delta=0.5)
        self.assertEqual(profile["brake_pct"], 0.0)
        self.assertEqual(profile["high_lat_pct"], 0.0)

    def test_tighter_circle_loads_the_tyres_more(self):
        loads = []
        for radius in (400.0, 200.0, 100.0):
            tel = circle_telemetry(radius, 144.0)
            loads.append(circuits.profile_from_telemetry(tel, 2 * np.pi * radius / 40.0)["mean_lat_g"])
        self.assertTrue(loads[0] < loads[1] < loads[2])

    def test_corner_and_braking_counts(self):
        distance = np.arange(0, 3000, 5.0)
        speed = 250.0 - 150.0 * np.exp(-((distance - 1000) / 60.0) ** 2) - 120.0 * np.exp(-((distance - 2200) / 60.0) ** 2)
        brake = (np.abs(distance - 940) < 60) | (np.abs(distance - 2140) < 60)
        theta = distance / 500.0
        tel = pd.DataFrame({"Distance": distance, "Speed": speed, "Throttle": np.where(brake, 0.0, 100.0), "Brake": brake, "X": 5000 * np.cos(theta), "Y": 5000 * np.sin(theta), "Z": 0.0, "Time": pd.to_timedelta(np.cumsum(5.0 / (speed / 3.6)), unit="s")})
        profile = circuits.profile_from_telemetry(tel, 60.0)
        self.assertEqual(profile["slow_corners"] + profile["medium_corners"] + profile["fast_corners"], 2)
        self.assertEqual(profile["heavy_braking_zones"], 2)


if __name__ == "__main__":
    unittest.main()
