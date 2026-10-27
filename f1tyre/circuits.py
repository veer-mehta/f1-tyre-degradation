"""Circuit descriptors computed from one fast-lap telemetry trace (no FastF1 dependency)."""
import numpy as np
import pandas as pd


GRID_STEP_M = 5.0
G = 9.81
DESCRIPTOR_LABELS = {"length_km": "Lap length (km)", "avg_speed_kmh": "Average speed (km/h)", "top_speed_kmh": "Top speed (km/h)", "min_speed_kmh": "Slowest corner (km/h)", "full_throttle_pct": "Full throttle (% of lap)", "brake_pct": "Braking (% of lap)", "heavy_braking_zones": "Heavy braking zones", "mean_lat_g": "Mean cornering load (g)", "high_lat_pct": "High-load corners (% of lap)", "slow_corners": "Slow corners", "medium_corners": "Medium corners", "fast_corners": "Fast corners", "climb_m_per_km": "Elevation change (m/km)"}
PROFILE_COLUMNS = ["length_km", "avg_speed_kmh", "top_speed_kmh", "min_speed_kmh", "full_throttle_pct", "brake_pct", "heavy_braking_zones", "mean_lat_g", "high_lat_pct", "slow_corners", "medium_corners", "fast_corners", "climb_m_per_km"]


def moving_average(values, window):
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def local_minima(speed, half_window, prominence_window, min_prominence):
    n = len(speed)
    view = np.lib.stride_tricks.sliding_window_view(np.pad(speed, half_window, mode="edge"), 2 * half_window + 1)
    is_min = speed <= view.min(axis=1)
    wide = np.lib.stride_tricks.sliding_window_view(np.pad(speed, prominence_window, mode="edge"), 2 * prominence_window + 1)
    prominent = (wide.max(axis=1) - speed) >= min_prominence
    return np.flatnonzero(is_min & prominent & (np.arange(n) > 0) & (np.arange(n) < n - 1))


def braking_zones(brake_flag, speed, min_drop_kmh):
    zones = 0
    start = None
    for i, braking in enumerate(brake_flag):
        if braking and start is None:
            start = i
        if (not braking or i == len(brake_flag) - 1) and start is not None:
            if speed[start] - speed[start : i + 1].min() >= min_drop_kmh:
                zones += 1
            start = None
    return zones


def profile_from_telemetry(tel, lap_time_s):
    tel = tel.sort_values("Distance").drop_duplicates("Distance")
    distance = tel["Distance"].to_numpy(dtype=float)
    length_m = float(distance.max())
    time_s = tel["Time"].dt.total_seconds().to_numpy() if hasattr(tel["Time"], "dt") else tel["Time"].to_numpy(dtype=float)
    dt = np.diff(time_s, append=time_s[-1])
    total_time = dt.sum()
    grid = np.arange(0.0, length_m, GRID_STEP_M)
    speed = np.interp(grid, distance, tel["Speed"].to_numpy(dtype=float))
    brake = np.interp(grid, distance, tel["Brake"].to_numpy(dtype=float)) > 0.5
    x = moving_average(np.interp(grid, distance, tel["X"].to_numpy(dtype=float)) / 10.0, 5)
    y = moving_average(np.interp(grid, distance, tel["Y"].to_numpy(dtype=float)) / 10.0, 5)
    z = moving_average(np.interp(grid, distance, tel["Z"].to_numpy(dtype=float)) / 10.0, 9)
    heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    curvature = moving_average(np.gradient(heading) / GRID_STEP_M, 5)
    v = np.maximum(speed / 3.6, 1.0)
    lateral_g = v ** 2 * np.abs(curvature) / G
    weight = GRID_STEP_M / v
    weight = weight / weight.sum()
    smooth_speed = moving_average(speed, 5)
    corners = local_minima(smooth_speed, half_window=15, prominence_window=60, min_prominence=25.0)
    corner_speed = smooth_speed[corners]
    return {
        "length_km": length_m / 1000.0,
        "avg_speed_kmh": length_m / lap_time_s * 3.6,
        "top_speed_kmh": float(speed.max()),
        "min_speed_kmh": float(speed.min()),
        "full_throttle_pct": 100.0 * float(dt[tel["Throttle"].to_numpy(dtype=float) >= 98].sum() / total_time),
        "brake_pct": 100.0 * float(dt[tel["Brake"].to_numpy(dtype=float) > 0.5].sum() / total_time),
        "heavy_braking_zones": braking_zones(brake, speed, 100.0),
        "mean_lat_g": float((weight * lateral_g).sum()),
        "high_lat_pct": 100.0 * float(weight[lateral_g > 2.5].sum()),
        "slow_corners": int((corner_speed < 120).sum()),
        "medium_corners": int(((corner_speed >= 120) & (corner_speed < 200)).sum()),
        "fast_corners": int((corner_speed >= 200).sum()),
        "climb_m_per_km": float(np.abs(np.diff(z)).sum() / (length_m / 1000.0)),
    }
