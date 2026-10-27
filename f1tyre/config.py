"""Paths, constants, feature tiers and model hyperparameters."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEARS = [2025]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
SLOW_LAP_LIMIT = 1.04
DIRTY_AIR_GAP_S = 1.5
NEUTRAL_PACE_RATIO = 1.0  # lap time / field median for a car of exactly median pace (form fallback)
BASE_FEATURES = ["tyre_life", "tyre_life_sq", "tyre_life_log", "age_le2", "lap_number", "race_fraction", "stint", "track_temp", "air_temp", "humidity", "wind_speed", "temp_gap", "fresh_tyre", "compound_soft", "compound_medium", "compound_hard", "age_x_soft", "age_x_medium", "age_x_hard", "driver_form", "team_form", "driver_form_3", "driver_form_10", "team_form_3", "team_form_10"]
TRAFFIC_FEATURES = ["gap_ahead_prev", "gap_ahead_prev2", "gap_behind_prev", "position_prev", "dirty_air_prev"]
FIELD_FEATURES = ["field_vs_cum", "laps_since_neutral"]
CIRCUIT_FEATURES = ["climb_m_per_km"]  # selected from 13 telemetry descriptors by cross-validation on training years only (evaluate_degradation.py)
LIVE_FEATURES = ["prev_delta", "prev3_delta", "prev_lap_gap"]
PRE_STINT_FEATURES = ["race_mean_delta_before", "prev_stint_mean_delta", "prev_stint_last3_delta", "prev_stint_laps"]
PRE_LAP_FEATURES = BASE_FEATURES + TRAFFIC_FEATURES + FIELD_FEATURES + CIRCUIT_FEATURES
TIERS = {"pre_lap": PRE_LAP_FEATURES, "pre_stint": PRE_LAP_FEATURES + PRE_STINT_FEATURES, "live": PRE_LAP_FEATURES + PRE_STINT_FEATURES + LIVE_FEATURES}
GB_PARAMS = {"live": {"n_trees": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample": 0.8, "min_samples_leaf": 20, "reg_lambda": 5.0}, "pre_lap": {"n_trees": 150, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.5, "colsample": 0.6, "min_samples_leaf": 100, "reg_lambda": 5.0}}
FEATURE_GROUPS = {"tyre": ["tyre_life", "tyre_life_sq", "tyre_life_log", "age_le2", "fresh_tyre", "compound_soft", "compound_medium", "compound_hard", "age_x_soft", "age_x_medium", "age_x_hard", "stint"], "fuel_progress": ["lap_number", "race_fraction"], "weather": ["track_temp", "air_temp", "humidity", "wind_speed", "temp_gap"], "form": ["driver_form", "team_form", "driver_form_3", "driver_form_10", "team_form_3", "team_form_10"], "traffic": TRAFFIC_FEATURES, "field_context": FIELD_FEATURES, "circuit": CIRCUIT_FEATURES, "earlier_stints": PRE_STINT_FEATURES}
