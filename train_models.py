"""Fit the pre-lap and live gradient boosting models on all races and save them to models/ for the app.

Usage: python train_models.py      (about one minute)
"""
from f1tyre import data, features, lapmodel


def main():
    laps = features.build_lap_table(data.load_laps())
    lapmodel.train_and_save(laps)
    print(f"saved {', '.join(str(lapmodel.model_path(t).name) for t in lapmodel.SERVED_TIERS)} to {lapmodel.MODEL_DIR}")


if __name__ == "__main__":
    main()
