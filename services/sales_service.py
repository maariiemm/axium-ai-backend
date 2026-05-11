import json
import os
from datetime import date


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SALE_PERIODS_FILE = os.path.join(CONFIG_DIR, "sale_periods.json")


def load_sale_periods():
    with open(SALE_PERIODS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sale_features(target_date: date) -> dict:
    features = {
        "is_black_friday_period": 0,
        "is_christmas_period": 0,
        "is_boxing_week": 0,
        "is_back_to_school": 0,
    }

    for period in load_sale_periods():
        start = date.fromisoformat(period["start"])
        end = date.fromisoformat(period["end"])

        if start <= target_date <= end:
            period_type = period["type"]

            if period_type == "black_friday":
                features["is_black_friday_period"] = 1
            elif period_type == "christmas":
                features["is_christmas_period"] = 1
            elif period_type == "boxing_week":
                features["is_boxing_week"] = 1
            elif period_type == "back_to_school":
                features["is_back_to_school"] = 1

    return features