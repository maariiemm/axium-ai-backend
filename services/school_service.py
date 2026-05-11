import json
import os
from datetime import date


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SCHOOL_BREAKS_FILE = os.path.join(CONFIG_DIR, "school_breaks.json")


def load_school_breaks():
    with open(SCHOOL_BREAKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_school_break(target_date: date) -> bool:
    for period in load_school_breaks():
        start = date.fromisoformat(period["start"])
        end = date.fromisoformat(period["end"])

        if start <= target_date <= end:
            return True

    return False