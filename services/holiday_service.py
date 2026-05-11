import json
import os

from datetime import date


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CONFIG_DIR = os.path.join(
    BASE_DIR,
    "config"
)

HOLIDAYS_FILE = os.path.join(
    CONFIG_DIR,
    "JOURS_FERIES_FIXES.json"
)


def load_fixed_holidays():

    with open(
        HOLIDAYS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def is_fixed_holiday(
    target_date: date
) -> bool:

    holidays = load_fixed_holidays()

    for holiday in holidays:

        if (
            holiday["month"] == target_date.month
            and
            holiday["day"] == target_date.day
        ):

            return True

    return False