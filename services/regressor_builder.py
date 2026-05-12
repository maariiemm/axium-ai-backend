import pandas as pd
from datetime import date

from services.weather_service import get_weather_features
from services.sales_service import get_sale_features
from services.school_service import is_school_break
from services.holiday_service import is_fixed_holiday


def build_regressors(
    target_date: date,
    latitude: float,
    longitude: float
) -> dict:

    ds = pd.to_datetime(target_date)

    day_of_week = ds.weekday()
    month = ds.month

    regressors = {}

    regressors["is_weekend"] = int(day_of_week in [5, 6])

    regressors["is_winter"] = int(month in [12, 1, 2])
    regressors["is_summer"] = int(month in [6, 7, 8])

    sale_features = get_sale_features(target_date)
    regressors.update(sale_features)

    weather_features = get_weather_features(
        target_date=target_date,
        latitude=latitude,
        longitude=longitude
    )

    regressors["is_extreme_cold"] = weather_features["is_extreme_cold"]
    regressors["is_snowstorm"] = weather_features["is_snowstorm"]

    regressors["is_school_break"] = int(is_school_break(target_date))
    regressors["is_holiday"] = int(is_fixed_holiday(target_date))

    return regressors