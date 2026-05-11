import requests
from datetime import date


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather_features(
    target_date: date,
    latitude: float,
    longitude: float
) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_min,temperature_2m_max,snowfall_sum,precipitation_sum",
        "timezone": "auto",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat()
    }

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

    except requests.exceptions.RequestException:
        return {
            "is_extreme_cold": 0,
            "is_snowstorm": 0,
            "temperature_min": None,
            "temperature_max": None,
            "snowfall": None,
            "precipitation": None,
            "weather_source": "fallback"
        }

    data = response.json()
    daily = data.get("daily", {})

    if not daily or not daily.get("time"):
        return {
            "is_extreme_cold": 0,
            "is_snowstorm": 0
        }

    temp_min = daily["temperature_2m_min"][0]
    temp_max = daily["temperature_2m_max"][0]
    snowfall = daily["snowfall_sum"][0]
    precipitation = daily["precipitation_sum"][0]

    is_extreme_cold = 1 if temp_min <= -15 else 0

    is_snowstorm = 1 if (
        snowfall >= 5 or
        (temp_max <= 2 and precipitation >= 10)
    ) else 0

    return {
        "is_extreme_cold": is_extreme_cold,
        "is_snowstorm": is_snowstorm,
        "temperature_min": temp_min,
        "temperature_max": temp_max,
        "snowfall": snowfall,
        "precipitation": precipitation,
        "weather_source": "open_meteo"
    }