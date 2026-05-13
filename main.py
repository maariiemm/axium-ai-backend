import os
import joblib
import pandas as pd

from datetime import date, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from services.fraud_detection_service import detect_fraud
from services.regressor_builder import build_regressors


# ═════════════════════════════════════
# PATHS
# ═════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

COUNT_MODEL_PATH = os.path.join(MODEL_DIR, "prophet_count_model.pkl")
AMOUNT_MODEL_PATH = os.path.join(MODEL_DIR, "prophet_amount_model.pkl")


# ═════════════════════════════════════
# REGRESSORS USED BY PROPHET
# IMPORTANT:
# Must match train_prophet.py exactly.
# ═════════════════════════════════════

REGRESSORS = [
    "is_weekend",
    "is_black_friday_period",
    "is_christmas_period",
    "is_boxing_week",
    "is_back_to_school",
    "is_winter",
    "is_summer",
    "is_extreme_cold",
    "is_snowstorm",
]


# ═════════════════════════════════════
# LOAD MODELS
# ═════════════════════════════════════

if not os.path.exists(COUNT_MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {COUNT_MODEL_PATH}")

if not os.path.exists(AMOUNT_MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {AMOUNT_MODEL_PATH}")

count_model = joblib.load(COUNT_MODEL_PATH)
amount_model = joblib.load(AMOUNT_MODEL_PATH)


# ═════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════

app = FastAPI(
    title="Axium AI Backend",
    description="Prediction backend using Prophet models, JSON configs and real weather API",
    version="1.0.0"
)


# ═════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═════════════════════════════════════

class ProphetPredictionRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30)
    latitude: float = Field(default=45.5017)     # Montreal default
    longitude: float = Field(default=-73.5673)   # Montreal default


class ProphetPredictionItem(BaseModel):
    date: str

    transactions_prevues: int
    transactions_min: int
    transactions_max: int

    montant_prevu_CAD: float
    montant_min_CAD: float
    montant_max_CAD: float

    is_weekend: int
    is_winter: int
    is_summer: int
    is_black_friday_period: int
    is_christmas_period: int
    is_boxing_week: int
    is_back_to_school: int
    is_extreme_cold: int
    is_snowstorm: int


class ProphetPredictionResponse(BaseModel):
    days: int
    latitude: float
    longitude: float
    predictions: list[ProphetPredictionItem]

# ═════════════════════════════════════
# FRAUD DETECTION MODELS
# ═════════════════════════════════════

class FraudDetectionRequest(BaseModel):
    
    nb_transactions: int = 0

    montant_moyen: float = 0
    montant_max: float = 0
    montant_min: float = 0
    montant_std: float = 0
    montant_total: float = 0

    nb_approved: int = 0
    nb_declined: int = 0
    nb_signals: int = 0
    nb_nonfatal: int = 0
    nb_cancelled: int = 0
    nb_failed_signal: int = 0
    nb_declined_signal: int = 0

    nb_error_52: int = 0
    nb_error_29: int = 0
    nb_error_56: int = 0
    nb_error_events: int = 0
    nb_error_10: int = 0
    nb_error_8: int = 0

    real_error_rate: float = 0
    timeout_rate: float = 0
    cancel_error_rate: float = 0
    card_mismatch_rate: float = 0
    retry_error_rate: float = 0
    magnetic_read_error_rate: float = 0

    approval_rate: float = 0
    taux_echec: float = 0
    taux_erreur_nonfatal: float = 0
    taux_annulation: float = 0
    failed_signal_rate: float = 0

    refund_rate: float = 0
    void_rate: float = 0
    preauth_rate: float = 0

    chip_rate: float = 0
    contactless_rate: float = 0
    swipe_rate: float = 0
    manual_rate: float = 0

    unique_cards: int = 0
    unique_cards_ratio: float = 0

    error_to_success_ratio: float = 0
    recovery_proxy: int = 0
    nonfatal_per_signal: float = 0
    is_low_volume: int = 0

    hour: int = 0
    day_of_week: int = 0
    is_weekend: int = 0
    month: int = 0

class FraudDetectionResponse(BaseModel):

    is_anomaly: bool

    anomaly_score: float

    risk_level: str

    reason: str

# ═════════════════════════════════════
# HELPERS
# ═════════════════════════════════════

def build_future_dataframe(
    days: int,
    latitude: float,
    longitude: float
) -> pd.DataFrame:
    """
    Builds the Prophet input dataframe:
    ds + required regressors.
    """

    today = date.today()
    rows = []

    for i in range(0, days + 1):
        target_date = today + timedelta(days=i)

        all_features = build_regressors(
            target_date=target_date,
            latitude=latitude,
            longitude=longitude
        )

        row = {
            "ds": pd.to_datetime(target_date)
        }

        for regressor in REGRESSORS:
            row[regressor] = int(all_features.get(regressor, 0))

        rows.append(row)

    future = pd.DataFrame(rows)

    missing_columns = [
        col for col in ["ds"] + REGRESSORS
        if col not in future.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Prophet regressors: {missing_columns}"
        )

    return future[["ds"] + REGRESSORS]


def make_predictions(future: pd.DataFrame) -> list[dict]:
    count_forecast = count_model.predict(future)
    amount_forecast = amount_model.predict(future)

    results = []

    for i in range(len(future)):
        count_yhat = max(0, int(round(count_forecast.iloc[i]["yhat"])))
        count_min = max(0, int(round(count_forecast.iloc[i]["yhat_lower"])))
        count_max = max(0, int(round(count_forecast.iloc[i]["yhat_upper"])))

        amount_yhat = max(0.0, round(float(amount_forecast.iloc[i]["yhat"]), 2))
        amount_min = max(0.0, round(float(amount_forecast.iloc[i]["yhat_lower"]), 2))
        amount_max = max(0.0, round(float(amount_forecast.iloc[i]["yhat_upper"]), 2))

        results.append({
            "date": str(future.iloc[i]["ds"].date()),

            "transactions_prevues": count_yhat,
            "transactions_min": count_min,
            "transactions_max": count_max,

            "montant_prevu_CAD": amount_yhat,
            "montant_min_CAD": amount_min,
            "montant_max_CAD": amount_max,

            "is_weekend": int(future.iloc[i]["is_weekend"]),
            "is_winter": int(future.iloc[i]["is_winter"]),
            "is_summer": int(future.iloc[i]["is_summer"]),
            "is_black_friday_period": int(future.iloc[i]["is_black_friday_period"]),
            "is_christmas_period": int(future.iloc[i]["is_christmas_period"]),
            "is_boxing_week": int(future.iloc[i]["is_boxing_week"]),
            "is_back_to_school": int(future.iloc[i]["is_back_to_school"]),
            "is_extreme_cold": int(future.iloc[i]["is_extreme_cold"]),
            "is_snowstorm": int(future.iloc[i]["is_snowstorm"]),
        })

    return results


# ═════════════════════════════════════
# ROUTES
# ═════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Axium AI Backend is running",
        "models_loaded": {
            "prophet_count_model": os.path.exists(COUNT_MODEL_PATH),
            "prophet_amount_model": os.path.exists(AMOUNT_MODEL_PATH),
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "Axium AI Backend",
        "models_dir": MODEL_DIR
    }


@app.post("/predict/prophet", response_model=ProphetPredictionResponse)
def predict_prophet(request: ProphetPredictionRequest):
    try:
        future = build_future_dataframe(
            days=request.days,
            latitude=request.latitude,
            longitude=request.longitude
        )

        predictions = make_predictions(future)

        return {
            "days": request.days,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
        


@app.post("/fraud/detect", response_model=FraudDetectionResponse)
def fraud_detection(request: FraudDetectionRequest):
    try:
        result = detect_fraud(request.dict())
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fraud detection failed: {str(e)}"
        )