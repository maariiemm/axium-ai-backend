import os
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CONTEXT_MODEL_PATH = os.path.join(MODEL_DIR, "context_model.pkl")
BEHAVIOR_MODEL_PATH = os.path.join(MODEL_DIR, "behavior_model.pkl")


VOLUME_HORAIRE = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0,
    1, 4, 8, 12, 14, 12, 10, 12,
    26, 19, 10, 6, 3, 1,
    0
]


context_saved = joblib.load(CONTEXT_MODEL_PATH)
behavior_saved = joblib.load(BEHAVIOR_MODEL_PATH)

context_model = context_saved["model"]
context_features = context_saved["features"]

behavior_model = behavior_saved["model"]
behavior_features = behavior_saved["features"]


def safe_div(a, b):
    return a / b if b > 0 else 0


def volume_bucket_id(nb):
    if nb == 0:
        return 0
    elif nb <= 5:
        return 1
    elif nb <= 15:
        return 2
    elif nb <= 30:
        return 3
    elif nb <= 60:
        return 4
    elif nb <= 100:
        return 5
    return 6


def enrich_features(data: dict) -> dict:
    data = data.copy()

    hour = int(data.get("hour", 0))
    day = int(data.get("day_of_week", 0))
    nb_txn = int(data.get("nb_transactions", 0))
    nb_approved = int(data.get("nb_approved", 0))

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    data["day_sin"] = np.sin(2 * np.pi * day / 7)
    data["day_cos"] = np.cos(2 * np.pi * day / 7)

    data["is_night"] = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    data["is_business_hours"] = 1 if 9 <= hour <= 21 else 0

    expected_hour_volume = VOLUME_HORAIRE[hour]

    data["expected_hour_volume"] = expected_hour_volume
    data["volume_vs_expected_hour"] = safe_div(nb_txn, expected_hour_volume + 1)
    data["volume_bucket_id"] = volume_bucket_id(nb_txn)
    data["is_zero_activity_hour"] = 1 if expected_hour_volume == 0 else 0

    data["activity_in_zero_activity_hour"] = (
        1 if expected_hour_volume == 0 and nb_txn > 0 else 0
    )

    data["night_volume_ratio"] = nb_txn if data["is_night"] == 1 else 0

    data["approval_rate"] = safe_div(nb_approved, nb_txn)
    data["decline_rate"] = safe_div(data.get("nb_declined", 0), nb_txn)
    data["refund_rate"] = safe_div(data.get("nb_refund", 0), nb_txn)
    data["void_rate"] = safe_div(data.get("nb_void", 0), nb_txn)
    data["error_rate"] = safe_div(data.get("nb_nonfatal_error", 0), nb_txn)
    data["timeout_rate"] = safe_div(data.get("nb_timed_out", 0), nb_txn)
    data["cancelled_rate"] = safe_div(data.get("nb_cancelled", 0), nb_txn)

    data["error_to_success_ratio"] = safe_div(
        data.get("nb_nonfatal_error", 0)
        + data.get("nb_timed_out", 0)
        + data.get("nb_declined", 0),
        nb_approved
    )

    data["amount_per_approved_txn"] = safe_div(
        data.get("total_amount", 0),
        nb_approved
    )

    data["amount_per_transaction"] = safe_div(
        data.get("total_amount", 0),
        nb_txn
    )

    return data


def explain_context(data):
    reasons = []

    if data["activity_in_zero_activity_hour"] == 1:
        reasons.append("Transaction activity occurred during an hour that is usually inactive.")

    if data["volume_vs_expected_hour"] >= 3:
        reasons.append("Transaction volume is much higher than expected for this hour.")

    if data["is_night"] == 1 and data["nb_transactions"] > 0:
        reasons.append("Transaction activity detected during night hours.")

    if not reasons:
        reasons.append("Abnormal transaction context detected by the context model.")

    return reasons


def explain_behavior(data):
    reasons = []

    if data["decline_rate"] >= 0.30:
        reasons.append("Decline rate is high compared to normal transaction behavior.")

    if data["error_rate"] >= 0.15:
        reasons.append("Non-fatal error rate is high compared to normal behavior.")

    if data["timeout_rate"] >= 0.10:
        reasons.append("Timeout rate is high compared to normal behavior.")

    if data["cancelled_rate"] >= 0.10:
        reasons.append("Cancelled transaction rate is high compared to normal behavior.")

    if data["error_to_success_ratio"] >= 0.40:
        reasons.append("Error-to-success ratio is higher than expected.")

    if not reasons:
        reasons.append("Behavior model detected an abnormal combination of transaction ratios.")

    return reasons


def get_level(score: float) -> str:
    if score < -0.05:
        return "HIGH"
    if score < 0:
        return "MEDIUM"
    return "LOW"


def detect_fraud(features: dict):
    enriched = enrich_features(features)

    x_context = pd.DataFrame([enriched])
    x_context = x_context.reindex(columns=context_features, fill_value=0)

    context_prediction = context_model.predict(x_context)[0]
    context_score = float(context_model.decision_function(x_context)[0])

    if context_prediction == -1:
        return {
            "is_anomaly": True,
            "stage": "context",
            "anomaly_score": context_score,
            "risk_level": get_level(context_score),
            "reason": ", ".join(explain_context(enriched)),
        }

    x_behavior = pd.DataFrame([enriched])
    x_behavior = x_behavior.reindex(columns=behavior_features, fill_value=0)

    behavior_prediction = behavior_model.predict(x_behavior)[0]
    behavior_score = float(behavior_model.decision_function(x_behavior)[0])

    if behavior_prediction == -1:
        return {
            "is_anomaly": True,
            "stage": "behavior",
            "anomaly_score": behavior_score,
            "risk_level": get_level(behavior_score),
            "reason": ", ".join(explain_behavior(enriched)),
        }

    return {
        "is_anomaly": False,
        "stage": "normal",
        "anomaly_score": behavior_score,
        "risk_level": "LOW",
        "reason": "Normal terminal behavior",
    }