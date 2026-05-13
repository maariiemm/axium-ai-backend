import os
import joblib
import pandas as pd

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "fraud_scaler.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "fraud_feature_stats.pkl")

# =========================
# LOAD MODEL
# =========================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_stats = joblib.load(STATS_PATH)

# =========================
# FEATURES
# =========================

FEATURE_COLUMNS = [
    "nb_transactions",
    "montant_moyen",
    "montant_max",
    "montant_min",
    "montant_std",
    "montant_total",

    "nb_approved",
    "nb_declined",
    "nb_signals",
    "nb_nonfatal",
    "nb_cancelled",
    "nb_failed_signal",
    "nb_declined_signal",
    "nb_error_52",
    "nb_error_29",
    "nb_error_56",
    "nb_error_events",
    "nb_error_10",
    "nb_error_8",

    "real_error_rate",
    "timeout_rate",
    "cancel_error_rate",
    "card_mismatch_rate",
    "retry_error_rate",
    "magnetic_read_error_rate",

    "approval_rate",
    "taux_echec",
    "taux_erreur_nonfatal",
    "taux_annulation",
    "failed_signal_rate",

    "refund_rate",
    "void_rate",
    "preauth_rate",

    "chip_rate",
    "contactless_rate",
    "swipe_rate",
    "manual_rate",

    "unique_cards",
    "unique_cards_ratio",

    "error_to_success_ratio",
    "recovery_proxy",
    "nonfatal_per_signal",
    "is_low_volume",

    "hour",
    "day_of_week",
    "is_weekend",
    "month",
]

FEATURE_LABELS = {
    "nb_transactions": "transaction volume",
    "montant_moyen": "average transaction amount",
    "montant_max": "maximum transaction amount",
    "montant_min": "minimum transaction amount",
    "montant_std": "amount variability",
    "montant_total": "total hourly amount",

    "nb_approved": "approved transaction count",
    "nb_declined": "declined transaction count",
    "nb_signals": "technical signal count",
    "nb_nonfatal": "non-fatal technical error count",
    "nb_cancelled": "cancelled transaction count",
    "nb_failed_signal": "failed signal count",
    "nb_declined_signal": "declined signal count",
    "nb_error_52": "magnetic card read error count",
    "nb_error_29": "last 4 digits mismatch count",
    "nb_error_56": "timeout error count",
    "nb_error_events": "real error event count",
    "nb_error_10": "retry error count",
    "nb_error_8": "transaction cancellation error count",

    "real_error_rate": "real error rate",
    "timeout_rate": "timeout rate",
    "cancel_error_rate": "cancellation error rate",
    "card_mismatch_rate": "card mismatch rate",
    "retry_error_rate": "retry error rate",
    "magnetic_read_error_rate": "magnetic read error rate",

    "approval_rate": "approval rate",
    "taux_echec": "failure rate",
    "taux_erreur_nonfatal": "non-fatal technical error rate",
    "taux_annulation": "cancellation rate",
    "failed_signal_rate": "failed signal rate",

    "refund_rate": "refund rate",
    "void_rate": "void rate",
    "preauth_rate": "preauthorization rate",

    "chip_rate": "chip usage rate",
    "contactless_rate": "contactless usage rate",
    "swipe_rate": "swipe usage rate",
    "manual_rate": "manual entry rate",

    "unique_cards": "unique card count",
    "unique_cards_ratio": "unique card ratio",

    "error_to_success_ratio": "error-to-success ratio",
    "recovery_proxy": "successful recovery after error",
    "nonfatal_per_signal": "non-fatal error share",
    "is_low_volume": "low-volume window",

    "hour": "activity hour",
    "day_of_week": "day of week",
    "is_weekend": "weekend activity",
    "month": "month",
}

# =========================
# HELPERS
# =========================

def complete_missing_features(features: dict) -> dict:
    """
    Ensures the API can still work even if a few fields are missing.
    Missing values are replaced with 0.
    """

    completed = {}

    for col in FEATURE_COLUMNS:
        completed[col] = features.get(col, 0)

    return completed


def explain_anomaly(features: dict, top_k: int = 3) -> str:
    deviations = []

    for col in FEATURE_COLUMNS:
        stats = feature_stats.get(col)

        if not stats:
            continue

        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)

        if std == 0:
            continue

        value = float(features.get(col, 0))
        z_score = (value - mean) / std

        deviations.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "value": value,
            "mean": mean,
            "z_score": z_score
        })

    deviations = sorted(
        deviations,
        key=lambda item: abs(item["z_score"]),
        reverse=True
    )

    top_deviations = deviations[:top_k]

    if not top_deviations:
        return "Behavior differs from the learned terminal profile"

    explanations = []

    for deviation in top_deviations:
        direction = (
            "higher than usual"
            if deviation["z_score"] > 0
            else "lower than usual"
        )

        explanations.append(
            f"{deviation['label']} is {direction}"
        )

    return ", ".join(explanations)


def get_risk_level(anomaly_score: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "LOW"

    if anomaly_score < -0.15:
        return "HIGH"

    if anomaly_score < -0.05:
        return "MEDIUM"

    return "LOW"


# =========================
# PREDICTION
# =========================

def detect_fraud(features: dict):
    features = complete_missing_features(features)

    df = pd.DataFrame([features])
    X = df[FEATURE_COLUMNS]

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    anomaly_score = float(model.decision_function(X_scaled)[0])

    # Plus strict than prediction == -1 to avoid weak false positives
    is_anomaly = bool(
        prediction == -1 and anomaly_score < -0.03
    )

    risk_level = get_risk_level(
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly
    )

    if is_anomaly:
        reason = explain_anomaly(features)
    else:
        reason = "Normal terminal behavior"

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "risk_level": risk_level,
        "reason": reason
    }