import os
import joblib
import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_isolation_forest.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_scaler.pkl"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# FEATURES
# =========================

FEATURE_COLUMNS = [
    "nb_transactions",
    "montant_moyen",
    "montant_max",
    "montant_std",
    "taux_echec",
    "taux_erreur_nonfatal",
    "taux_annulation",
    "hour",
    "day_of_week",
    "is_weekend"
]

# =========================
# PREDICTION
# =========================

def detect_fraud(features: dict):

    df = pd.DataFrame([features])

    X = df[FEATURE_COLUMNS]

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    anomaly_score = float(
        model.decision_function(X_scaled)[0]
    )

    is_anomaly = prediction == -1

    # =========================
    # RISK LEVEL
    # =========================

    if anomaly_score < -0.20:
        risk_level = "HIGH"

    elif anomaly_score < -0.05:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # =========================
    # REASONING
    # =========================

    reasons = []

    if features["taux_echec"] > 0.4:
        reasons.append("High failure rate")

    if features["taux_erreur_nonfatal"] > 0.3:
        reasons.append("High technical error rate")

    if features["nb_transactions"] > 40:
        reasons.append("Abnormally high transaction volume")

    if features["montant_max"] > 20000:
        reasons.append("Unusually high transaction amount")

    if features["hour"] in [1, 2, 3, 4, 5]:
        reasons.append("Night activity detected")

    reason = (
        ", ".join(reasons)
        if reasons
        else "Normal terminal behavior"
    )

    # =========================
    # RESPONSE
    # =========================

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "risk_level": risk_level,
        "reason": reason
    }
