import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = r"C:\Users\MARIEM\Desktop\PFE2026\Génerateur de données\ai-training"

TEST_DIR = os.path.join(BASE_DIR, "DataTest")
PREDICT_DIR = os.path.join(BASE_DIR, "DataPredict")

PREDICTIONS_FILE = os.path.join(PREDICT_DIR, "prophet_predictions_period.csv")
REAL_COUNT_FILE = os.path.join(TEST_DIR, "prophet_count.csv")
REAL_AMOUNT_FILE = os.path.join(TEST_DIR, "prophet_amount.csv")

OUTPUT_FILE = os.path.join(PREDICT_DIR, "prophet_evaluation_results.csv")


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return 0.0

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def evaluate_transactions(pred_df):
    real_df = pd.read_csv(REAL_COUNT_FILE)

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    real_df["ds"] = pd.to_datetime(real_df["ds"])

    merged = pred_df.merge(
        real_df[["ds", "y"]],
        left_on="date",
        right_on="ds",
        how="inner"
    )

    y_true = merged["y"]
    y_pred = merged["transactions_prevues"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)

    coverage = (
        (
            (y_true >= merged["transactions_min"])
            &
            (y_true <= merged["transactions_max"])
        ).mean() * 100
    )

    merged["transactions_reelles"] = y_true
    merged["ecart_transactions"] = merged["transactions_reelles"] - merged["transactions_prevues"]
    merged["ecart_transactions_abs"] = merged["ecart_transactions"].abs()
    merged["ecart_transactions_pct"] = (
        merged["ecart_transactions_abs"] / merged["transactions_reelles"].clip(lower=1)
    ) * 100

    merged["alerte_transactions"] = np.where(
        (merged["transactions_reelles"] < merged["transactions_min"])
        | (merged["transactions_reelles"] > merged["transactions_max"]),
        "ECART_ANORMAL",
        "NORMAL"
    )

    return merged, {
        "MAE transactions": mae,
        "RMSE transactions": rmse,
        "MAPE transactions": mape,
        "Coverage transactions": coverage,
    }


def evaluate_amounts(pred_df):
    real_df = pd.read_csv(REAL_AMOUNT_FILE)

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    real_df["ds"] = pd.to_datetime(real_df["ds"])

    merged = pred_df.merge(
        real_df[["ds", "y"]],
        left_on="date",
        right_on="ds",
        how="inner"
    )

    y_true = merged["y"]
    y_pred = merged["montant_prevu_CAD"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)

    coverage = (
        (
            (y_true >= merged["montant_min_CAD"])
            &
            (y_true <= merged["montant_max_CAD"])
        ).mean() * 100
    )

    merged["montant_reel_CAD"] = y_true
    merged["ecart_montant_CAD"] = merged["montant_reel_CAD"] - merged["montant_prevu_CAD"]
    merged["ecart_montant_abs_CAD"] = merged["ecart_montant_CAD"].abs()
    merged["ecart_montant_pct"] = (
        merged["ecart_montant_abs_CAD"] / merged["montant_reel_CAD"].clip(lower=1)
    ) * 100

    merged["alerte_montant"] = np.where(
        (merged["montant_reel_CAD"] < merged["montant_min_CAD"])
        | (merged["montant_reel_CAD"] > merged["montant_max_CAD"]),
        "ECART_ANORMAL",
        "NORMAL"
    )

    return merged, {
        "MAE montant": mae,
        "RMSE montant": rmse,
        "MAPE montant": mape,
        "Coverage montant": coverage,
    }


def evaluate_prophet():
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(f"Fichier prédictions introuvable : {PREDICTIONS_FILE}")

    pred_df = pd.read_csv(PREDICTIONS_FILE)

    trans_eval, trans_metrics = evaluate_transactions(pred_df.copy())
    amount_eval, amount_metrics = evaluate_amounts(pred_df.copy())

    final_df = trans_eval.merge(
        amount_eval[[
            "date",
            "montant_reel_CAD",
            "ecart_montant_CAD",
            "ecart_montant_abs_CAD",
            "ecart_montant_pct",
            "alerte_montant"
        ]],
        on="date",
        how="inner"
    )

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nÉvaluation Prophet")
    print("═" * 80)

    print(f"Nombre de jours comparés : {len(final_df)}")

    print("\nVolume transactions")
    print(f"MAE transactions      : {trans_metrics['MAE transactions']:.2f}")
    print(f"RMSE transactions     : {trans_metrics['RMSE transactions']:.2f}")
    print(f"MAPE transactions     : {trans_metrics['MAPE transactions']:.2f}%")
    print(f"Coverage transactions : {trans_metrics['Coverage transactions']:.2f}%")

    print("\nMontant total")
    print(f"MAE montant           : {amount_metrics['MAE montant']:.2f} CAD")
    print(f"RMSE montant          : {amount_metrics['RMSE montant']:.2f} CAD")
    print(f"MAPE montant          : {amount_metrics['MAPE montant']:.2f}%")
    print(f"Coverage montant      : {amount_metrics['Coverage montant']:.2f}%")

    print(f"\nFichier sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_prophet()