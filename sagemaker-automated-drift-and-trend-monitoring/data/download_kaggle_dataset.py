#!/usr/bin/env python3
"""
Download the Kaggle credit card fraud dataset and transform it to the
project's business-friendly schema for model training.

Downloads from mlg-ulb/creditcardfraud, renames V1-V28 columns to
domain-specific names, and adds metadata columns (transaction_id,
fraud_prediction, fraud_probability, customer_gender).

Produces:
  - creditcard_predictions_final.csv  (284,807 rows)

Usage:
    python data/download_kaggle_dataset.py

Requirements:
    pip install -e .  (installs kagglehub via pyproject.toml)
"""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
RANDOM_STATE = 42

KAGGLE_COLUMN_MAP = {
    "Time": "transaction_timestamp",
    "V1": "transaction_hour",
    "V2": "transaction_day_of_week",
    "V3": "customer_age",
    "V4": "account_age_days",
    "V5": "merchant_category_code",
    "V6": "distance_from_home_km",
    "V7": "distance_from_last_transaction_km",
    "V8": "online_transaction",
    "V9": "chip_transaction",
    "V10": "pin_used",
    "V11": "recurring_transaction",
    "V12": "international_transaction",
    "V13": "high_risk_country",
    "V14": "num_transactions_24h",
    "V15": "num_transactions_7days",
    "V16": "avg_transaction_amount_30days",
    "V17": "max_transaction_amount_30days",
    "V18": "card_present",
    "V19": "address_verification_match",
    "V20": "cvv_match",
    "V21": "velocity_score",
    "V22": "merchant_reputation_score",
    "V23": "time_since_last_transaction_min",
    "V24": "transaction_type_code",
    "V25": "customer_tenure_months",
    "V26": "credit_limit",
    "V27": "available_credit_ratio",
    "V28": "previous_fraud_incidents",
    "Amount": "transaction_amount",
    "Class": "is_fraud",
}

FEATURE_COLUMNS = [
    "transaction_hour",
    "transaction_day_of_week",
    "customer_age",
    "account_age_days",
    "merchant_category_code",
    "distance_from_home_km",
    "distance_from_last_transaction_km",
    "online_transaction",
    "chip_transaction",
    "pin_used",
    "recurring_transaction",
    "international_transaction",
    "high_risk_country",
    "num_transactions_24h",
    "num_transactions_7days",
    "avg_transaction_amount_30days",
    "max_transaction_amount_30days",
    "card_present",
    "address_verification_match",
    "cvv_match",
    "velocity_score",
    "merchant_reputation_score",
    "time_since_last_transaction_min",
    "transaction_type_code",
    "customer_tenure_months",
    "credit_limit",
    "available_credit_ratio",
    "previous_fraud_incidents",
]

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.45, 0.45, 0.10]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_STATE)

    print("Downloading Kaggle credit card fraud dataset (mlg-ulb/creditcardfraud)...")
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(dataset_path) / "creditcard.csv"

    print("Transforming to project schema...")
    df = pd.read_csv(csv_path)
    df = df.rename(columns=KAGGLE_COLUMN_MAP)

    n = len(df)
    is_fraud = df["is_fraud"].astype(bool).values

    fraud_prob = np.where(
        is_fraud,
        rng.uniform(0.5, 0.99, n),
        rng.uniform(0.01, 0.25, n),
    )

    df.insert(0, "transaction_id", np.arange(n))
    df["fraud_prediction"] = fraud_prob > 0.5
    df["fraud_probability"] = np.round(fraud_prob, 16)
    df["customer_gender"] = rng.choice(GENDERS, size=n, p=GENDER_WEIGHTS)

    col_order = [
        "transaction_id", "transaction_timestamp",
        *FEATURE_COLUMNS,
        "transaction_amount", "fraud_prediction", "fraud_probability",
        "customer_gender", "is_fraud",
    ]
    df = df[col_order]

    path = SCRIPT_DIR / "creditcard_predictions_final.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("\nDone.")


if __name__ == "__main__":
    main()
