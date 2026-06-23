#!/usr/bin/env python3
"""
Generate CSV datasets for the fraud detection pipeline.

Downloads the real Kaggle credit card fraud dataset (mlg-ulb/creditcardfraud)
and renames columns to match the project's business-friendly schema.

The drifted and ground truth datasets are synthetically generated
(they are used for monitoring/testing, not model training).

Produces:
  - creditcard_predictions_final.csv       (284,807 rows - from Kaggle)
  - creditcard_drifted.csv                 (5,000 rows - synthetic)
  - creditcard_ground_truth.csv            (50,000 rows - synthetic)

Usage:
    python data/generate_datasets.py                # generate all three
    python data/generate_datasets.py --predictions   # only predictions (Kaggle download)
    python data/generate_datasets.py --drifted       # only drifted (synthetic)
    python data/generate_datasets.py --ground-truth  # only ground truth (synthetic)

Requirements:
    pip install -e .  (installs kagglehub via pyproject.toml)
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
RANDOM_STATE = 42

N_DRIFTED = 5_000
N_GROUND_TRUTH = 50_000

FRAUD_RATE = 0.00173

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
# Generators
# ---------------------------------------------------------------------------

def generate_predictions(out_dir: Path) -> Path:
    """Download Kaggle dataset and transform to project schema."""
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

    path = out_dir / "creditcard_predictions_final.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def generate_drifted(out_dir: Path) -> Path:
    """Generate creditcard_drifted.csv with intentional feature drift."""
    rng = np.random.default_rng(RANDOM_STATE + 1)
    print(f"Generating drifted dataset ({N_DRIFTED:,} rows) ...")

    features = {col: rng.standard_normal(N_DRIFTED) for col in FEATURE_COLUMNS}
    is_fraud = rng.random(N_DRIFTED) < FRAUD_RATE
    fraud_prob = np.where(
        is_fraud,
        rng.uniform(0.5, 0.99, N_DRIFTED),
        rng.uniform(0.01, 0.25, N_DRIFTED),
    )

    df = pd.DataFrame({
        "transaction_id": np.arange(N_DRIFTED),
        "transaction_timestamp": np.round(rng.uniform(0, 172_800, N_DRIFTED), 1),
        **features,
        "transaction_amount": np.round(rng.lognormal(mean=3.5, sigma=1.5, size=N_DRIFTED), 2),
        "fraud_prediction": fraud_prob > 0.5,
        "fraud_probability": np.round(fraud_prob, 16),
        "customer_gender": rng.choice(GENDERS, size=N_DRIFTED, p=GENDER_WEIGHTS),
        "is_fraud": is_fraud,
    })

    # Apply drift
    drift_rng = np.random.default_rng(123)
    n = len(df)
    df["transaction_amount"] = np.round(df["transaction_amount"] * drift_rng.uniform(1.26, 1.54, n), 2)
    df["transaction_timestamp"] += drift_rng.uniform(45_000, 55_000, n)
    df["distance_from_home_km"] *= drift_rng.uniform(1.4, 2.6, n)
    df["velocity_score"] *= drift_rng.uniform(1.2, 1.8, n)
    df["num_transactions_24h"] = np.round(df["num_transactions_24h"] + drift_rng.uniform(2, 4, n))

    path = out_dir / "creditcard_drifted.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


def generate_ground_truth(out_dir: Path) -> Path:
    """Generate creditcard_ground_truth.csv (synthetic windowed ground truth)."""
    rng = np.random.default_rng(RANDOM_STATE + 2)
    print(f"Generating ground truth ({N_GROUND_TRUTH:,} rows) ...")

    num_windows = 10
    samples_per_window = N_GROUND_TRUTH // num_windows
    rows = []
    base_ts = datetime(2025, 11, 19, 16, 59, 39)

    for window_id in range(1, num_windows + 1):
        window_ts = base_ts + timedelta(hours=window_id)
        features = {col: rng.standard_normal(samples_per_window) for col in FEATURE_COLUMNS}

        is_fraud = rng.random(samples_per_window) < FRAUD_RATE
        fraud_prob = np.where(
            is_fraud,
            rng.uniform(0.5, 0.99, samples_per_window),
            rng.uniform(0.01, 0.25, samples_per_window),
        )
        observed = is_fraud.copy()
        flip_mask = rng.random(samples_per_window) < 0.05
        observed[flip_mask] = ~observed[flip_mask]

        window_df = pd.DataFrame({
            "transaction_id": [f"TXN_{window_id}_{i:05d}" for i in range(samples_per_window)],
            "prediction_timestamp": str(window_ts),
            "window_id": window_id,
            "transaction_timestamp": np.round(rng.uniform(50_000, 200_000, samples_per_window), 1),
            **features,
            "transaction_amount": np.round(rng.lognormal(mean=3.5, sigma=1.5, size=samples_per_window), 2),
            "ground_truth_fraud": is_fraud,
            "observed_fraud": observed,
            "fraud_probability": np.round(fraud_prob, 16),
        })
        rows.append(window_df)

    df = pd.concat(rows, ignore_index=True)
    path = out_dir / "creditcard_ground_truth.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV datasets for the fraud detection pipeline."
    )
    parser.add_argument("--predictions", action="store_true", help="Generate predictions CSV only")
    parser.add_argument("--drifted", action="store_true", help="Generate drifted CSV only")
    parser.add_argument("--ground-truth", action="store_true", help="Generate ground truth CSV only")
    args = parser.parse_args()

    out_dir = SCRIPT_DIR
    generate_all = not (args.predictions or args.drifted or args.ground_truth)

    if generate_all or args.predictions:
        generate_predictions(out_dir)
    if generate_all or args.drifted:
        generate_drifted(out_dir)
    if generate_all or args.ground_truth:
        generate_ground_truth(out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
