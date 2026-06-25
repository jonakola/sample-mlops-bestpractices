#!/usr/bin/env python3
"""
Download the Kaggle credit-card fraud dataset, transform to the project's
business-friendly schema, upload to S3, seed the Athena training_data table,
and verify that fraud labels remain correlated with PCA features.

The CFN lifecycle script bootstraps the JupyterLab space but does NOT touch
training data — refresh is owned by this script (re-runnable from the notebook
or the command line), so a corrupted file in S3 never silently blocks training
again.

Usage:
    # Force a full refresh (download, upload, re-seed, verify)
    python -m src.setup.download_kaggle_dataset

    # From the notebook (recommended):
    from src.setup.download_kaggle_dataset import ensure_training_data_ready
    ensure_training_data_ready()        # idempotent — skips work if data is healthy
    ensure_training_data_ready(force=True)  # always re-seed

Requires:
    pip install -e .  (installs kagglehub, boto3 via pyproject.toml)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

# Make `src.config.config` importable when this file is run as a script.
# Layout: src/setup/download_kaggle_dataset.py — three parents to project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.config import (  # noqa: E402
    ATHENA_DATABASE,
    ATHENA_OUTPUT_S3,
    ATHENA_TRAINING_TABLE,
    AWS_DEFAULT_REGION,
    DATA_S3_BUCKET,
    DATA_S3_PREFIX,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)


RANDOM_STATE = 42
# Local CSV is written to the project's data/ scratch directory — the same
# location used by the drift-dataset generators (data/creditcard_drifted.csv,
# data/drifted_data_runN.csv) and read by notebook 2.
_DATA_DIR = _PROJECT_ROOT / "data"
LOCAL_CSV = _DATA_DIR / "creditcard_predictions_final.csv"

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.45, 0.45, 0.10]

# Kaggle V1..V28 are anonymized PCA components. We relabel them with
# business-friendly names; values are unchanged. The rename is purely cosmetic
# — V14 (renamed `num_transactions_24h`) keeps its real predictive power and
# we use it as our integrity-check probe.
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
    "transaction_hour", "transaction_day_of_week", "customer_age",
    "account_age_days", "merchant_category_code", "distance_from_home_km",
    "distance_from_last_transaction_km", "online_transaction",
    "chip_transaction", "pin_used", "recurring_transaction",
    "international_transaction", "high_risk_country", "num_transactions_24h",
    "num_transactions_7days", "avg_transaction_amount_30days",
    "max_transaction_amount_30days", "card_present",
    "address_verification_match", "cvv_match", "velocity_score",
    "merchant_reputation_score", "time_since_last_transaction_min",
    "transaction_type_code", "customer_tenure_months", "credit_limit",
    "available_credit_ratio", "previous_fraud_incidents",
]

# Order of columns in creditcard_predictions_final.csv after this script
# writes it. The S3 → Athena staging step uses this exact order to declare
# the staging table.
CSV_COLUMN_ORDER = (
    ["transaction_id", "transaction_timestamp"]
    + FEATURE_COLUMNS
    + ["transaction_amount", "fraud_prediction", "fraud_probability",
       "customer_gender", "is_fraud"]
)

# Athena `training_data` Iceberg target column order. Different from the CSV
# order — preserved for compatibility with the CFN-created table.
ATHENA_COLUMN_ORDER = [
    "transaction_timestamp", "transaction_hour", "transaction_day_of_week",
    "transaction_amount", "transaction_type_code", "customer_age",
    "customer_gender", "customer_tenure_months", "account_age_days",
    "distance_from_home_km", "distance_from_last_transaction_km",
    "time_since_last_transaction_min", "online_transaction",
    "international_transaction", "high_risk_country",
    "merchant_category_code", "merchant_reputation_score",
    "chip_transaction", "pin_used", "card_present", "cvv_match",
    "address_verification_match", "num_transactions_24h",
    "num_transactions_7days", "avg_transaction_amount_30days",
    "max_transaction_amount_30days", "velocity_score",
    "recurring_transaction", "previous_fraud_incidents",
    "credit_limit", "available_credit_ratio",
]


# ---------------------------------------------------------------------------
# Step 1 — Download + transform Kaggle data into the project's schema
# ---------------------------------------------------------------------------
def download_and_transform() -> Path:
    """Download Kaggle creditcardfraud, rename columns, write local CSV."""
    import kagglehub  # imported lazily so test environments don't need it

    logger.info("Downloading Kaggle credit-card fraud dataset (mlg-ulb/creditcardfraud)…")
    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(dataset_path) / "creditcard.csv"

    logger.info("Transforming to project schema…")
    df = pd.read_csv(csv_path)
    df = df.rename(columns=KAGGLE_COLUMN_MAP)

    rng = np.random.default_rng(RANDOM_STATE)
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

    df = df[CSV_COLUMN_ORDER]
    df.to_csv(LOCAL_CSV, index=False)
    logger.info("Wrote %s (%.1f MB, %d rows)", LOCAL_CSV, LOCAL_CSV.stat().st_size / 1024**2, n)
    return LOCAL_CSV


# ---------------------------------------------------------------------------
# Step 2 — Upload to S3 (both the canonical archive and the seeding location)
# ---------------------------------------------------------------------------
def upload_to_s3() -> None:
    """Upload the local CSV to s3://.../data/{predictions/data.csv,creditcard_predictions_final.csv}.

    The training pipeline reads from `predictions/data.csv`; the
    `creditcard_predictions_final.csv` copy is a human-readable archive.
    """
    if not DATA_S3_BUCKET:
        raise RuntimeError(
            "DATA_S3_BUCKET is empty — check src/config/config.yaml (project.name) "
            "and your AWS credentials."
        )
    if not LOCAL_CSV.exists():
        raise FileNotFoundError(f"{LOCAL_CSV} not found — run download_and_transform() first.")

    s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    for key in (
        f"{DATA_S3_PREFIX}data/predictions/data.csv",
        f"{DATA_S3_PREFIX}data/creditcard_predictions_final.csv",
    ):
        logger.info("Uploading to s3://%s/%s …", DATA_S3_BUCKET, key)
        s3.upload_file(str(LOCAL_CSV), DATA_S3_BUCKET, key)


# ---------------------------------------------------------------------------
# Step 3 — Re-seed Athena training_data
# ---------------------------------------------------------------------------
def _run_athena_query(sql: str, *, expect_results: bool = False, timeout: int = 300):
    athena = boto3.client("athena", region_name=AWS_DEFAULT_REGION)
    qid = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
    )["QueryExecutionId"]

    deadline = time.time() + timeout
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state = status["State"]
        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"Athena query {state}: {status.get('StateChangeReason', 'unknown')}\nSQL: {sql[:300]}"
            )
        if time.time() > deadline:
            raise RuntimeError(f"Athena query timed out after {timeout}s")
        time.sleep(2)

    if not expect_results:
        return qid

    rows, header = [], None
    for page in athena.get_paginator("get_query_results").paginate(QueryExecutionId=qid):
        for r in page["ResultSet"]["Rows"]:
            vals = [c.get("VarCharValue") for c in r["Data"]]
            if header is None:
                header = vals
            else:
                rows.append(dict(zip(header, vals)))
    return rows


def seed_athena_training_data() -> None:
    """Drop + recreate `training_data`, then bulk-INSERT from the predictions CSV.

    Idempotent — always replaces, so a re-run can never leave stale rows.
    """
    if not DATA_S3_BUCKET:
        raise RuntimeError("DATA_S3_BUCKET is empty — cannot seed Athena.")

    target = f"{ATHENA_DATABASE}.{ATHENA_TRAINING_TABLE}"
    stage = f"{ATHENA_DATABASE}.tmp_seed_predictions"
    target_loc = f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}{ATHENA_TRAINING_TABLE}/"
    predictions_loc = f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}data/predictions/"

    logger.info("Dropping existing %s (if any) and staging table…", target)
    _run_athena_query(f"DROP TABLE IF EXISTS {target}")
    _run_athena_query(f"DROP TABLE IF EXISTS {stage}")

    feat_cols = ", ".join(
        f"{c} STRING" if c == "customer_gender" else f"{c} DOUBLE"
        for c in ATHENA_COLUMN_ORDER
    )
    logger.info("Creating Iceberg target table %s…", target)
    _run_athena_query(f"""
        CREATE TABLE {target} (
            transaction_id STRING,
            {feat_cols},
            fraud_prediction BOOLEAN, fraud_probability DOUBLE, is_fraud BOOLEAN,
            data_version STRING, created_at TIMESTAMP, updated_at TIMESTAMP
        )
        LOCATION '{target_loc}'
        TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
    """)

    logger.info("Creating staging table %s over CSV…", stage)
    _run_athena_query(
        f"CREATE EXTERNAL TABLE {stage} ("
        + ", ".join(f"{c} STRING" for c in CSV_COLUMN_ORDER) + ") "
        "ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' "
        f"LOCATION '{predictions_loc}' "
        "TBLPROPERTIES ('skip.header.line.count'='1')"
    )

    def cast(c: str) -> str:
        return c if c == "customer_gender" else f"CAST({c} AS DOUBLE) AS {c}"

    select_features = ", ".join(cast(c) for c in ATHENA_COLUMN_ORDER)

    logger.info("Loading rows into %s…", target)
    _run_athena_query(f"""
        INSERT INTO {target}
        SELECT transaction_id, {select_features},
               CAST(lower(fraud_prediction) AS BOOLEAN),
               CAST(fraud_probability AS DOUBLE),
               CAST(lower(is_fraud) AS BOOLEAN),
               'v1', current_timestamp, current_timestamp
        FROM {stage}
    """)

    _run_athena_query(f"DROP TABLE IF EXISTS {stage}")


# ---------------------------------------------------------------------------
# Step 4 — Integrity check (catches the feature/label-desync bug)
# ---------------------------------------------------------------------------
def verify_integrity(*, max_fraud_v14_mean: float = -3.0) -> dict:
    """On the Kaggle dataset V14 (renamed `num_transactions_24h`) has a fraud-class
    mean around -7. If the table is correctly seeded, this query returns a
    strongly negative mean. If it returns ~0, the labels are uncorrelated with
    the features — the table is corrupted.
    """
    rows = _run_athena_query(
        f"SELECT is_fraud, COUNT(*) AS n, "
        f"AVG(num_transactions_24h) AS v14_mean "
        f"FROM {ATHENA_DATABASE}.{ATHENA_TRAINING_TABLE} "
        f"GROUP BY is_fraud",
        expect_results=True,
    )
    by_class = {r["is_fraud"].lower(): r for r in rows if r.get("is_fraud")}
    f = by_class.get("true", {})
    n = by_class.get("false", {})
    fraud_mean = float(f.get("v14_mean") or 0)
    return {
        "passed": fraud_mean < max_fraud_v14_mean,
        "fraud_count": int(f.get("n") or 0),
        "non_fraud_count": int(n.get("n") or 0),
        "fraud_v14_mean": fraud_mean,
        "non_fraud_v14_mean": float(n.get("v14_mean") or 0),
        "threshold": max_fraud_v14_mean,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ensure_training_data_ready(*, force: bool = False) -> dict:
    """Idempotent end-to-end. Skips if the Athena table is already healthy.

    Returns the final integrity-check result.
    """
    if not force:
        try:
            check = verify_integrity()
            if check["passed"]:
                logger.info(
                    "✓ training_data healthy — skipping refresh "
                    "(fraud_v14_mean=%.4f, %d fraud / %d non-fraud).",
                    check["fraud_v14_mean"],
                    check["fraud_count"], check["non_fraud_count"],
                )
                return check
            logger.warning(
                "⚠ training_data integrity check FAILED "
                "(fraud_v14_mean=%.4f, threshold < %.2f). Re-seeding.",
                check["fraud_v14_mean"], check["threshold"],
            )
        except Exception as e:
            logger.info("Table not queryable (%s); proceeding with full refresh.", e)

    if not LOCAL_CSV.exists():
        download_and_transform()
    upload_to_s3()
    seed_athena_training_data()

    final = verify_integrity()
    if not final["passed"]:
        raise RuntimeError(
            f"Re-seed completed but integrity check still failing "
            f"(fraud_v14_mean={final['fraud_v14_mean']:.4f}). "
            f"Inspect the Kaggle download and predictions CSV in S3."
        )
    logger.info(
        "✓ training_data ready — %d fraud / %d non-fraud, fraud_v14_mean=%.4f",
        final["fraud_count"], final["non_fraud_count"], final["fraud_v14_mean"],
    )
    return final


if __name__ == "__main__":
    ensure_training_data_ready(force=True)
