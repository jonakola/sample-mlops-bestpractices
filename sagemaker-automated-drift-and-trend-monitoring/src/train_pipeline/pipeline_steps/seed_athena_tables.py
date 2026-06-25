#!/usr/bin/env python3
"""
Seed Athena training_data and evaluation_data tables with deterministic 80/20 split.

This is a one-time operation that runs as the first step in the training pipeline.
If the tables are already populated, it skips the seeding (idempotent).

Usage:
    python -m src.train_pipeline.pipeline_steps.seed_athena_tables

    # Or as part of SageMaker Pipeline processing step
"""

import argparse
import logging
import sys
from pathlib import Path

import boto3

# Make src.config importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.config import (
    ATHENA_DATABASE,
    ATHENA_EVALUATION_TABLE,
    ATHENA_OUTPUT_S3,
    ATHENA_TRAINING_TABLE,
    AWS_DEFAULT_REGION,
    DATA_S3_BUCKET,
    DATA_S3_PREFIX,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# Column order from download_kaggle_dataset.py
ATHENA_COLUMN_ORDER = [
    'transaction_timestamp', 'transaction_hour', 'transaction_day_of_week',
    'transaction_amount', 'transaction_type_code', 'customer_age', 'customer_gender',
    'customer_tenure_months', 'account_age_days', 'distance_from_home_km',
    'distance_from_last_transaction_km', 'time_since_last_transaction_min',
    'online_transaction', 'international_transaction', 'high_risk_country',
    'merchant_category_code', 'merchant_reputation_score', 'chip_transaction',
    'pin_used', 'card_present', 'cvv_match', 'address_verification_match',
    'num_transactions_24h', 'num_transactions_7days',
    'avg_transaction_amount_30days', 'max_transaction_amount_30days',
    'velocity_score', 'recurring_transaction', 'previous_fraud_incidents',
    'credit_limit', 'available_credit_ratio',
]

CSV_COLUMN_ORDER = [
    'transaction_id', 'transaction_timestamp', 'transaction_hour',
    'transaction_day_of_week', 'transaction_amount', 'transaction_type_code',
    'customer_age', 'customer_gender', 'customer_tenure_months', 'account_age_days',
    'distance_from_home_km', 'distance_from_last_transaction_km',
    'time_since_last_transaction_min', 'online_transaction', 'international_transaction',
    'high_risk_country', 'merchant_category_code', 'merchant_reputation_score',
    'chip_transaction', 'pin_used', 'card_present', 'cvv_match',
    'address_verification_match', 'num_transactions_24h', 'num_transactions_7days',
    'avg_transaction_amount_30days', 'max_transaction_amount_30days',
    'velocity_score', 'recurring_transaction', 'previous_fraud_incidents',
    'credit_limit', 'available_credit_ratio', 'fraud_prediction', 'fraud_probability',
    'is_fraud',
]


def _run_athena_query(query: str, expect_results: bool = False) -> list:
    """Execute Athena query and optionally return results."""
    athena = boto3.client('athena', region_name=AWS_DEFAULT_REGION)

    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_S3},
    )
    query_id = response['QueryExecutionId']

    # Wait for completion
    import time
    while True:
        result = athena.get_query_execution(QueryExecutionId=query_id)
        state = result['QueryExecution']['Status']['State']
        if state in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
            break
        time.sleep(1)

    if state != 'SUCCEEDED':
        reason = result['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
        raise RuntimeError(f"Query failed: {reason}\nQuery: {query}")

    if expect_results:
        results = athena.get_query_results(QueryExecutionId=query_id)
        rows = results['ResultSet']['Rows'][1:]  # Skip header
        cols = [c['VarCharValue'] for c in results['ResultSet']['Rows'][0]['Data']]
        return [dict(zip(cols, [d.get('VarCharValue') for d in row['Data']])) for row in rows]

    return []


def check_tables_populated() -> dict:
    """Check if both tables exist and are populated."""
    try:
        train_count = _run_athena_query(
            f"SELECT COUNT(*) as cnt FROM {ATHENA_DATABASE}.{ATHENA_TRAINING_TABLE}",
            expect_results=True
        )[0]["cnt"]

        eval_count = _run_athena_query(
            f"SELECT COUNT(*) as cnt FROM {ATHENA_DATABASE}.{ATHENA_EVALUATION_TABLE}",
            expect_results=True
        )[0]["cnt"]

        return {
            'tables_exist': True,
            'training_count': int(train_count),
            'evaluation_count': int(eval_count),
            'populated': int(train_count) > 0 and int(eval_count) > 0
        }
    except Exception as e:
        logger.warning(f"Could not query tables: {e}")
        return {
            'tables_exist': False,
            'training_count': 0,
            'evaluation_count': 0,
            'populated': False
        }


def seed_tables() -> None:
    """Load data into training_data and evaluation_data with 80/20 split."""
    if not DATA_S3_BUCKET:
        raise RuntimeError("DATA_S3_BUCKET is empty — cannot seed Athena.")

    training_target = f"{ATHENA_DATABASE}.{ATHENA_TRAINING_TABLE}"
    evaluation_target = f"{ATHENA_DATABASE}.{ATHENA_EVALUATION_TABLE}"
    stage = f"{ATHENA_DATABASE}.tmp_seed_predictions"
    predictions_loc = f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}data/predictions/"

    logger.info("=" * 80)
    logger.info("SEEDING ATHENA TABLES WITH 80/20 TRAIN/EVALUATION SPLIT")
    logger.info("=" * 80)
    logger.info("Training table: %s", training_target)
    logger.info("Evaluation table: %s", evaluation_target)
    logger.info("Split method: Deterministic hash-based (MOD 10)")
    logger.info("Data source: %s", predictions_loc)
    logger.info("")

    # Verify tables exist (CloudFormation should have created them)
    try:
        _run_athena_query(f"DESCRIBE {training_target}")
        _run_athena_query(f"DESCRIBE {evaluation_target}")
        logger.info("✓ Tables exist (created by CloudFormation)")
    except Exception as e:
        raise RuntimeError(
            f"Tables do not exist. Deploy CloudFormation first:\n"
            f"  cd cloudformation && ./deploy-stack.sh\n"
            f"Error: {e}"
        )

    # Clear existing data
    logger.info("Clearing existing data from tables…")
    _run_athena_query(f"DELETE FROM {training_target}")
    _run_athena_query(f"DELETE FROM {evaluation_target}")
    _run_athena_query(f"DROP TABLE IF EXISTS {stage}")

    # Create staging table over CSV
    logger.info("Creating staging table over CSV predictions…")
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

    # Load training data (80%)
    logger.info("Loading training_data (80%% split)…")
    _run_athena_query(f"""
        INSERT INTO {training_target}
        SELECT transaction_id, {select_features},
               CAST(lower(fraud_prediction) AS BOOLEAN),
               CAST(fraud_probability AS DOUBLE),
               CAST(lower(is_fraud) AS BOOLEAN),
               'v1', current_timestamp, current_timestamp
        FROM {stage}
        WHERE MOD(ABS(xxhash64(transaction_id)), 10) < 8
    """)

    # Load evaluation data (20%)
    logger.info("Loading evaluation_data (20%% split)…")
    _run_athena_query(f"""
        INSERT INTO {evaluation_target}
        SELECT transaction_id, {select_features},
               CAST(lower(fraud_prediction) AS BOOLEAN),
               CAST(fraud_probability AS DOUBLE),
               CAST(lower(is_fraud) AS BOOLEAN),
               'v1', current_timestamp, current_timestamp
        FROM {stage}
        WHERE MOD(ABS(xxhash64(transaction_id)), 10) >= 8
    """)

    # Cleanup
    _run_athena_query(f"DROP TABLE IF EXISTS {stage}")

    # Verify
    train_count = _run_athena_query(
        f"SELECT COUNT(*) as cnt FROM {training_target}",
        expect_results=True
    )[0]["cnt"]
    eval_count = _run_athena_query(
        f"SELECT COUNT(*) as cnt FROM {evaluation_target}",
        expect_results=True
    )[0]["cnt"]
    total = int(train_count) + int(eval_count)

    logger.info("")
    logger.info("✓ Training data: %s rows (%.1f%%)", train_count, 100 * int(train_count) / total if total > 0 else 0)
    logger.info("✓ Evaluation data: %s rows (%.1f%%)", eval_count, 100 * int(eval_count) / total if total > 0 else 0)
    logger.info("✓ Total: %d rows", total)
    logger.info("")
    logger.info("⚠️  IMPORTANT: evaluation_data is now FROZEN for reproducible model comparison")
    logger.info("=" * 80)


def main():
    """Main entry point - idempotent seeding."""
    parser = argparse.ArgumentParser(
        description='Seed Athena tables with one-time 80/20 train/evaluation split'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-seeding even if tables are already populated'
    )
    args = parser.parse_args()

    logger.info("Checking if Athena tables are already populated…")
    status = check_tables_populated()

    if status['populated'] and not args.force:
        logger.info("=" * 80)
        logger.info("✓ TABLES ALREADY POPULATED - SKIPPING SEED")
        logger.info("=" * 80)
        logger.info("Training data: %d rows", status['training_count'])
        logger.info("Evaluation data: %d rows", status['evaluation_count'])
        logger.info("")
        logger.info("To re-seed, run with --force flag:")
        logger.info("  python -m src.train_pipeline.pipeline_steps.seed_athena_tables --force")
        logger.info("=" * 80)
        return

    if args.force and status['populated']:
        logger.warning("⚠️  --force flag detected: Re-seeding tables (existing data will be replaced)")

    # Seed the tables
    seed_tables()

    logger.info("✓ Seeding complete. Tables are ready for training pipeline.")


if __name__ == '__main__':
    main()
