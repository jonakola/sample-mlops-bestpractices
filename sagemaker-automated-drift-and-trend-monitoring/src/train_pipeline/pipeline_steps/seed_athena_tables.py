#!/usr/bin/env python3
"""
Seed the Athena ``training_data`` and ``evaluation_data`` tables from the
predictions CSV in S3.

This runs as the first step of the SageMaker training pipeline. The two
tables are populated with a deterministic 80/20 split keyed on
``transaction_id`` so the evaluation slice is stable across model versions
— that stability is what makes ``evaluation_data`` a valid baseline for
the drift monitor.

Both tables are idempotent — if both pass the integrity check, the step
is a no-op.

The CloudFormation lifecycle creates the empty Iceberg tables; this step
fills them. The downstream preprocessing step reads training_data for
the train channel and evaluation_data for the validation/test channel.

Configuration is read from environment variables (set by the pipeline
ProcessingStep) so this script has no dependency on src.config — it must
run inside a vanilla SageMaker Processing container with only boto3.

Required env vars:
    AWS_DEFAULT_REGION
    ATHENA_DATABASE
    ATHENA_OUTPUT_S3        # s3://bucket/athena-results/
    ATHENA_TRAINING_TABLE
    ATHENA_EVALUATION_TABLE
    DATA_S3_BUCKET
    DATA_S3_PREFIX          # e.g. "fraud-detection/"

Usage (local debug):
    python -m src.train_pipeline.pipeline_steps.seed_athena_tables [--force]
"""

import argparse
import logging
import os
import time

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# NOTE: Column order is duplicated here because this script ships standalone
# into the SageMaker ScriptProcessor container — src/setup is not on
# PYTHONPATH there, so we cannot import. KEEP THESE TWO LISTS IN SYNC WITH
# src/setup/download_kaggle_dataset.py:
#   - CSV_COLUMN_ORDER   <-> download_kaggle_dataset.CSV_COLUMN_ORDER
#   - ATHENA_COLUMN_ORDER must also match the Iceberg DDL in
#     cloudformation/sagemaker-mlflow-setup.yaml.

# Order of columns the staging external table declares. MUST match exactly the
# header order in data/creditcard_predictions_final.csv that
# download_kaggle_dataset.py writes (column names bind to CSV positions in
# the order declared).
CSV_COLUMN_ORDER = [
    "transaction_id", "transaction_timestamp",
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
    "transaction_amount", "fraud_prediction", "fraud_probability",
    "customer_gender", "is_fraud",
]

# Order of columns the INSERT projects into the Iceberg target. MUST match the
# Iceberg DDL in cloudformation/sagemaker-mlflow-setup.yaml (excludes
# transaction_id and the trailing fraud_prediction/fraud_probability/is_fraud,
# which the seed script handles with explicit casts).
ATHENA_COLUMN_ORDER = [
    "transaction_timestamp", "transaction_hour", "transaction_day_of_week",
    "transaction_amount", "transaction_type_code", "customer_age", "customer_gender",
    "customer_tenure_months", "account_age_days", "distance_from_home_km",
    "distance_from_last_transaction_km", "time_since_last_transaction_min",
    "online_transaction", "international_transaction", "high_risk_country",
    "merchant_category_code", "merchant_reputation_score", "chip_transaction",
    "pin_used", "card_present", "cvv_match", "address_verification_match",
    "num_transactions_24h", "num_transactions_7days",
    "avg_transaction_amount_30days", "max_transaction_amount_30days",
    "velocity_score", "recurring_transaction", "previous_fraud_incidents",
    "credit_limit", "available_credit_ratio",
]

# V14 (renamed num_transactions_24h) has a fraud-class mean around -7 on the
# Kaggle dataset. If the table is correctly seeded the AVG over fraud rows is
# strongly negative. If it returns ~0 the table is corrupted — re-seed.
INTEGRITY_THRESHOLD = -3.0

# Deterministic hash-based 80/20 split on transaction_id. The same predicate
# is used for both inserts so the partitioning is reproducible: re-running
# the seed produces the same rows in each table.
# xxhash64 returns VARBINARY (8 bytes); convert to BIGINT with
# from_big_endian_64 before ABS/MOD or Athena errors with
# "Unexpected parameters (varbinary) for function abs".
TRAIN_PREDICATE = "MOD(ABS(from_big_endian_64(xxhash64(CAST(transaction_id AS VARBINARY)))), 10) < 8"
EVAL_PREDICATE = "MOD(ABS(from_big_endian_64(xxhash64(CAST(transaction_id AS VARBINARY)))), 10) >= 8"


class Config:
    def __init__(self):
        self.region = _require_env("AWS_DEFAULT_REGION")
        self.database = _require_env("ATHENA_DATABASE")
        self.output_s3 = _require_env("ATHENA_OUTPUT_S3")
        self.training_table = _require_env("ATHENA_TRAINING_TABLE")
        self.evaluation_table = _require_env("ATHENA_EVALUATION_TABLE")
        self.bucket = _require_env("DATA_S3_BUCKET")
        self.prefix = _require_env("DATA_S3_PREFIX")


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Required env var {name} is empty")
    return val


def _run_athena_query(cfg: Config, query: str, *, expect_results: bool = False,
                      timeout: int = 300) -> list:
    athena = boto3.client("athena", region_name=cfg.region)
    qid = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": cfg.database},
        ResultConfiguration={"OutputLocation": cfg.output_s3},
    )["QueryExecutionId"]

    deadline = time.time() + timeout
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state = status["State"]
        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"Athena query {state}: {status.get('StateChangeReason', 'unknown')}\n"
                f"SQL: {query[:300]}"
            )
        if time.time() > deadline:
            raise RuntimeError(f"Athena query timed out after {timeout}s")
        time.sleep(2)

    if not expect_results:
        return []

    rows, header = [], None
    for page in athena.get_paginator("get_query_results").paginate(QueryExecutionId=qid):
        for r in page["ResultSet"]["Rows"]:
            vals = [c.get("VarCharValue") for c in r["Data"]]
            if header is None:
                header = vals
            else:
                rows.append(dict(zip(header, vals)))
    return rows


def _check_one_table(cfg: Config, table: str) -> dict:
    """Returns ``{passed, fraud_count, non_fraud_count, fraud_v14_mean}`` for one table."""
    try:
        rows = _run_athena_query(
            cfg,
            f"SELECT is_fraud, COUNT(*) AS n, AVG(num_transactions_24h) AS v14_mean "
            f"FROM {cfg.database}.{table} GROUP BY is_fraud",
            expect_results=True,
        )
    except Exception as e:
        logger.info("Integrity check on %s skipped (not queryable yet): %s", table, e)
        return {"passed": False, "fraud_count": 0, "non_fraud_count": 0,
                "fraud_v14_mean": 0.0}

    by_class = {r["is_fraud"].lower(): r for r in rows if r.get("is_fraud")}
    f = by_class.get("true", {})
    n = by_class.get("false", {})
    fraud_mean = float(f.get("v14_mean") or 0)
    fraud_n = int(f.get("n") or 0)
    non_fraud_n = int(n.get("n") or 0)
    return {
        "passed": fraud_n > 0 and non_fraud_n > 0 and fraud_mean < INTEGRITY_THRESHOLD,
        "fraud_count": fraud_n,
        "non_fraud_count": non_fraud_n,
        "fraud_v14_mean": fraud_mean,
    }


def verify_integrity(cfg: Config) -> dict:
    """Run integrity checks on both tables. ``passed`` is True only if BOTH pass."""
    training = _check_one_table(cfg, cfg.training_table)
    evaluation = _check_one_table(cfg, cfg.evaluation_table)
    return {
        "passed": training["passed"] and evaluation["passed"],
        "training": training,
        "evaluation": evaluation,
    }


def seed_tables(cfg: Config) -> None:
    """Replace the contents of both tables from the predictions CSV with an 80/20 split."""
    training_target = f"{cfg.database}.{cfg.training_table}"
    evaluation_target = f"{cfg.database}.{cfg.evaluation_table}"
    stage = f"{cfg.database}.tmp_seed_predictions"
    predictions_loc = f"s3://{cfg.bucket}/{cfg.prefix}data/predictions/"

    logger.info("=" * 80)
    logger.info("SEEDING training_data (80%%) + evaluation_data (20%%) FROM %s", predictions_loc)
    logger.info("Split: deterministic on transaction_id (xxhash64 MOD 10)")
    logger.info("=" * 80)

    # Verify both target tables exist (CloudFormation should have created them).
    for target in (training_target, evaluation_target):
        try:
            _run_athena_query(cfg, f"DESCRIBE {target}")
            logger.info("✓ Target table exists: %s", target)
        except Exception as e:
            raise RuntimeError(
                f"Target table {target} does not exist. Deploy CloudFormation first:\n"
                f"  cd cloudformation && ./deploy-stack.sh\nError: {e}"
            )

    logger.info("Clearing existing rows…")
    _run_athena_query(cfg, f"DELETE FROM {training_target}")
    _run_athena_query(cfg, f"DELETE FROM {evaluation_target}")
    _run_athena_query(cfg, f"DROP TABLE IF EXISTS {stage}")

    logger.info("Creating staging table over CSV…")
    _run_athena_query(
        cfg,
        f"CREATE EXTERNAL TABLE {stage} ("
        + ", ".join(f"{c} STRING" for c in CSV_COLUMN_ORDER) + ") "
        "ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' "
        f"LOCATION '{predictions_loc}' "
        "TBLPROPERTIES ('skip.header.line.count'='1')"
    )

    def cast(c: str) -> str:
        return c if c == "customer_gender" else f"CAST({c} AS DOUBLE) AS {c}"

    select_features = ", ".join(cast(c) for c in ATHENA_COLUMN_ORDER)

    insert_template = (
        "INSERT INTO {target} "
        f"SELECT transaction_id, {select_features}, "
        "CAST(lower(fraud_prediction) AS BOOLEAN), "
        "CAST(fraud_probability AS DOUBLE), "
        "CAST(lower(is_fraud) AS BOOLEAN), "
        "'v1', current_timestamp, current_timestamp "
        f"FROM {stage} WHERE {{predicate}}"
    )

    logger.info("Loading training_data (80%% — train predicate)…")
    _run_athena_query(
        cfg,
        insert_template.format(target=training_target, predicate=TRAIN_PREDICATE),
    )

    logger.info("Loading evaluation_data (20%% — eval predicate)…")
    _run_athena_query(
        cfg,
        insert_template.format(target=evaluation_target, predicate=EVAL_PREDICATE),
    )

    _run_athena_query(cfg, f"DROP TABLE IF EXISTS {stage}")


def _log_check(label: str, c: dict) -> None:
    logger.info(
        "  %s: %d fraud / %d non-fraud, fraud_v14_mean=%.4f → %s",
        label, c["fraud_count"], c["non_fraud_count"], c["fraud_v14_mean"],
        "PASS" if c["passed"] else "FAIL",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Athena training_data + evaluation_data from predictions CSV"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-seed even if both integrity checks already pass"
    )
    args = parser.parse_args()

    cfg = Config()

    if not args.force:
        check = verify_integrity(cfg)
        if check["passed"]:
            logger.info("✓ Both tables healthy — skipping seed:")
            _log_check("training_data ", check["training"])
            _log_check("evaluation_data", check["evaluation"])
            return
        logger.info("Integrity check failed — re-seeding:")
        _log_check("training_data ", check["training"])
        _log_check("evaluation_data", check["evaluation"])

    seed_tables(cfg)

    final = verify_integrity(cfg)
    if not final["passed"]:
        _log_check("training_data ", final["training"])
        _log_check("evaluation_data", final["evaluation"])
        raise RuntimeError(
            "Seed completed but integrity check still failing. "
            "Inspect the predictions CSV in S3."
        )
    logger.info("✓ Seed complete:")
    _log_check("training_data ", final["training"])
    _log_check("evaluation_data", final["evaluation"])


if __name__ == "__main__":
    main()
