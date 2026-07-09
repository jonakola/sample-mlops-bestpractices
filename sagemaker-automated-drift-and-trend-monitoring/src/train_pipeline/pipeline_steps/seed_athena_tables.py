#!/usr/bin/env python3
"""
Seed the Athena ``training_data`` and ``evaluation_data`` tables from the
predictions CSV in S3.

This runs as the first step of the SageMaker training pipeline. The two
tables are populated with a deterministic split (default 80/20, see
``dataset_schema.yaml``'s ``dataset.split`` section) keyed on the
configured identifier column so the evaluation slice is stable across
model versions — that stability is what makes ``evaluation_data`` a valid
baseline for the drift monitor.

Both tables are idempotent — if both pass the integrity check, the step
is a no-op.

The CloudFormation lifecycle creates the empty Iceberg tables; this step
fills them. The downstream preprocessing step reads training_data for
the train channel and evaluation_data for the validation/test channel.

Configuration is read from environment variables (set by the pipeline
ProcessingStep) so this script has no dependency on ``src.config.config``
— it must run inside a vanilla SageMaker Processing container.

It DOES depend on ``src.config.schema`` for the feature list — the
ProcessingStep's ``source_dir`` bundles ``src/config/schema.py`` and its
sibling ``dataset_schema.yaml`` alongside this script (see
``pipeline.py::_create_seed_athena_step``), so the import below resolves
inside the container without needing the rest of the ``src`` package.

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
import sys
import time
from pathlib import Path

import boto3

# When this script ships standalone (bundled via ProcessingStep source_dir),
# `src/config/schema.py` and `dataset_schema.yaml` land as siblings in the
# same directory. Add that directory to sys.path so `from src.config import
# schema` style absolute imports aren't required — import schema directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402  (bundled sibling module — see source_dir wiring)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical column order for the staging external table AND the source
# predictions CSV — both driven by dataset_schema.yaml via schema.py.
# There is exactly one order; see schema.csv_column_order()'s docstring.
CSV_COLUMN_ORDER = schema.csv_column_order()

# Feature columns (+ their declared types) the INSERT casts and projects
# into the Iceberg target, in the same order the Iceberg DDL declares them
# (both generated from schema.py — see src/setup/create_athena_tables.py).
FEATURES = schema.features()

ID_COLUMN = schema.identifier_column()
TARGET_COLUMN = schema.target_column()

# Generic corruption check: both classes must be non-empty. This deliberately
# does NOT rely on any dataset-specific statistical signature (e.g. a known
# feature's fraud-class mean) so it works for any dataset_schema.yaml, not
# just the Kaggle sample data.
MIN_ROWS_PER_CLASS = 1

# Deterministic hash-based split on the configured identifier column
# (schema.split_config()). The same predicate is used for both inserts so
# the partitioning is reproducible: re-running the seed produces the same
# rows in each table.
# xxhash64 returns VARBINARY (8 bytes); convert to BIGINT with
# from_big_endian_64 before ABS/MOD or Athena errors with
# "Unexpected parameters (varbinary) for function abs".
_split_cfg = schema.split_config()
_hash_col = _split_cfg.get("hash_column", ID_COLUMN)
_train_pct = int(round(_split_cfg.get("train_ratio", 0.8) * 10))
_HASH_EXPR = f"MOD(ABS(from_big_endian_64(xxhash64(CAST({_hash_col} AS VARBINARY)))), 10)"
TRAIN_PREDICATE = f"{_HASH_EXPR} < {_train_pct}"
EVAL_PREDICATE = f"{_HASH_EXPR} >= {_train_pct}"


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
    """Returns ``{passed, fraud_count, non_fraud_count}`` for one table.

    "Passed" means both classes are present with at least MIN_ROWS_PER_CLASS
    rows — a generic corruption check that works for any target column
    declared in dataset_schema.yaml, not a dataset-specific statistical
    signature.
    """
    try:
        rows = _run_athena_query(
            cfg,
            f"SELECT {TARGET_COLUMN}, COUNT(*) AS n "
            f"FROM {cfg.database}.{table} GROUP BY {TARGET_COLUMN}",
            expect_results=True,
        )
    except Exception as e:
        logger.info("Integrity check on %s skipped (not queryable yet): %s", table, e)
        return {"passed": False, "fraud_count": 0, "non_fraud_count": 0}

    by_class = {r[TARGET_COLUMN].lower(): r for r in rows if r.get(TARGET_COLUMN)}
    f = by_class.get("true", {})
    n = by_class.get("false", {})
    fraud_n = int(f.get("n") or 0)
    non_fraud_n = int(n.get("n") or 0)
    return {
        "passed": fraud_n >= MIN_ROWS_PER_CLASS and non_fraud_n >= MIN_ROWS_PER_CLASS,
        "fraud_count": fraud_n,
        "non_fraud_count": non_fraud_n,
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
    logger.info("SEEDING training_data (%d%%) + evaluation_data (%d%%) FROM %s",
                _train_pct * 10, 100 - _train_pct * 10, predictions_loc)
    logger.info("Split: deterministic on %s (xxhash64 MOD 10)", _hash_col)
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

    # Cast rule per feature/auxiliary/target column comes from its declared
    # type in dataset_schema.yaml (schema.cast_expr) — no hardcoded
    # per-column exception. A dataset with different auxiliary columns or
    # a differently-typed target works here without any code change.
    select_features = ", ".join(schema.cast_expr(f) for f in FEATURES)
    aux_columns = schema.auxiliary_columns()
    select_aux = ", ".join(schema.cast_expr(c) for c in aux_columns)
    select_aux_fragment = f"{select_aux}, " if select_aux else ""
    target_cast = schema.cast_expr(
        schema.Feature(TARGET_COLUMN, schema.target_type())
    )

    insert_template = (
        "INSERT INTO {target} "
        f"SELECT {ID_COLUMN}, {select_features}, "
        f"{select_aux_fragment}{target_cast}, "
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
        "  %s: %d positive-class / %d negative-class rows → %s",
        label, c["fraud_count"], c["non_fraud_count"],
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
