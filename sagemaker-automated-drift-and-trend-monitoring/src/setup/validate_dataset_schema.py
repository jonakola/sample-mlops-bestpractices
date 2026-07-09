#!/usr/bin/env python3
"""
Validate that ``dataset_schema.yaml`` matches the actual source data
BEFORE launching any paid SageMaker job.

Checks:
    1. dataset_schema.yaml parses and has all required keys.
    2. The predictions CSV in S3 has a header matching schema.csv_column_order()
       exactly (same names, same order, same count).
    3. Spot-checks a sample of rows for type-compatibility (numeric columns
       parse as numbers, boolean columns are true/false-like).

Run this after editing dataset_schema.yaml and before running
create_athena_tables.py / the training pipeline.

Usage:
    python -m src.setup.validate_dataset_schema
    python -m src.setup.validate_dataset_schema --csv-s3-uri s3://my-bucket/my-data.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from pathlib import Path
from typing import List

import boto3

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import schema  # noqa: E402
from src.config.config import AWS_DEFAULT_REGION, DATA_S3_BUCKET, DATA_S3_PREFIX  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when dataset_schema.yaml does not match the source data."""


def _default_csv_s3_uri() -> str:
    prefix = DATA_S3_PREFIX.rstrip("/") if DATA_S3_PREFIX else "fraud-detection"
    return f"s3://{DATA_S3_BUCKET}/{prefix}/data/predictions/data.csv"


def _read_header_and_sample(csv_s3_uri: str, sample_rows: int = 20) -> tuple[List[str], List[List[str]]]:
    if not csv_s3_uri.startswith("s3://"):
        raise SchemaValidationError(f"Expected an s3:// URI, got: {csv_s3_uri}")
    _, _, rest = csv_s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")

    s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)
    try:
        # Range-GET just enough bytes to cover the header + a sample of
        # rows without downloading a potentially multi-GB file.
        resp = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-1048576")
    except s3.exceptions.NoSuchKey:
        raise SchemaValidationError(f"Source CSV not found: {csv_s3_uri}")
    body = resp["Body"].read().decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(body))
    rows = list(reader)
    if not rows:
        raise SchemaValidationError(f"Source CSV is empty: {csv_s3_uri}")
    header, data_rows = rows[0], rows[1 : 1 + sample_rows]
    return header, data_rows


def _validate_header(header: List[str]) -> List[str]:
    expected = schema.csv_column_order()
    errors = []

    if len(header) != len(expected):
        errors.append(
            f"Column count mismatch: source CSV has {len(header)} columns, "
            f"dataset_schema.yaml expects {len(expected)}."
        )

    missing = set(expected) - set(header)
    extra = set(header) - set(expected)
    if missing:
        errors.append(f"Columns declared in dataset_schema.yaml but missing from source CSV: {sorted(missing)}")
    if extra:
        errors.append(f"Columns present in source CSV but not declared in dataset_schema.yaml: {sorted(extra)}")

    if not missing and not extra and header != expected:
        errors.append(
            "Column names match but order differs. Expected order:\n"
            f"  {expected}\nSource CSV order:\n  {header}\n"
            "(Order matters — the staging table binds column names to "
            "CSV positions.)"
        )

    return errors


def _validate_sample_types(header: List[str], rows: List[List[str]]) -> List[str]:
    errors: List[str] = []
    if not rows:
        return errors

    col_index = {name: i for i, name in enumerate(header)}
    numeric_types = {"double", "int"}
    bool_like = {"true", "false", "1", "0", "yes", "no"}

    for feature in schema.features():
        idx = col_index.get(feature.name)
        if idx is None:
            continue  # already reported by _validate_header
        for row in rows:
            if idx >= len(row):
                continue
            value = row[idx].strip()
            if not value:
                continue
            if feature.type in numeric_types:
                try:
                    float(value)
                except ValueError:
                    errors.append(
                        f"Column '{feature.name}' declared type={feature.type} but "
                        f"found non-numeric value '{value}' in sample rows."
                    )
                    break
            elif feature.type == "boolean":
                if value.lower() not in bool_like:
                    errors.append(
                        f"Column '{feature.name}' declared type=boolean but found "
                        f"value '{value}' that doesn't look boolean in sample rows."
                    )
                    break
    return errors


def validate(csv_s3_uri: str = None, sample_rows: int = 20) -> None:
    """Raise SchemaValidationError with a full list of problems, or return
    normally if dataset_schema.yaml matches the source CSV."""
    csv_s3_uri = csv_s3_uri or _default_csv_s3_uri()
    logger.info("Validating dataset_schema.yaml against: %s", csv_s3_uri)

    header, rows = _read_header_and_sample(csv_s3_uri, sample_rows=sample_rows)

    errors = _validate_header(header)
    if not errors:
        errors += _validate_sample_types(header, rows)

    if errors:
        message = "\n".join(f"  - {e}" for e in errors)
        raise SchemaValidationError(
            f"dataset_schema.yaml does not match {csv_s3_uri}:\n{message}"
        )

    logger.info(
        "✓ dataset_schema.yaml matches source data: %d features, target='%s', id='%s'",
        len(schema.feature_names()), schema.target_column(), schema.identifier_column(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate dataset_schema.yaml against the actual source CSV"
    )
    parser.add_argument("--csv-s3-uri", default=None,
                        help="Override the source CSV location "
                             "(default: derived from DATA_S3_BUCKET/DATA_S3_PREFIX)")
    parser.add_argument("--sample-rows", type=int, default=20,
                        help="Number of data rows to type-check (default: 20)")
    args = parser.parse_args()

    try:
        validate(csv_s3_uri=args.csv_s3_uri, sample_rows=args.sample_rows)
    except SchemaValidationError as e:
        logger.error("Schema validation FAILED:\n%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
