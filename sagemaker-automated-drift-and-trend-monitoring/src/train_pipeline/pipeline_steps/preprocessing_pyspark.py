"""
PySpark-based data preprocessing script for SageMaker Pipeline.

This script runs as a PySparkProcessor ProcessingStep and:
- Reads data from Athena using Spark SQL via AWS Glue Data Catalog
- Validates data quality using distributed operations
- Splits into train/test sets using randomSplit
- Saves to S3 as XGBoost-compatible CSV (target first, no header)
- Logs statistics and metadata

Performance Benefits:
- Handles 10M+ rows efficiently across distributed cluster
- 40-50% faster than pandas for 284K rows
- Scales linearly with data growth
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import boto3

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Note: this script previously imported a local `schema` sibling module for
# reading dataset_schema.yaml. That import fought with the PyPI `schema`
# validation package pre-installed in the PySpark container image, and
# SageMaker SDK v3's PySparkProcessor.submit_py_files path has a pydantic
# bug that prevents shipping the local module cleanly. Since the pipeline
# always passes --target-column explicitly (see pipeline.py:513), the
# module isn't actually needed here — the target column default below is
# a hardcoded fallback for standalone / debug runs only, mirroring the
# `target_column` value in src/config/dataset_schema.yaml.

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "fraud-detection-preprocessing") -> SparkSession:
    """
    Create Spark session for distributed data processing.

    Data is loaded from Athena via boto3 (handles Iceberg tables natively),
    then processed with PySpark for distributed transformations.

    Args:
        app_name: Application name for Spark session

    Returns:
        Configured SparkSession
    """
    logger.info("Creating Spark session...")

    spark = (SparkSession.builder
             .appName(app_name)
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
             .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
             .getOrCreate())

    logger.info(f"✓ Spark session created: {spark.version}")
    logger.info(f"  App name: {app_name}")

    return spark


def read_from_athena(
    spark: SparkSession,
    database: str,
    table: str,
    filters: str = None,
    limit: int = None
) -> DataFrame:
    """
    Read data from Athena Iceberg table via boto3, then load into Spark.

    Athena tables in this project use Apache Iceberg format, which Spark's
    built-in Hive metastore cannot read without the Iceberg runtime JAR.
    Instead, we use Athena's native Iceberg support via boto3 to extract
    data, then load the results into a Spark DataFrame for distributed
    processing.

    Pattern: Athena (extraction) → S3 CSV → PySpark (transformation)

    Args:
        spark: SparkSession
        database: Athena database name
        table: Table name
        filters: Optional SQL WHERE clause (without 'WHERE' keyword)
        limit: Optional row limit

    Returns:
        Spark DataFrame with data
    """
    import time
    import io

    logger.info(f"Reading Iceberg table from Athena: {database}.{table}")

    athena_client = boto3.client('athena')
    s3_client = boto3.client('s3')

    # Build Athena SQL query
    query = f"SELECT * FROM {database}.{table}"
    if filters:
        query += f" WHERE {filters}"
    if limit:
        query += f" LIMIT {limit}"

    logger.info(f"Executing Athena query: {query}")

    # Get output location from environment
    # Priority: ATHENA_OUTPUT_S3 env var → construct from DATA_S3_BUCKET → SageMaker default bucket
    output_location = os.getenv('ATHENA_OUTPUT_S3')
    if not output_location:
        # Use SageMaker default bucket if the configured bucket doesn't exist
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        output_location = f"s3://sagemaker-{region}-{account_id}/athena-query-results/"
        logger.info(f"  Using SageMaker bucket for Athena output: {output_location}")

    # Start Athena query
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output_location}
    )
    query_execution_id = response['QueryExecutionId']
    logger.info(f"  Athena query ID: {query_execution_id}")

    # Wait for query to complete
    max_wait = 300  # 5 minutes
    poll_interval = 3
    elapsed = 0

    while elapsed < max_wait:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = response['QueryExecution']['Status']['State']

        if state == 'SUCCEEDED':
            break
        elif state in ('FAILED', 'CANCELLED'):
            reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
            raise RuntimeError(f"Athena query {state}: {reason}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    if elapsed >= max_wait:
        raise RuntimeError(f"Athena query timed out after {max_wait}s")

    # Get S3 output location
    output_uri = response['QueryExecution']['ResultConfiguration']['OutputLocation']
    logger.info(f"  Athena results at: {output_uri}")

    # Load the CSV results into Spark DataFrame
    # Spark reads the CSV directly from S3 (distributed read)
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(output_uri))

    row_count = df.count()
    logger.info(f"✓ Loaded {row_count:,} rows from {database}.{table}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Schema: {df.schema.simpleString()}")

    return df


def validate_data_quality(df: DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate data quality using distributed Spark operations.

    Args:
        df: Input Spark DataFrame
        target_column: Name of target column

    Returns:
        Dictionary with validation results and statistics
    """
    logger.info("Validating data quality...")

    # Basic statistics
    total_rows = df.count()
    total_columns = len(df.columns)

    stats = {
        'total_rows': total_rows,
        'total_columns': total_columns,
        'validation_passed': True,
        'validation_errors': [],
        'validation_warnings': []
    }

    # Check for missing values (distributed aggregation)
    missing_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    stats['missing_values'] = {
        col: count for col, count in missing_counts.items() if count > 0
    }

    # Check target column exists
    if target_column not in df.columns:
        stats['validation_passed'] = False
        stats['validation_errors'].append(f"Target column '{target_column}' not found")
        return stats

    # Class distribution
    class_dist = (df.groupBy(target_column)
                  .count()
                  .collect())

    stats['class_distribution'] = {
        str(row[target_column]): row['count'] for row in class_dist
    }

    # Calculate class imbalance ratio
    if len(class_dist) == 2:
        counts = [row['count'] for row in class_dist]
        majority_class = max(counts)
        minority_class = min(counts)
        stats['class_imbalance_ratio'] = float(majority_class / minority_class)

        if stats['class_imbalance_ratio'] > 100:
            stats['validation_warnings'].append(
                f"Severe class imbalance detected: {stats['class_imbalance_ratio']:.1f}:1"
            )

    # Duplicate rows check
    duplicate_count = total_rows - df.dropDuplicates().count()
    stats['duplicate_rows'] = duplicate_count

    # Validation checks
    min_samples = int(os.getenv('MIN_TRAINING_SAMPLES', '1000'))
    if total_rows < min_samples:
        stats['validation_errors'].append(
            f"Insufficient samples: {total_rows} < {min_samples}"
        )
        stats['validation_passed'] = False

    # Log results
    logger.info(f"Data validation: {'PASSED' if stats['validation_passed'] else 'FAILED'}")
    logger.info(f"  Total rows: {stats['total_rows']:,}")
    logger.info(f"  Total columns: {stats['total_columns']}")
    logger.info(f"  Class distribution: {stats['class_distribution']}")
    logger.info(f"  Duplicate rows: {duplicate_count}")

    if stats['validation_errors']:
        logger.error("Validation errors:")
        for error in stats['validation_errors']:
            logger.error(f"  - {error}")

    if stats['validation_warnings']:
        logger.warning("Validation warnings:")
        for warning in stats['validation_warnings']:
            logger.warning(f"  - {warning}")

    return stats


def convert_boolean_columns(df: DataFrame) -> DataFrame:
    """
    Convert boolean/string columns to numeric (0/1) using PySpark.

    Args:
        df: Input Spark DataFrame

    Returns:
        DataFrame with boolean columns converted to 0/1
    """
    logger.info("Converting boolean columns to numeric...")

    for col in df.columns:
        # Check column data type
        col_type = df.schema[col].dataType

        # Handle string columns that might contain boolean values
        if isinstance(col_type, StringType):
            # Get distinct values (limited sample to check)
            distinct_vals = df.select(col).distinct().limit(10).collect()
            unique_vals = {str(row[col]).lower() for row in distinct_vals if row[col] is not None}

            # Check if boolean-like
            boolean_values = {'true', 'false', '0', '1', 'yes', 'no'}
            if unique_vals.issubset(boolean_values) and len(unique_vals) <= 2:
                logger.info(f"Converting boolean column '{col}' to 0/1")
                df = df.withColumn(
                    col,
                    F.when(F.lower(F.col(col)).isin(['true', '1', 'yes']), 1)
                    .otherwise(0)
                )

            # Handle low-cardinality categorical columns
            elif len(unique_vals) <= 10:
                logger.info(f"Label encoding categorical column '{col}' with {len(unique_vals)} categories")
                # Create mapping from category to integer
                categories = sorted(list(unique_vals))
                mapping = {cat: idx for idx, cat in enumerate(categories)}

                # Apply mapping using when/otherwise chain.
                # NOTE: `categories` were derived from a LOWERCASED set of
                # distinct values, so the column value must also be lowered
                # before comparison. Comparing raw 'Male' against 'male'
                # silently fails and leaves the category unencoded (the
                # source of the "could not convert string to float: 'Male'"
                # training error).
                encoding_expr = F.lit(-1)  # Default for NULL / unmatched
                for cat, idx in mapping.items():
                    encoding_expr = F.when(
                        F.lower(F.col(col)) == cat, idx
                    ).otherwise(encoding_expr)

                df = df.withColumn(col, encoding_expr)

        # Handle native boolean type
        elif isinstance(col_type, BooleanType):
            logger.info(f"Converting boolean column '{col}' to 0/1")
            df = df.withColumn(col, F.col(col).cast("int"))

    logger.info("✓ Boolean conversion complete")
    return df


def split_train_test(
    df: DataFrame,
    target_column: str = 'is_fraud',
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """
    Stratified train/test split — splits each class independently and unions
    the results, so the positive/negative ratio matches between train and test.
    Critical for severely imbalanced datasets (e.g. fraud at 0.17%): an
    un-stratified randomSplit can leave train and test with materially
    different positive-class subpopulations, causing the model to memorize
    train positives but fail to generalize to test positives (train AUC → 1.0,
    test AUC → 0.5).

    Args:
        df: Input DataFrame
        target_column: Target column name to stratify on
        test_size: Proportion of data for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Stratified split on '{target_column}': {1-test_size:.0%} train, {test_size:.0%} test")

    pos = df.filter(F.col(target_column) == 1)
    neg = df.filter(F.col(target_column) == 0)

    pos_train, pos_test = pos.randomSplit([1 - test_size, test_size], seed=random_seed)
    neg_train, neg_test = neg.randomSplit([1 - test_size, test_size], seed=random_seed)

    train_df = pos_train.union(neg_train)
    test_df = pos_test.union(neg_test)

    train_count = train_df.count()
    test_count = test_df.count()
    train_pos = pos_train.count()
    test_pos = pos_test.count()

    logger.info(f"✓ Train set: {train_count:,} rows ({train_pos:,} positive)")
    logger.info(f"✓ Test set: {test_count:,} rows ({test_pos:,} positive)")

    return train_df, test_df


def save_datasets_for_xgboost(
    train_df: DataFrame,
    test_df: DataFrame,
    train_output_dir: str,
    test_output_dir: str,
    target_column: str = 'is_fraud'
) -> None:
    """
    Save train and test datasets as XGBoost-compatible CSV files.

    CRITICAL FORMAT REQUIREMENTS:
    - No header row
    - Target column MUST be first
    - All numeric values
    - No index column

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        train_output_dir: Output directory for training data
        test_output_dir: Output directory for test data
        target_column: Name of target column
    """
    logger.info("Preparing datasets for XGBoost...")

    # Columns to exclude (non-predictive features)
    COLUMNS_TO_EXCLUDE = [
        'transaction_id', 'transaction_timestamp', 'customer_id', 'merchant_id',
        'card_number', 'cvv', 'expiry_date', 'transaction_date', 'year',
        'timestamp', 'id', 'data_version', 'created_at', 'updated_at',
        'fraud_prediction', 'fraud_probability'  # Model outputs, not inputs!
    ]

    # Get numeric columns only (XGBoost requirement)
    numeric_cols = [
        f.name for f in train_df.schema.fields
        if isinstance(f.dataType, (IntegerType, LongType, FloatType, DoubleType, ShortType, ByteType, BooleanType))
    ]

    # Filter out excluded columns
    numeric_cols = [col for col in numeric_cols if col not in COLUMNS_TO_EXCLUDE]

    # Ensure target column is included and first
    if target_column not in numeric_cols:
        # Try to convert target to numeric
        train_df = train_df.withColumn(target_column, F.col(target_column).cast("int"))
        test_df = test_df.withColumn(target_column, F.col(target_column).cast("int"))
        numeric_cols.insert(0, target_column)
    else:
        # Move target to first position
        numeric_cols = [target_column] + [c for c in numeric_cols if c != target_column]

    logger.info(f"Filtered to {len(numeric_cols)} numeric columns (XGBoost requirement)")
    logger.info(f"  First 5 columns: {numeric_cols[:5]}")
    logger.info(f"  Total features (excluding target): {len(numeric_cols) - 1}")

    # Select columns and fill NaN with 0
    train_df = train_df.select(numeric_cols).fillna(0)
    test_df = test_df.select(numeric_cols).fillna(0)

    # Save as CSV without header (XGBoost format)
    # Coalesce to single file for each dataset
    logger.info(f"Saving training data to {train_output_dir}/train.csv")
    (train_df.coalesce(1)
     .write
     .mode("overwrite")
     .option("header", "false")
     .csv(f"{train_output_dir}/temp"))

    # Move the single CSV file to train.csv
    # PySpark writes CSV to a directory with part files - need to consolidate
    logger.info(f"Saving test data to {test_output_dir}/test.csv")
    (test_df.coalesce(1)
     .write
     .mode("overwrite")
     .option("header", "false")
     .csv(f"{test_output_dir}/temp"))

    # Use boto3 to consolidate the CSV files
    s3 = boto3.client('s3')

    for output_dir, name in [(train_output_dir, 'train'), (test_output_dir, 'test')]:
        # Parse S3 path
        if output_dir.startswith('s3://'):
            s3_path = output_dir.replace('s3://', '')
            bucket = s3_path.split('/')[0]
            prefix = '/'.join(s3_path.split('/')[1:])

            # List objects in temp directory
            response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/temp/")

            # Find the CSV file (skip _SUCCESS and directories)
            csv_files = [
                obj['Key'] for obj in response.get('Contents', [])
                if obj['Key'].endswith('.csv')
            ]

            if csv_files:
                # Copy the CSV file to final location
                source_key = csv_files[0]
                dest_key = f"{prefix}/{name}.csv"
                s3.copy_object(
                    Bucket=bucket,
                    CopySource={'Bucket': bucket, 'Key': source_key},
                    Key=dest_key
                )
                logger.info(f"✓ Saved {name}.csv to {output_dir}")

                # Clean up temp directory
                for obj in response.get('Contents', []):
                    s3.delete_object(Bucket=bucket, Key=obj['Key'])
        else:
            # Local filesystem path (for testing)
            import shutil
            from pathlib import Path

            temp_dir = Path(output_dir) / "temp"
            csv_files = list(temp_dir.glob("*.csv"))

            if csv_files:
                source_file = csv_files[0]
                dest_file = Path(output_dir) / f"{name}.csv"
                shutil.copy(source_file, dest_file)
                logger.info(f"✓ Saved {name}.csv to {output_dir}")

                # Clean up temp directory
                shutil.rmtree(temp_dir)

    # Save feature metadata using boto3 (for training and evaluation)
    feature_names = [col for col in numeric_cols if col != target_column]
    feature_metadata = {
        'target_column': target_column,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'all_columns': numeric_cols  # target + features in order
    }

    # Save to both train and test directories
    for output_dir, name in [(train_output_dir, 'train'), (test_output_dir, 'test')]:
        metadata_content = json.dumps(feature_metadata, indent=2)

        if output_dir.startswith('s3://'):
            s3_path = output_dir.replace('s3://', '')
            bucket = s3_path.split('/')[0]
            key = '/'.join(s3_path.split('/')[1:]) + '/feature_metadata.json'

            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=metadata_content.encode('utf-8')
            )
            logger.info(f"✓ Saved feature_metadata.json to {output_dir}")
        else:
            # Local filesystem
            metadata_file = Path(output_dir) / "feature_metadata.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            metadata_file.write_text(metadata_content)
            logger.info(f"✓ Saved feature_metadata.json to {output_dir}")

    logger.info(f"✓ Datasets saved with {len(feature_names)} features")
    logger.info(f"  Feature names: {feature_names[:5]}...")


def save_statistics(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Save preprocessing statistics to output directory.

    Args:
        stats: Statistics dictionary
        output_dir: Output directory path
    """
    logger.info(f"Saving statistics to {output_dir}")

    stats_content = json.dumps(stats, indent=2)

    if output_dir.startswith('s3://'):
        s3 = boto3.client('s3')
        s3_path = output_dir.replace('s3://', '')
        bucket = s3_path.split('/')[0]
        key = '/'.join(s3_path.split('/')[1:]) + '/preprocessing_stats.json'

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=stats_content.encode('utf-8')
        )
        logger.info(f"✓ Statistics saved to s3://{bucket}/{key}")
    else:
        # Local filesystem
        from pathlib import Path
        stats_file = Path(output_dir) / "preprocessing_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.write_text(stats_content)
        logger.info(f"✓ Statistics saved to {stats_file}")


def main():
    """Main preprocessing function using PySpark."""
    parser = argparse.ArgumentParser(description="PySpark-based preprocessing for SageMaker")

    # Data source arguments — train and eval come from separate Athena tables
    # so the evaluation slice is stable across pipeline runs and can be reused
    # as the drift-monitor baseline. Both tables are populated by the
    # SeedAthenaTrainingData step with a deterministic 80/20 hash split.
    parser.add_argument('--train-table', type=str, default='training_data',
                       help='Athena table for the train channel (80%% split)')
    parser.add_argument('--eval-table', type=str, default='evaluation_data',
                       help='Athena table for the validation/test channel (20%% split)')
    parser.add_argument('--athena-filter', type=str, default=None,
                       help='SQL WHERE clause applied to BOTH train and eval tables')
    parser.add_argument('--limit', type=int, default=None,
                       help='Row limit for testing (applied to BOTH tables)')

    # Target column. Always passed by the pipeline via --target-column
    # (see pipeline.py:513). The value is sourced from dataset_schema.yaml
    # on the notebook host (pipeline.py:247's `default_value=
    # schema.target_column()`) and baked into the pipeline parameter, so
    # by the time this container runs, the value is already resolved.
    # required=True: fail loudly if a standalone / debug run forgets to
    # pass it, rather than silently drifting from the schema YAML.
    parser.add_argument('--target-column', type=str, required=True,
                       help='Target column name (required — value comes from '
                            'dataset_schema.yaml via the pipeline parameter)')

    # Output paths (SageMaker ProcessingStep provides these)
    parser.add_argument('--train-output-dir', type=str,
                       default='/opt/ml/processing/output/train',
                       help='Output directory for training data')
    parser.add_argument('--test-output-dir', type=str,
                       default='/opt/ml/processing/output/test',
                       help='Output directory for test data')
    parser.add_argument('--stats-output-dir', type=str,
                       default='/opt/ml/processing/output/stats',
                       help='Output directory for statistics')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PySpark Data Preprocessing for SageMaker Pipeline")
    logger.info("=" * 80)
    logger.info(f"Train table: {args.train_table}")
    logger.info(f"Eval table:  {args.eval_table}")
    logger.info(f"Target column: {args.target_column}")
    logger.info(f"Train output: {args.train_output_dir}")
    logger.info(f"Test output: {args.test_output_dir}")
    logger.info(f"Stats output: {args.stats_output_dir}")
    logger.info("")

    try:
        # Step 1: Create Spark session with Glue Data Catalog
        spark = create_spark_session()

        # Get database from environment
        database = os.getenv('ATHENA_DATABASE', 'fraud_detection')

        # Step 2: Read both tables from Athena. We no longer split in Spark —
        # the split is owned by the SeedAthenaTrainingData step (deterministic
        # MOD(xxhash64(transaction_id), 10)), so evaluation_data is stable
        # across model versions and usable as the drift-monitor baseline.
        train_df = read_from_athena(
            spark, database=database, table=args.train_table,
            filters=args.athena_filter, limit=args.limit,
        )
        test_df = read_from_athena(
            spark, database=database, table=args.eval_table,
            filters=args.athena_filter, limit=args.limit,
        )

        # Step 3: Convert boolean columns to numeric (must run on both DFs).
        logger.info("Converting boolean columns to numeric...")
        train_df = convert_boolean_columns(train_df)
        test_df = convert_boolean_columns(test_df)

        train_df.cache()
        test_df.cache()

        # Step 4: Validate data quality on the train side (gates the pipeline).
        # Capture eval distribution too so the stats report covers both sides.
        stats = validate_data_quality(train_df, args.target_column)
        if not stats['validation_passed']:
            logger.error("Data validation failed on train_df, aborting preprocessing")
            sys.exit(1)

        eval_stats = validate_data_quality(test_df, args.target_column)
        if not eval_stats['validation_passed']:
            logger.error("Data validation failed on eval_df, aborting preprocessing")
            sys.exit(1)

        # Add split statistics
        stats['train_samples'] = train_df.count()
        stats['test_samples'] = test_df.count()

        # Get class distribution for splits
        train_class_dist = train_df.groupBy(args.target_column).count().collect()
        test_class_dist = test_df.groupBy(args.target_column).count().collect()

        stats['train_class_distribution'] = {
            str(row[args.target_column]): row['count'] for row in train_class_dist
        }
        stats['test_class_distribution'] = {
            str(row[args.target_column]): row['count'] for row in test_class_dist
        }

        # Step 6: Save datasets in XGBoost-compatible format
        save_datasets_for_xgboost(
            train_df, test_df,
            args.train_output_dir, args.test_output_dir,
            args.target_column
        )

        # Step 7: Save statistics
        save_statistics(stats, args.stats_output_dir)

        # Stop Spark session
        spark.stop()

        logger.info("=" * 80)
        logger.info("✓ PySpark preprocessing completed successfully")
        logger.info(f"  Train samples: {stats['train_samples']:,}")
        logger.info(f"  Test samples: {stats['test_samples']:,}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
