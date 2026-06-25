"""Configuration module for the drift monitoring pipeline.

Loads configuration from config.yaml and .env file overrides.
Exposes all configuration constants as module-level variables.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: src/config/config.py)
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent.parent

# Load .env (environment-specific overrides take precedence)
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Load config.yaml (central defaults)
# ---------------------------------------------------------------------------
_CONFIG_YAML_PATH = _CONFIG_DIR / "config.yaml"
_yaml_cfg: dict = {}
if _CONFIG_YAML_PATH.exists():
    with open(_CONFIG_YAML_PATH, "r") as _f:
        _yaml_cfg = yaml.safe_load(_f) or {}


def _get(yaml_section: str, yaml_key: str, env_var: str, default=None):
    """Return env-var override → YAML value → default, in that priority.

    Empty values (empty string / whitespace) are treated as "not set" so the
    next source in the priority chain is used. This prevents blank placeholders
    in config.yaml or the environment from masking derived account-agnostic
    defaults (e.g. S3 bucket/path constants resolving to "").
    """
    env_val = os.environ.get(env_var)
    if env_val is not None and env_val.strip() != "":
        return env_val
    section = _yaml_cfg.get(yaml_section, {})
    if isinstance(section, dict) and yaml_key in section:
        yaml_val = section[yaml_key]
        if not (isinstance(yaml_val, str) and yaml_val.strip() == ""):
            return yaml_val
    return default


# ===================================================================
# AWS
# ===================================================================
AWS_DEFAULT_REGION: str = _get("aws", "region", "AWS_DEFAULT_REGION", "us-east-1")

# ===================================================================
# SageMaker
# ===================================================================
SAGEMAKER_EXEC_ROLE: str = _get(
    "sagemaker", "exec_role", "SAGEMAKER_EXEC_ROLE", ""
)
SERVERLESS_MEMORY_SIZE: int = int(
    _get("sagemaker", "serverless_memory_size", "SERVERLESS_MEMORY_SIZE", "2048")
)
SERVERLESS_MAX_CONCURRENCY: int = int(
    _get("sagemaker", "serverless_max_concurrency", "SERVERLESS_MAX_CONCURRENCY", "5")
)
BATCH_TRANSFORM_INSTANCE: str = _get(
    "sagemaker", "batch_transform_instance", "BATCH_TRANSFORM_INSTANCE", "ml.m5.xlarge"
)
BATCH_TRANSFORM_MAX_CONCURRENT: int = int(
    _get("sagemaker", "batch_transform_max_concurrent", "BATCH_TRANSFORM_MAX_CONCURRENT", "4")
)

# ===================================================================
# MLflow
# ===================================================================
MLFLOW_TRACKING_URI: str = _get(
    "mlflow", "tracking_uri", "MLFLOW_TRACKING_URI", ""
)
MLFLOW_EXPERIMENT_NAME: str = _get(
    "mlflow", "experiment_name", "MLFLOW_EXPERIMENT_NAME",
    "credit-card-fraud-detection-training",
)
MLFLOW_INFERENCE_EXPERIMENT_NAME: str = _get(
    "mlflow", "inference_experiment_name", "MLFLOW_INFERENCE_EXPERIMENT_NAME",
    "credit-card-fraud-detection-inference",
)
MLFLOW_BATCH_EXPERIMENT_NAME: str = _get(
    "mlflow", "batch_experiment_name", "MLFLOW_BATCH_EXPERIMENT_NAME",
    "credit-card-fraud-detection-batch",
)
MLFLOW_MONITORING_EXPERIMENT_NAME: str = _get(
    "mlflow", "monitoring_experiment_name", "MLFLOW_MONITORING_EXPERIMENT_NAME",
    "credit-card-fraud-detection-monitoring",
)
MLFLOW_MODEL_NAME: str = _get(
    "mlflow", "model_name", "MLFLOW_MODEL_NAME", "xgboost-fraud-detector"
)

# ===================================================================
# Athena
# ===================================================================
ATHENA_DATABASE: str = _get("athena", "database", "ATHENA_DATABASE", "fraud_detection")
ATHENA_WORKGROUP: str = _get("athena", "workgroup", "ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT_S3: str = _get("athena", "output_s3", "ATHENA_OUTPUT_S3", "")
ATHENA_QUERY_TIMEOUT: int = int(
    _get("athena", "query_timeout", "ATHENA_QUERY_TIMEOUT", "300")
)

ATHENA_TRAINING_TABLE: str = _get(
    "athena", "training_table", "ATHENA_TRAINING_TABLE", "training_data"
)
ATHENA_INFERENCE_TABLE: str = _get(
    "athena", "inference_table", "ATHENA_INFERENCE_TABLE", "inference_responses"
)
ATHENA_GROUND_TRUTH_TABLE: str = _get(
    "athena", "ground_truth_table", "ATHENA_GROUND_TRUTH_TABLE", "ground_truth"
)
ATHENA_GROUND_TRUTH_UPDATES_TABLE: str = _get(
    "athena", "ground_truth_updates_table", "ATHENA_GROUND_TRUTH_UPDATES_TABLE",
    "ground_truth_updates",
)
ATHENA_DRIFTED_DATA_TABLE: str = _get(
    "athena", "drifted_data_table", "ATHENA_DRIFTED_DATA_TABLE", "drifted_data"
)
ATHENA_MONITORING_RESPONSES_TABLE: str = os.environ.get(
    "ATHENA_MONITORING_RESPONSES_TABLE", "monitoring_responses"
)

# ===================================================================
# S3 paths
# ===================================================================
def _derive_data_bucket() -> str:
    """Derive the data bucket account-agnostically.

    Priority: DATA_S3_BUCKET env/yaml -> ${PROJECT_NAME}-data-${account_id}.
    The CFN data bucket follows the convention ``${ProjectName}-data-${AWS::AccountId}``,
    so when no explicit bucket is configured we reconstruct it from PROJECT_NAME and
    the caller's AWS account ID. This keeps the config valid in any account/deployment
    without hardcoding a bucket name.
    """
    explicit = _get("s3", "bucket", "DATA_S3_BUCKET", "")
    if explicit:
        return explicit

    project_name = _get("project", "name", "PROJECT_NAME", "")
    if not project_name:
        return ""

    try:
        import boto3

        account_id = boto3.client(
            "sts", region_name=AWS_DEFAULT_REGION
        ).get_caller_identity()["Account"]
    except Exception:
        return ""

    return f"{project_name}-data-{account_id}"


DATA_S3_BUCKET: str = _derive_data_bucket()
DATA_S3_PREFIX: str = _get("s3", "prefix", "DATA_S3_PREFIX", "fraud-detection/")

# Derive the Athena query-results location from the data bucket when not set
# explicitly (account-agnostic, matches the CFN lifecycle/Lambda convention
# ``s3://${ProjectName}-data-${AWS::AccountId}/athena-results/``).
if not ATHENA_OUTPUT_S3 and DATA_S3_BUCKET:
    ATHENA_OUTPUT_S3 = f"s3://{DATA_S3_BUCKET}/athena-results/"


def _s3_path(yaml_key: str, env_var: str, suffix: str) -> str:
    """Build S3 path from config or derive from bucket + prefix + suffix."""
    val = _get("s3", yaml_key, env_var, "")
    if val:
        return val
    if DATA_S3_BUCKET:
        return f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}{suffix}"
    return ""


S3_TRAINING_DATA: str = _s3_path("training_data", "S3_TRAINING_DATA", "training_data/")
S3_GROUND_TRUTH: str = _s3_path("ground_truth", "S3_GROUND_TRUTH", "ground_truth/")
S3_INFERENCE_RESPONSES: str = _s3_path(
    "inference_responses", "S3_INFERENCE_RESPONSES", "inference_responses/"
)
S3_DRIFTED_DATA: str = _s3_path("drifted_data", "S3_DRIFTED_DATA", "drifted_data/")
S3_GROUND_TRUTH_UPDATES: str = _s3_path(
    "ground_truth_updates", "S3_GROUND_TRUTH_UPDATES", "ground_truth_updates/"
)
S3_MODEL_ARTIFACTS: str = _s3_path(
    "model_artifacts", "S3_MODEL_ARTIFACTS", "model_artifacts/"
)
S3_TRAINING_DATA_EXPORT: str = _s3_path(
    "training_data_export", "S3_TRAINING_DATA_EXPORT", "training_data_export/"
)
S3_BATCH_TRANSFORM_INPUT: str = _s3_path(
    "batch_transform_input", "S3_BATCH_TRANSFORM_INPUT", "batch_transform/input/"
)
S3_BATCH_TRANSFORM_OUTPUT: str = _s3_path(
    "batch_transform_output", "S3_BATCH_TRANSFORM_OUTPUT", "batch_transform/output/"
)

# ===================================================================
# SQS
# ===================================================================
SQS_QUEUE_NAME: str = _get(
    "sqs", "inference_queue_name", "SQS_QUEUE_NAME", "fraud-inference-logs"
)
SQS_QUEUE_URL: str = _get("sqs", "inference_queue_url", "SQS_QUEUE_URL", "")
MONITORING_SQS_QUEUE_NAME: str = _get(
    "sqs", "monitoring_queue_name", "MONITORING_SQS_QUEUE_NAME",
    "fraud-monitoring-results",
)
MONITORING_SQS_QUEUE_URL: str = _get(
    "sqs", "monitoring_queue_url", "MONITORING_SQS_QUEUE_URL", ""
)

# ===================================================================
# Inference logging
# ===================================================================
# These configure the legacy direct-Athena writer (inference_handler.py). The
# CloudFormation-provisioned SQS→Lambda batch parameters are set in the CFN
# template; these values only apply when SQS_QUEUE_URL is unset.
ENABLE_ATHENA_LOGGING: bool = (
    _get("inference_logging", "enable_athena_logging", "ENABLE_ATHENA_LOGGING", "true")
    .lower() in ("true", "1", "yes")
)
INFERENCE_LOG_BATCH_SIZE: int = int(
    _get("inference_logging", "batch_size", "INFERENCE_LOG_BATCH_SIZE", "50")
)
INFERENCE_LOG_FLUSH_INTERVAL: int = int(
    _get("inference_logging", "flush_interval", "INFERENCE_LOG_FLUSH_INTERVAL", "300")
)

# ===================================================================
# Inference confidence thresholds
# ===================================================================
HIGH_CONFIDENCE_THRESHOLD: float = float(
    _get("inference", "high_confidence_threshold", "HIGH_CONFIDENCE_THRESHOLD", "0.9")
)
LOW_CONFIDENCE_LOWER: float = float(
    _get("inference", "low_confidence_lower", "LOW_CONFIDENCE_LOWER", "0.4")
)
LOW_CONFIDENCE_UPPER: float = float(
    _get("inference", "low_confidence_upper", "LOW_CONFIDENCE_UPPER", "0.6")
)

# ===================================================================
# Lambda
# ===================================================================
LAMBDA_LOGGER_NAME: str = _get(
    "lambda", "logger_name", "LAMBDA_LOGGER_NAME", "fraud-inference-log-consumer"
)
LAMBDA_EXEC_ROLE: str = _get("lambda", "exec_role", "LAMBDA_EXEC_ROLE", "")

# ===================================================================
# Data file paths
# ===================================================================
DATA_DIR: Path = _PROJECT_ROOT / "data"

# Local CSV paths (used only by upload_data_to_s3.py and download_kaggle_dataset.py)
CSV_TRAINING_DATA: Path = _PROJECT_ROOT / _get(
    "data", "csv_training_data", "CSV_TRAINING_DATA",
    "data/creditcard_predictions_final.csv",
)
CSV_GROUND_TRUTH: Path = _PROJECT_ROOT / _get(
    "data", "csv_ground_truth", "CSV_GROUND_TRUTH",
    "data/creditcard_ground_truth.csv",
)
CSV_DRIFTED_DATA: Path = _PROJECT_ROOT / _get(
    "data", "csv_drifted_data", "CSV_DRIFTED_DATA",
    "data/creditcard_drifted.csv",
)

# S3 data paths — pipeline and Athena code should use these
_s3_data_base = f"s3://{DATA_S3_BUCKET}/{DATA_S3_PREFIX}data" if DATA_S3_BUCKET else ""

S3_CSV_TRAINING_DATA: str = _get(
    "data", "s3_training_data", "S3_TRAINING_DATA",
    f"{_s3_data_base}/creditcard_predictions_final.csv" if _s3_data_base else "",
)
S3_CSV_GROUND_TRUTH: str = _get(
    "data", "s3_ground_truth", "S3_GROUND_TRUTH",
    f"{_s3_data_base}/creditcard_ground_truth.csv" if _s3_data_base else "",
)
S3_CSV_DRIFTED_DATA: str = _get(
    "data", "s3_drifted_data", "S3_DRIFTED_DATA",
    f"{_s3_data_base}/creditcard_drifted.csv" if _s3_data_base else "",
)

# ===================================================================
# Training
# ===================================================================
RANDOM_STATE: int = int(_get("training", "random_state", "RANDOM_STATE", "42"))

# Target column name in the Kaggle dataset post-rename. Not in config.yaml
# because it's a fixed property of data/download_kaggle_dataset.py — exposed
# here so notebooks/3_inference_monitoring.ipynb can reference it.
TARGET_COLUMN: str = os.environ.get("TARGET_COLUMN", "is_fraud")

# The 30 feature columns (PCA components V1-V28 + transaction_amount-style
# names from data/download_kaggle_dataset.py KAGGLE_COLUMN_MAP). Listed here
# so monitoring code can parse the inference_responses.input_features JSON
# without rediscovering the schema on every call. Order matches the staging
# table created during the Athena seed.
TRAINING_FEATURES: list[str] = [
    "transaction_hour", "transaction_day_of_week", "transaction_amount",
    "transaction_type_code", "customer_age", "customer_gender",
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

_training_cfg = _yaml_cfg.get("training", {}) if isinstance(
    _yaml_cfg.get("training"), dict
) else {}

XGBOOST_PARAMS: dict = _training_cfg.get("xgboost_params") or {
    "max_depth": 4,
    "learning_rate": 0.05,
    "num_boost_round": 200,
    "min_child_weight": 10,
    "early_stopping_rounds": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
}

# ===================================================================
# Drift thresholds (used by monitor_model_performance.py)
# ===================================================================
# The daily drift Lambda reads its own thresholds from env vars
# (DATA_DRIFT_THRESHOLD, KS_PVALUE_THRESHOLD, MODEL_DRIFT_THRESHOLD,
#  DATA_DRIFT_LOOKBACK_DAYS, MODEL_DRIFT_LOOKBACK_DAYS) — not from this file.
_drift_cfg = _yaml_cfg.get("drift_thresholds", {}) if isinstance(
    _yaml_cfg.get("drift_thresholds"), dict
) else {}

MIN_ROC_AUC_THRESHOLD: float = float(
    _drift_cfg.get("min_roc_auc", os.environ.get("MIN_ROC_AUC_THRESHOLD", "0.85"))
)

# ===================================================================
# Monitoring lookback windows (used by notebook 2 + monitor_model_performance.py)
# ===================================================================
# Override via env vars when the Lambda runs — see also DATA_DRIFT_LOOKBACK_DAYS /
# MODEL_DRIFT_LOOKBACK_DAYS read directly by lambda_drift_monitor.py.
MONITORING_DATA_DRIFT_LOOKBACK_DAYS: int = int(
    os.environ.get("MONITORING_DATA_DRIFT_LOOKBACK_DAYS", "7")
)
MONITORING_MODEL_DRIFT_LOOKBACK_DAYS: int = int(
    os.environ.get("MONITORING_MODEL_DRIFT_LOOKBACK_DAYS", "30")
)

# ===================================================================
# Ground-truth simulation (dev/test only — see notebook 2 Section 4)
# ===================================================================
# These knobs let notebook 2 inject controlled drift into the ground-truth
# simulator so the monitoring pipeline can be exercised end-to-end. In
# production, ground truth comes from fraud-investigation feeds; these
# constants are unused.
GROUND_TRUTH_SIM_ACCURACY: float = float(
    os.environ.get("GROUND_TRUTH_SIM_ACCURACY", "0.85")
)
GROUND_TRUTH_SIM_FEATURE_DRIFT_MAG: float = float(
    os.environ.get("GROUND_TRUTH_SIM_FEATURE_DRIFT_MAG", "0.0")
)
GROUND_TRUTH_SIM_FEATURE_DRIFT_COUNT: int = int(
    os.environ.get("GROUND_TRUTH_SIM_FEATURE_DRIFT_COUNT", "0")
)
GROUND_TRUTH_SIM_FEATURE_DRIFT_IMPACT: float = float(
    os.environ.get("GROUND_TRUTH_SIM_FEATURE_DRIFT_IMPACT", "0.0")
)
GROUND_TRUTH_SIM_MODEL_DRIFT_MAG: float = float(
    os.environ.get("GROUND_TRUTH_SIM_MODEL_DRIFT_MAG", "0.0")
)

# ===================================================================
# Drift dataset generation
# ===================================================================
_drift_gen_cfg = _yaml_cfg.get("drift_generation", {}) if isinstance(
    _yaml_cfg.get("drift_generation"), dict
) else {}

DRIFT_GEN_DEFAULT_CONFIG: dict = _drift_gen_cfg.get("default_drift", {})
DRIFT_GEN_VARIABLE_PATTERNS: dict = _drift_gen_cfg.get("variable_patterns", {})
DRIFT_GEN_NUM_SAMPLES: int = int(_drift_gen_cfg.get("num_samples", "5000"))
DRIFT_GEN_NUM_SAMPLES_PER_RUN: int = int(_drift_gen_cfg.get("num_samples_per_run", "2000"))
DRIFT_GEN_RANDOM_STATE: int = int(_drift_gen_cfg.get("random_state", "123"))

# ===================================================================
# QuickSight (used by notebooks/4_governance_dashboard.ipynb)
# ===================================================================
# These IDs and display names are constants of the deployed dashboard, not
# tunables — they live here as plain literals (no YAML mirror).
QUICKSIGHT_DATASOURCE_ID: str = "fraud-governance-athena-datasource"
QUICKSIGHT_DATASOURCE_NAME: str = "Fraud Governance - Athena"
QUICKSIGHT_INFERENCE_DATASET_ID: str = "fraud-governance-inference-dataset"
QUICKSIGHT_INFERENCE_DATASET_NAME: str = "Fraud Governance - Inference Monitoring"
QUICKSIGHT_DRIFT_DATASET_ID: str = "fraud-governance-drift-dataset"
QUICKSIGHT_DRIFT_DATASET_NAME: str = "Fraud Governance - Drift Monitoring"
QUICKSIGHT_FEATURE_DRIFT_DATASET_ID: str = "fraud-governance-feature-drift-dataset"
QUICKSIGHT_FEATURE_DRIFT_DATASET_NAME: str = "Fraud Governance - Feature Drift Analysis"
QUICKSIGHT_ANALYSIS_ID: str = "fraud-governance-analysis"
QUICKSIGHT_ANALYSIS_NAME: str = "Fraud Detection Governance Analysis"
QUICKSIGHT_DASHBOARD_ID: str = "fraud-governance-dashboard"
QUICKSIGHT_DASHBOARD_NAME: str = "Fraud Detection Governance"
QUICKSIGHT_SERVICE_ROLE_NAME: str = "aws-quicksight-service-role-v0"
