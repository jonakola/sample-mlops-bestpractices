# Automated Drift and Trend Monitoring for ML Models on Amazon SageMaker

End-to-end MLOps reference architecture for credit card fraud detection with automated drift detection, ground truth integration, and governance dashboard. Built on SageMaker Pipelines, MLflow, Evidently AI, and QuickSight.

## Architecture

![MLOps Architecture](docs/guides/MetaMonitoring.png)

See [ARCHITECTURE_STEPS.md](docs/ARCHITECTURE_STEPS.md) for detailed step-by-step descriptions.

## Quickstart

### Step 1: Deploy the CloudFormation Stack

Provisions everything: SageMaker domain, user profile, JupyterLab space, MLflow tracking server, S3 bucket, VPC, SQS queue, Lambda inference logger, and IAM role with all required permissions. On first space launch, the lifecycle script clones this repo, downloads the Kaggle training data, uploads to S3, creates Athena tables, and writes a populated `.env` file.

```bash
./cloudformation/deploy-stack.sh                       # default: fraud-detection-monitoring in us-west-2
./cloudformation/deploy-stack.sh my-stack              # override stack name
AWS_REGION=us-east-1 ./cloudformation/deploy-stack.sh  # override region
```

Idempotent — re-runs create if missing, update if present. First create takes ~10–15 minutes.

**Prerequisites:** AWS CLI configured, IAM permissions for CloudFormation/IAM/SageMaker/Lambda/VPC, a region with SageMaker + MLflow availability (`us-east-1`, `us-west-2`, `eu-west-1`).

See [`cloudformation/README.md`](cloudformation/README.md) for parameter reference, troubleshooting, and update/delete instructions.

### Step 2: Run the JupyterLab Space

1. Open the SageMaker console → Domains → `fraud-detection-monitoring-domain`
2. Select the user profile → **Spaces** → click **Run Space** on the JupyterLab space
3. Once JupyterLab starts, verify the lifecycle script completed:
   - `sample-mlops-bestpractices/` directory is present
   - `.env` file has region, role, MLflow ARN, and S3 bucket populated

To pick up new commits without redeploying: **Stop Space**, then **Run Space** (the lifecycle script runs `git pull --ff-only` on every start). Uncommitted local edits are preserved.

### Step 3: Run the Notebooks in Order

Open `sample-mlops-bestpractices/sagemaker-automated-drift-and-trend-monitoring/notebooks/` and run these notebooks sequentially.

> 📊 **Viewing the notebooks online**: GitHub's renderer strips the JavaScript that powers Evidently's interactive drift reports, plotly charts, and ipywidgets — so the saved output cells degrade to raw HTML. To see the live, interactive output without re-running the notebooks, open them via [nbviewer](https://nbviewer.org/github/aws-samples/sample-mlops-bestpractices/tree/main/sagemaker-automated-drift-and-trend-monitoring/notebooks/) instead. Each notebook has a direct nbviewer link in its title cell.


| # | Notebook | Purpose | Time |
|---|----------|---------|------|
| 1 | `1_training_pipeline.ipynb` | Builds and executes the SageMaker training pipeline: preprocessing → XGBoost training → evaluation (quality gate at ROC-AUC ≥ 0.70) → MLflow registration. Does NOT deploy endpoint — see notebook 2 for deployment. | ~20 min |
| 2 | `2_deployment.ipynb` | Deploys the trained model to a SageMaker serverless endpoint with custom Athena-logging inference handler. Select model from registry, configure resources, test deployment. | ~10 min |
| 3 | `3_inference_monitoring.ipynb` | Tests the deployed endpoint, simulates ground truth, applies ground truth via Athena MERGE, runs Evidently data + model drift detection, logs interactive reports to MLflow, and sets up the daily-scheduled drift Lambda. | ~30 min |
| 4 | `4_governance_dashboard.ipynb` | Creates a QuickSight governance dashboard (datasource, datasets, analysis, published dashboard) with auto-refresh via EventBridge + Lambda. Requires QuickSight Enterprise subscription. | ~15 min |

**Optional notebooks** (run as needed):

| # | Notebook | Purpose |
|---|----------|---------|
| 5 | `5_optional_version_validation.ipynb` | Verifies MLflow model version matches the deployed endpoint and Athena inference logs (traceability check). |
| 6 | `6_optional_cleanup.ipynb` | Deletes all AWS resources created outside CloudFormation (Lambda functions, endpoints, SNS topics, dashboards, CloudWatch alarms). |
| 7 | `7_shap_explainability.ipynb` | Generates SHAP global and per-prediction feature importance plots for the trained model. |

> **Note:** Notebook 1 cell 2 runs `! uv pip install -e ../`. If you see `No virtual environment found`, the error message tells you how to resolve it (run `uv venv` or add `--system`).

## What This Solution Does

### Training Pipeline (Notebook 1)
1. **Seed Athena** — Idempotently loads the predictions CSV into `training_data` (80%) + `evaluation_data` (20%), deterministic hash split on `transaction_id`
2. **Preprocess** — Reads `training_data` (train channel) and `evaluation_data` (test channel) from Athena; encodes categoricals; emits XGBoost-format CSVs
3. **Train** — XGBoost with automatic class imbalance handling (`scale_pos_weight`)
4. **Evaluate** — Computes ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix; writes `baseline.json` with metrics + Iceberg snapshot IDs + code commit SHA
5. **Quality Gate** — Registers the model only if ROC-AUC ≥ 0.70
6. **MLflow Registration** — Logs metrics, parameters, and model artifact to MLflow. The Model Registry record's `ModelStatistics` URI points at `baseline.json` (used by the drift monitor)

### Deployment (Notebook 2)
- **Model Selection** — Choose approved model from SageMaker Model Registry
- **Endpoint Creation** — Serverless endpoint with custom inference handler
- **Athena Logging** — Every prediction logged to SQS → Lambda → Athena (zero added latency)
- **Testing** — Verify endpoint responds correctly to test predictions

### Monitoring (Notebook 3)
- **Inference logging** — Every prediction → SQS → Lambda batches (10 msgs or 30s) → `inference_responses` Iceberg table
- **Ground truth integration** — Simulated (dev) or fed from fraud investigation systems (prod) → `ground_truth_updates` table → MERGE into `inference_responses`
- **Drift detection** — Evidently `DataDriftPreset` against the frozen `training_data` baseline (KS for numerics, chi-square for categoricals, PSI per feature) + `ClassificationPreset` against the frozen `evaluation_data` baseline (ROC, PR, confusion matrix). Thresholds configurable via `src/config/config.yaml`.
- **Per-run traceability** — Each drift run gets a `monitoring_run_id` (`notebook-drift-*` from the notebook, `drift-*` from the Lambda). Both writers (a) record one row in `monitoring_responses` keyed on this id, and (b) backfill the same id onto the `inference_responses` rows the run scored — `WHERE monitoring_run_id IS NULL` makes this naturally delta-shaped, so each run only tags predictions never measured before. QuickSight can join the two tables on `monitoring_run_id` to show "what predictions did this run measure?". The **notebook additionally** scopes the *drift compute window itself* to `request_timestamp > MAX(monitoring_timestamp)`, letting you re-run drift detection on just the new predictions since the last run.
- **Automated daily checks** — EventBridge → Lambda (`fraud-detection-drift-monitor`) at 2 AM UTC. Logs metrics + interactive HTML reports to MLflow, writes summary to `monitoring_responses` table, sends SNS alert if drift exceeds thresholds. Lambda scopes its drift compute to fixed 7/30-day rolling windows (data drift / model drift respectively); the `monitoring_run_id` backfill runs after every Lambda invocation.

### Governance (Notebook 4)
- QuickSight dashboard with prediction volume, fraud probability distribution, accuracy breakdown, risk tiers, latency trend, drift trends, ROC-AUC over time
- Auto-refreshes daily at 3 AM UTC via EventBridge + Lambda after the 2 AM drift monitoring run

## Project Structure

```
sagemaker-automated-drift-and-trend-monitoring/
├── cloudformation/
│   ├── sagemaker-mlflow-setup.yaml    # Single-stack deployment template
│   ├── deploy-stack.sh                # Create-or-update script
│   └── README.md                      # CloudFormation reference
├── notebooks/                         # Run these in order — see Step 3 above
│   ├── 1_training_pipeline.ipynb
│   ├── 2_deployment.ipynb
│   ├── 3_inference_monitoring.ipynb
│   ├── 4_governance_dashboard.ipynb
│   ├── 5_optional_version_validation.ipynb
│   ├── 6_optional_cleanup.ipynb
│   └── 7_shap_explainability.ipynb
├── src/
│   ├── train_pipeline/                # Pipeline definition + preprocessing/training/evaluation steps
│   ├── drift_monitoring/              # Drift detection Lambdas, ground truth utilities, Evidently wrappers
│   ├── governance/                    # QuickSight dashboard provisioning
│   ├── setup/                         # IAM role + infrastructure helpers
│   ├── config/                        # config.py + config.yaml (drift thresholds, simulation params)
│   └── utils/                         # AWS session, MLflow, visualization helpers
├── data/
│   └── download_kaggle_dataset.py     # Downloads Kaggle credit card fraud dataset
├── docs/
│   ├── ARCHITECTURE_STEPS.md          # 11-step architecture walkthrough
│   ├── VERSION_MANAGEMENT.md          # MLflow model versioning guide
│   └── guides/                        # Architecture diagrams (PNG + Excalidraw sources)
└── main.py                            # CLI entry point (alternative to notebook workflow)
```

## MLOps Lineage & Reproducibility

> The principle: **anything that participates in a model's lineage must be addressable by an immutable reference. Names are pointers. Pointers are fine for humans, fatal for joins.**

Four immutable references anchor every model in this system:

| Reference | Where it comes from | What it pins |
|---|---|---|
| **Model package ARN** | `sagemaker:Register` step | The model artifact + its metrics |
| **`training_snapshot_id`** | `INSERT` commit on `training_data` | The exact rows the model was **trained** on (used as the data-drift baseline) |
| **`evaluation_snapshot_id`** | `INSERT` commit on `evaluation_data` | The exact rows the model was **scored** on (used as the model-drift baseline) |
| **Code commit SHA** | Captured by `pipeline.py` at definition time (overridable via the `CodeCommitSha` pipeline parameter in CI) | The preprocessing & training logic |
| **`monitoring_run_id`** | Generated per drift run (notebook or Lambda) | Stamped on every row in `monitoring_responses` AND backfilled onto the `inference_responses` rows that the run scored — so QuickSight can join the two tables to answer "which predictions did monitoring run X measure?" |

These travel together inside `baseline.json`, which is registered as `ModelPackage.ModelMetrics.ModelStatistics` on every model. The drift Lambda dereferences them on every run.

### `baseline.json` schema (v2)

```json
{
  "schema_version": 2,
  "model_package_group": "xgboost-fraud-detector",
  "code_commit_sha": "a1b2c3...",
  "evaluation_table": "evaluation_data",
  "training_table":   "training_data",
  "evaluation_snapshot_id": "1827...",
  "training_snapshot_id":   "1825...",
  "feature_schema_version": 1,
  "feature_schema": [{"name": "transaction_amount", "dtype": "double"}, ...],
  "metrics": { "roc_auc": 0.94, "pr_auc": 0.78, ... },
  "sample_size": 56960
}
```

### How the drift Lambda resolves "what's in production"

```
ENDPOINT_NAME
   → describe_endpoint            (live config)
   → describe_endpoint_config     (production variant)
   → variant.ModelName            (model object)
   → describe_model               (containers)
   → ModelPackageName             (← the immutable reference)
   → describe_model_package
   → ModelMetrics.ModelStatistics.S3Uri
   → baseline.json
```

Never `ListModelPackages(SortOrder=Descending, MaxResults=1)` — that conflates "what we built last" with "what's running now," which diverges during rollouts, rollbacks, or pending approvals.

### How baseline data is sampled

Two drift checks, two different baselines — both Iceberg-snapshot-pinned via `baseline.json`:

```sql
-- DATA DRIFT — compares production features to the distribution the model was TRAINED on
SELECT ...
FROM fraud_detection.training_data
  FOR VERSION AS OF <training_snapshot_id>      -- Iceberg time travel
WHERE is_fraud IS NOT NULL
LIMIT 5000
```

```sql
-- MODEL DRIFT — compares (target, prediction) pairs to the labeled held-out set
SELECT CAST(is_fraud AS INT) AS target, CAST(fraud_prediction AS INT) AS prediction
FROM fraud_detection.evaluation_data
  FOR VERSION AS OF <evaluation_snapshot_id>    -- different snapshot, different table
WHERE is_fraud IS NOT NULL AND fraud_prediction IS NOT NULL
```

Re-seeding either table later cannot retroactively corrupt the reference because each model package's `baseline.json` pins the exact snapshot it was bound to. The Lambda falls back to the live table only when no snapshot ID is recorded (first runs, schema-v1 baselines).

**Why training_data for data drift?** Industry standard (SageMaker Model Monitor, NannyML, Arize): data drift measures whether the production input distribution has moved away from what the model was *trained on*. Comparing to `evaluation_data` instead would only flag shifts away from the held-out test slice — a narrower question. `evaluation_data` is the right baseline for model drift because it carries the model's scored predictions, which is what you compare current performance to.

### The seven-questions test

A production MLOps system should answer all of these in one API call or one query. Score this codebase as of today:

| # | Question | How it's answered | Status |
|---|---|---|---|
| 1 | What model is in production right now? | `describe_endpoint` → production variant → `ModelPackageName` | ✅ |
| 2 | What baseline does it have? | `baseline.json` registered on the resolved ModelPackage | ✅ |
| 3 | What data was it scored on? | Iceberg snapshot ID pinned in `baseline.json`, queryable via `FOR VERSION AS OF` | ✅ |
| 4 | What code built it? | `code_commit_sha` in `baseline.json` | ✅ |
| 5 | Is it drifting right now? | Scheduled drift Lambda → MLflow + SNS + `monitoring_responses` | ✅ |
| 6 | When did drift start? | `monitoring_responses` time series, filterable by `model_package_arn` | ✅ |
| 7 | How did the previous model compare on the same window? | `monitoring_responses` carries `model_package_arn` per record — query by ARN | ✅ (manual query) |

### Operational rules

- **Never overwrite an artifact registered to a ModelPackage.** Pipeline outputs are namespaced per-execution; resist tidying them up.
- **Schema changes are model-version changes.** Bump `FeatureSchemaVersion` (pipeline parameter) and force a retrain. Don't migrate old baselines forward.
- **One drift record per `(endpoint, variant, model_package_arn)` per run.** Multi-model production is the rule, not the exception.
- **Separate the three signals:** *data drift* (input distribution), *model drift* (performance degradation), *coverage gap* (no ground truth yet). Conflating them produces unhelpful alerts.
- **Run the pipeline from CI with `CodeCommitSha` set explicitly.** Local-machine SHAs are unreliable in shared environments.

### Schema iteration while building

The CFN lifecycle script uses `CREATE TABLE IF NOT EXISTS`. **Existing tables are not migrated when the template changes** — they're silently skipped. Until you have data you can't lose, drop the database between schema changes:

```sql
-- in Athena
DROP DATABASE fraud_detection CASCADE;
```

Then re-launch the JupyterLab Space (the lifecycle script re-runs and recreates everything fresh).

## Why This Architecture

**ML models degrade silently in production.** Most teams invest in training pipelines but leave inference monitoring as an afterthought. This solution closes that gap with an end-to-end, open-source MLOps system:

- **Open-source SDKs** (MLflow, Evidently, scikit-learn) — portable across AWS/GCP/Azure/on-prem, no vendor lock-in
- **Serverless cost profile** — scales to zero; ~$30–50/month for 1000 predictions/day vs. $200+/month for managed alternatives
- **Production-grade** — handles delayed ground truth (typical in fraud), concept drift, multi-feature drift, and alerting
- **Custom inference handler** — automatic prediction logging with zero added latency (fire-and-forget SQS → batched Lambda → Athena)
- **MLflow as single pane of glass** — all metrics, Evidently HTML reports, and artifacts in one place

### Why Not SageMaker `DataCaptureConfig`?

`DataCaptureConfig` captures raw request/response payloads to S3 in JSONL — useful, but only on real-time endpoints, and you still need drift detection and alerting on top. This solution prioritizes Evidently-powered drift detection, MLflow as the monitoring hub, Athena as the data lake, and supports both serverless and real-time endpoints.

## Drift Detection

### Statistical Tests

- **Numerical features:** Kolmogorov–Smirnov (default) — sensitive to tail changes, important for fraud pattern shifts. Configurable to Wasserstein, Jensen-Shannon, KL, or PSI via `EVIDENTLY_NUM_STAT_TEST` in `.env`.
- **Categorical features:** Chi-square test
- **Feature-level drift:** A feature is flagged if PSI ≥ 0.2 OR KS p-value < 0.05.

### Thresholds (configurable in `src/config/config.yaml`)

| PSI value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1     | No significant shift | None |
| 0.1 – 0.2 | Moderate shift | Monitor closely |
| ≥ 0.2     | Significant shift | **Alert / investigate / retrain** |

| Performance degradation | Action |
|-------------------------|--------|
| < 3%                    | Normal variance |
| 3–5%                    | Monitor |
| > 5%                    | **Alert triggered** |

### Reports

All drift runs log to MLflow experiment `fraud-detection-drift_monitoring`:
- `evidently_reports/data_drift_*.html` — interactive per-feature distribution comparisons
- `evidently_reports/classification_*.html` — ROC, PR, confusion matrix, F1
- `drift_reports/drift_summary_*.json` — structured summary

See [Understanding Drift Scores](docs/screenshots/quicksight/README.md) for a visual guide to interpreting PSI and drift metrics.

## Athena Data Lake

| Table | Type | Purpose |
|-------|------|---------|
| `training_data` | Iceberg | Training features (~80% of Kaggle rows, 30 features). Populated by the `SeedAthenaTrainingData` pipeline step. Also serves as the **data-drift baseline** (time-travelled via `training_snapshot_id`). |
| `evaluation_data` | Iceberg | Held-out evaluation slice (~20%, same hash split). Read by preprocessing for the test channel and by the drift monitor as the **model-drift baseline** (time-travelled via `evaluation_snapshot_id`) — its `is_fraud` + `fraud_prediction` columns let the monitor compare current performance to the model's scored test set. |
| `inference_responses` | Iceberg | All endpoint predictions, partitioned by day. `ground_truth` column populated via MERGE from `ground_truth_updates`. `monitoring_run_id` column backfilled by each drift run so QuickSight can join with `monitoring_responses` to show "which predictions this run measured". |
| `ground_truth_updates` | Iceberg | Lightweight patches: `inference_id` + `actual_fraud` + confirmation metadata |
| `monitoring_responses` | Iceberg | One row per drift run. Includes `monitoring_run_id`, `model_package_arn`, and `evaluation_snapshot_id` so the table can be queried per-run and per-version. |
| `drifted_data` | External (Parquet) | Synthetic drifted samples for testing |

### Ground Truth Flow

```text
inference_responses                ground_truth_updates
(prediction, ground_truth=NULL)    (inference_id + actual_fraud)
         │                                    │
         └────── MERGE ON inference_id ───────┘
                          ▼
              inference_responses (ground_truth populated)
                          ▼
              Drift Detection (computes metrics where ground_truth IS NOT NULL)
```

In **dev/test**, ground truth is simulated by `src/drift_monitoring/simulate_ground_truth_from_athena.py` (configurable accuracy, realistic confirmation delays). In **production**, replace the simulator with feeds from fraud investigation systems, chargeback notifications, or customer reports — all writing to `ground_truth_updates`.

## CLI Alternative

Each notebook step has a CLI equivalent if you're scripting CI/CD instead of using notebooks:

| Phase | CLI |
|-------|-----|
| Create pipeline | `python main.py pipeline create --pipeline-name fraud-detection-pipeline` |
| Train & deploy | `python main.py pipeline start --pipeline-name fraud-detection-pipeline --wait` |
| Test inference | `python main.py test --endpoint-name fraud-detector-endpoint --num-samples 100` |
| Simulate ground truth | `python -m src.drift_monitoring.simulate_ground_truth_from_athena --accuracy 0.85` |
| Apply ground truth | `python -m src.drift_monitoring.update_ground_truth --mode batch` |
| Monitor performance | `python -m src.drift_monitoring.monitor_model_performance --days 30` |

## Troubleshooting

### Lifecycle script didn't complete

The lifecycle script downloads training data, uploads to S3, and creates Athena tables on first space launch. If logs in CloudWatch (`/aws/sagemaker/studio`) show a failure, re-run from the JupyterLab terminal:

```bash
cd ~/sample-mlops-bestpractices/sagemaker-automated-drift-and-trend-monitoring
source .env
pip install -e .
python -m src.setup.download_kaggle_dataset
python -m src.setup.upload_data_to_s3
python -m src.setup.setup_athena_tables
```

Verify:
```bash
aws s3 ls s3://${DATA_S3_BUCKET}/fraud-detection/data/
aws athena start-query-execution --query-string "SHOW TABLES IN fraud_detection" \
  --result-configuration "OutputLocation=s3://${DATA_S3_BUCKET}/athena-results/" \
  --region ${AWS_DEFAULT_REGION}
```

### MLflow tracking URI errors

The SDK requires the **ARN** format (not the HTTPS UI URL). The CloudFormation stack writes the correct ARN to `.env`:
```
MLFLOW_TRACKING_URI=arn:aws:sagemaker:<region>:<account>:mlflow-app/app-<id>
```

To open the MLflow UI: SageMaker Studio → Partner AI Apps → MLflow.

### Pipeline fails with permission errors

If you used `UseExistingRole=true` or modified the role, re-apply the CloudFormation-managed permissions:
```bash
python -m src.setup.create_or_update_sagemaker_role
```

### No records in Athena after inference

The custom handler sends predictions to SQS; Lambda batches and writes to Athena every 10 messages or 30 seconds. If `inference_responses` is empty after >1 minute and 10+ predictions:
1. Check the Lambda logs: `aws logs tail /aws/lambda/fraud-detection-monitoring-inference-logger`
2. Verify the endpoint uses the custom handler (not the built-in XGBoost serving)
3. Confirm SQS queue URL is set in the endpoint environment (`SQS_QUEUE_URL`)

### Drift Lambda timing out or missing data

The drift Lambda compares the last 7 days of inference data against the frozen `training_data` baseline for **data drift** and the last 30 days against the frozen `evaluation_data` baseline for **model drift** (both resolved from the deployed model's registered `baseline.json` via Iceberg snapshots). If inference volume is low, override the lookback windows:
```python
lambda_client.update_function_configuration(
    FunctionName='fraud-detection-monitoring-drift-monitor',
    Environment={'Variables': {
        'DATA_DRIFT_LOOKBACK_DAYS': '14',
        'MIN_SAMPLES_FOR_DRIFT': '50',
    }}
)
```

## License

MIT — see [LICENSE](LICENSE).
