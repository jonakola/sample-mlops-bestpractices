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

Open `sample-mlops-bestpractices/sagemaker-automated-drift-and-trend-monitoring/notebooks/` and run these notebooks sequentially:

| # | Notebook | Purpose | Time |
|---|----------|---------|------|
| 1 | `1_training_pipeline.ipynb` | Builds and executes the SageMaker training pipeline: preprocessing → XGBoost training → evaluation (quality gate at ROC-AUC ≥ 0.70) → MLflow registration → endpoint deployment with custom Athena-logging handler. | ~25 min |
| 2 | `2_inference_monitoring.ipynb` | Tests the deployed endpoint, simulates ground truth, applies ground truth via Athena MERGE, runs Evidently data + model drift detection, logs interactive reports to MLflow, and sets up the daily-scheduled drift Lambda. | ~30 min |
| 3 | `3_governance_dashboard.ipynb` | Creates a QuickSight governance dashboard (datasource, datasets, analysis, published dashboard) with auto-refresh via EventBridge + Lambda. Requires QuickSight Enterprise subscription. | ~15 min |

**Optional notebooks** (run as needed):

| # | Notebook | Purpose |
|---|----------|---------|
| 4 | `4_optional_version_validation.ipynb` | Verifies MLflow model version matches the deployed endpoint and Athena inference logs (traceability check). |
| 5 | `5_optional_cleanup.ipynb` | Deletes all AWS resources created outside CloudFormation (Lambda functions, endpoints, SNS topics, dashboards, CloudWatch alarms). |
| 6 | `6_shap_explainability.ipynb` | Generates SHAP global and per-prediction feature importance plots for the trained model. |

> **Note:** Notebook 1 cell 2 runs `! uv pip install -e ../`. If you see `No virtual environment found`, the error message tells you how to resolve it (run `uv venv` or add `--system`).

## What This Solution Does

### Pipeline (Notebook 1)
1. **Preprocess** — Reads from Athena, validates, encodes categoricals, splits train/test (80/20)
2. **Train** — XGBoost with automatic class imbalance handling (`scale_pos_weight`)
3. **Evaluate** — Computes ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix
4. **Quality Gate** — Registers the model only if ROC-AUC ≥ 0.70
5. **Deploy** — Serverless endpoint with a custom inference handler that logs every prediction to SQS → Lambda → Athena (zero added latency)

### Monitoring (Notebook 2)
- **Inference logging** — Every prediction → SQS → Lambda batches (10 msgs or 30s) → `inference_responses` Iceberg table
- **Ground truth integration** — Simulated (dev) or fed from fraud investigation systems (prod) → `ground_truth_updates` table → MERGE into `inference_responses`
- **Drift detection** — Evidently `DataDriftPreset` (KS test for numerics, chi-square for categoricals, PSI per feature) + `ClassificationPreset` (ROC, PR, confusion matrix). Configurable via `src/config/config.yaml`.
- **Automated daily checks** — EventBridge → Lambda (`fraud-detection-drift-monitor`) at 2 AM UTC. Logs metrics + interactive HTML reports to MLflow, writes summary to `monitoring_responses` table, sends SNS alert if drift exceeds thresholds.

### Governance (Notebook 3)
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
│   ├── 2_inference_monitoring.ipynb
│   ├── 3_governance_dashboard.ipynb
│   ├── 4_optional_version_validation.ipynb
│   ├── 5_optional_cleanup.ipynb
│   └── 6_shap_explainability.ipynb
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
| `training_data` | Iceberg | Training features (284K rows, 30 features) — drift baseline |
| `inference_responses` | Iceberg | All endpoint predictions, partitioned by day. `ground_truth` column populated via MERGE from `ground_truth_updates`. |
| `ground_truth_updates` | Iceberg | Lightweight patches: `inference_id` + `actual_fraud` + confirmation metadata |
| `monitoring_responses` | Iceberg | Output of each drift monitoring run (metrics, drift flags, sample sizes, MLflow run ID) |
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

The drift Lambda compares the last 7 days of inference data against the training baseline. If inference volume is low, override the lookback window:
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
