# Data Directory

This directory holds the training CSV dataset used by the fraud detection pipeline. The file is not checked into git — download it locally using the included script.

## Downloading the Training Data

**Prerequisites:**
```bash
pip install -e .   # installs kagglehub and other dependencies via pyproject.toml
```

```bash
python -m src.setup.download_kaggle_dataset
```

(Or call `ensure_training_data_ready()` from `notebooks/1_training_pipeline.ipynb` Cell 4 — same effect, idempotent.)

The training dataset (`creditcard_predictions_final.csv`) is downloaded from the real [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and columns are renamed to match the project's business-friendly schema. This ensures the model can learn real fraud patterns and achieve strong ROC-AUC scores.

Drifted data and ground truth are generated at runtime by `src/drift_monitoring/generate_drift_dataset.py` and `src/drift_monitoring/simulate_ground_truth_from_athena.py` respectively during the monitoring workflow.

## creditcard_predictions_final.csv

The primary training and inference dataset, sourced from the Kaggle credit card fraud dataset with renamed columns.

- Rows: 284,807
- Columns: 35 (28 PCA features renamed to business concepts + transaction_id, transaction_timestamp, transaction_amount, fraud_prediction, fraud_probability, customer_gender, is_fraud)
- Fraud rate: ~0.17% (highly imbalanced, realistic for credit card fraud)
- Features: PCA-normalised floats from real credit card transactions with genuine fraud patterns

When to use:
- Training the XGBoost model via SageMaker Pipelines (`notebooks/1_training_pipeline.ipynb`)
- Uploading to S3 and migrating into Athena Iceberg tables (`main.py setup --migrate-data`)
- Baseline data for drift detection comparisons
- Testing inference endpoints with representative data

## Column Reference

The 28 PCA feature columns are renamed from Kaggle's V1-V28 to business-friendly names:

| Feature Group | Columns |
|---|---|
| Transaction | transaction_hour, transaction_day_of_week, transaction_amount, transaction_type_code, transaction_timestamp |
| Customer | customer_age, customer_gender, customer_tenure_months, account_age_days |
| Geography | distance_from_home_km, distance_from_last_transaction_km, international_transaction, high_risk_country |
| Security | chip_transaction, pin_used, card_present, cvv_match, address_verification_match |
| Merchant | merchant_category_code, merchant_reputation_score |
| Behavior | num_transactions_24h, num_transactions_7days, avg_transaction_amount_30days, max_transaction_amount_30days, velocity_score, recurring_transaction, time_since_last_transaction_min, previous_fraud_incidents |
| Credit | credit_limit, available_credit_ratio |

## Reproducibility

The script uses a fixed random seed (`RANDOM_STATE = 42`) for the synthetic columns (fraud_prediction, fraud_probability, customer_gender) added to the Kaggle data, so repeated runs produce identical output.
