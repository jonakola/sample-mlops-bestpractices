# Evidently Report Screenshots

Screenshots captured from actual Evidently reports logged to MLflow during drift monitoring runs. The full interactive HTML reports are available as MLflow artifacts under `evidently_reports/`.

## Data Drift Report

![Data Drift Report](data-drift-report-screenshot.png)

Shows per-feature drift detection using Evidently's `DataDriftPreset`. By default the preset runs Kolmogorov–Smirnov for numerical features and chi-square for categoricals (configurable to Wasserstein / Jensen-Shannon / KL / PSI via the `EVIDENTLY_NUM_STAT_TEST` env var). Reference data comes from the frozen `training_data` Iceberg snapshot (`training_snapshot_id` in `baseline.json`); current data is the recent `inference_responses` window.

## Classification Performance Report

![Classification Performance Report](classification-performance-screenshot.png)

Shows model quality metrics (accuracy, precision, recall, F1), confusion matrices, and classification quality by label for current vs reference data. Reference data here comes from the frozen `evaluation_data` Iceberg snapshot (`evaluation_snapshot_id` in `baseline.json`) — its `is_fraud` + `fraud_prediction` columns provide the labeled baseline. Current data is `inference_responses` joined with simulated/real ground truth.

> **Note on Evidently's ClassificationPreset**: both the reference and current datasets must contain BOTH class labels (0 AND 1). On highly imbalanced data (fraud ≈ 0.2%), a flat `LIMIT N` query against the baseline may return only class 0 — Evidently then crashes with `KeyError: '0'`. The notebook works around this by stratifying the baseline pull (UNION ALL of N fraud + N non-fraud rows); the `run_classification_report` wrapper also pre-flights both classes present and raises a clear `ValueError` if not.
