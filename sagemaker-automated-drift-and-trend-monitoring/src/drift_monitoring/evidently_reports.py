"""
Evidently-based drift detection and model performance reporting.

Replaces custom matplotlib visualizations with Evidently's built-in
interactive HTML reports for data drift and classification metrics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from evidently import Report
from evidently.core.datasets import BinaryClassification, DataDefinition, Dataset
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset

logger = logging.getLogger(__name__)


def run_data_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset report comparing baseline vs current data.

    Args:
        baseline_df: Reference/training data (numeric features only).
        current_df: Current/inference data (same columns as baseline).
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot (call .save_html() etc.)
            - 'drift_detected': bool, whether overall drift was detected
            - 'drifted_columns_count': number of drifted columns
            - 'drifted_columns_share': share of drifted columns
            - 'per_column': dict mapping column name -> {'drift_score': float, 'drifted': bool}
    """
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=baseline_df, current_data=current_df)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(output_path)
        logger.info(f"Data drift report saved to {output_path}")

    # Extract structured results from the report dict
    metrics_list = snapshot.dict().get("metrics", [])
    result: Dict[str, Any] = {
        "snapshot": snapshot,
        "drift_detected": False,
        "drifted_columns_count": 0,
        "drifted_columns_share": 0.0,
        "per_column": {},
    }

    drift_share_threshold = 0.5  # default from DriftedColumnsCount

    for m in metrics_list:
        name = m.get("metric_name", "")
        config = m.get("config", {})
        value = m.get("value")

        if "DriftedColumnsCount" in name:
            count = value.get("count", 0) if isinstance(value, dict) else 0
            share = value.get("share", 0.0) if isinstance(value, dict) else 0.0
            drift_share_threshold = config.get("drift_share", 0.5)
            result["drifted_columns_count"] = int(count)
            result["drifted_columns_share"] = float(share)
            result["drift_detected"] = share >= drift_share_threshold

        elif "ValueDrift" in name:
            col = config.get("column", "unknown")
            threshold = float(config.get("threshold", 0.05))
            method = config.get("method", "")
            drift_score = float(value) if value is not None else 1.0

            # Evidently picks the test per-column based on sample size and
            # column type. The comparison direction depends on which test:
            #   * p-value tests (KS, Chi-Square)   → drift when score < threshold
            #   * distance / divergence tests      → drift when score > threshold
            #     (Wasserstein, PSI, Jensen-Shannon, Hellinger, TVD, ...)
            # Evidently exposes the chosen test in `config.method` but has no
            # boolean drift flag in the per-metric dict — we compute it here.
            is_p_value = "p_value" in method.lower() or "p-value" in method.lower()
            drifted = drift_score < threshold if is_p_value else drift_score > threshold

            # drift_magnitude is a test-agnostic "how far past the threshold":
            #   1.0 = at threshold, >1.0 = drifted, higher = more drifted
            # Callers can sort by this descending to get "top N drifted" without
            # caring which test was used for which column.
            if is_p_value:
                # p-values: smaller = more drifted → invert ratio
                drift_magnitude = (threshold / drift_score) if drift_score > 0 else float("inf")
            else:
                drift_magnitude = (drift_score / threshold) if threshold > 0 else float("inf")

            result["per_column"][col] = {
                "drift_score": drift_score,
                "drifted": drifted,
                "method": method,
                "threshold": threshold,
                "drift_magnitude": drift_magnitude,
            }

    logger.info(
        f"Data drift report: {result['drifted_columns_count']} drifted columns "
        f"({result['drifted_columns_share']:.1%}), overall drift: {result['drift_detected']}"
    )
    return result


def run_classification_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str = "target",
    prediction_column: str = "prediction",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Evidently ClassificationPreset report comparing baseline vs current model performance.

    Both DataFrames must contain the target and prediction columns.

    ⚠ Evidently quirk — both class labels (0 AND 1) must appear in BOTH
    ``baseline_df`` and ``current_df``. If either side has only one class
    (common when sampling a highly-imbalanced eval set with a flat LIMIT),
    ClassificationPreset raises ``KeyError: '0'`` deep inside Evidently's
    metric parser. Stratify the baseline by class label (e.g. UNION ALL
    of N fraud + N non-fraud rows) to guarantee both classes appear.
    https://github.com/evidentlyai/evidently/issues  (see ClassMetric parsing)

    Args:
        baseline_df: Reference data with target and prediction columns. Must
            contain BOTH class labels (0 and 1).
        current_df: Current data with target and prediction columns. Must
            also contain BOTH class labels.
        target_column: Name of the ground-truth label column.
        prediction_column: Name of the predicted label column.
        output_path: If provided, saves the HTML report to this path.

    Returns:
        Dictionary with:
            - 'snapshot': The Evidently report snapshot
            - 'metrics': Raw metrics dict extracted from the report

    Raises:
        ValueError: If either dataframe lacks both class labels (caught
            early before Evidently crashes with a confusing KeyError).
    """
    # Pre-flight check — Evidently's KeyError is unhelpful; fail loudly here.
    # Both `target` AND `prediction` columns must contain BOTH classes (0 AND 1)
    # in BOTH datasets. Evidently's `ClassificationQualityByClass` internally
    # calls sklearn's `classification_report` without `labels=` — sklearn then
    # omits classes that never appear as predictions, and Evidently crashes
    # with `KeyError: '0'` when it tries to read the omitted class.
    for label, df in (("baseline_df", baseline_df), ("current_df", current_df)):
        for col in (target_column, prediction_column):
            unique = sorted(df[col].dropna().unique().tolist())
            if len(unique) < 2:
                raise ValueError(
                    f"{label}.{col} has only {len(unique)} unique value(s): {unique}. "
                    f"Evidently ClassificationPreset needs BOTH 0 and 1 in BOTH "
                    f"target and prediction columns of BOTH datasets. "
                    f"If the model never predicted the minority class on this sample, "
                    f"either increase the sample size, lower the decision threshold, "
                    f"or stratify the upstream query to guarantee class diversity."
                )

    data_def = DataDefinition(
        classification=[
            BinaryClassification(
                target=target_column,
                prediction_labels=prediction_column,
            )
        ]
    )

    ref_dataset = Dataset.from_pandas(baseline_df, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(current_df, data_definition=data_def)

    report = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(output_path)
        logger.info(f"Classification report saved to {output_path}")

    result: Dict[str, Any] = {
        "snapshot": snapshot,
        "metrics": snapshot.dict().get("metrics", []),
    }

    logger.info("Classification report generated successfully")
    return result

