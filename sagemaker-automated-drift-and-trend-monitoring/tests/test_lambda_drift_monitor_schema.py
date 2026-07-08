"""Unit tests for the Lambda drift monitor's schema integration
(Requirement 14, Task 17.3).

Covers:
  - `TRAINING_FEATURES` in `src/drift_monitoring/lambda_drift_monitor.py`
    resolves to exactly `schema.feature_names()` — not a hardcoded list
    (Requirement 14.1).
  - The baseline data-drift SQL's `SELECT` clause enumerates exactly
    `schema.feature_names()` in order, so the KS test compares training
    columns to inference columns of the same shape (Requirement 14.2).
  - The JSON `input_features` parse loop keys off exactly the schema
    feature names — extras in the JSON payload get silently ignored,
    missing keys don't crash (Requirement 14.1). This matches the
    real-world case where the endpoint's custom handler ships everything
    it knows about a prediction (identifiers, timestamps, derived
    features) and the drift monitor picks out only the model-input
    columns.
  - `Dockerfile.lambda` bundles `schema.py` and `dataset_schema.yaml`
    alongside the Lambda source (Requirement 14.3) — without this the
    Lambda would import a `schema` module that then couldn't find its
    YAML at runtime.

All boto3/AWS calls are mocked. The test does not actually invoke the
Lambda handler — the two things Requirement 14 pins are static
constructs (TRAINING_FEATURES at module load, SQL string interpolation)
that don't need a live event loop to validate.

Validates: Requirements 14.1, 14.2, 14.3
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import schema
from src.drift_monitoring import lambda_drift_monitor as ldm


# --- 14.1: TRAINING_FEATURES matches schema.feature_names() ----------------


def test_training_features_equals_schema_feature_names() -> None:
    """The Lambda's TRAINING_FEATURES constant must be the exact list
    schema.feature_names() returns — not a subset, superset, or any
    reordering. A drift in ordering would silently misalign the
    baseline SELECT with the current-window DataFrame columns.
    """
    assert ldm.TRAINING_FEATURES == schema.feature_names(), (
        "lambda_drift_monitor.TRAINING_FEATURES must equal "
        "schema.feature_names() exactly — divergence here breaks the "
        "baseline-vs-current column alignment the drift test depends on."
    )


def test_training_features_is_a_list_not_a_tuple_or_set() -> None:
    """Downstream code does `', '.join(TRAINING_FEATURES)` (order
    matters) and `for feat in TRAINING_FEATURES` — a set would break the
    former by producing non-deterministic SQL.
    """
    assert isinstance(ldm.TRAINING_FEATURES, list)


# --- 14.2: baseline SELECT enumerates schema.feature_names() ---------------


def test_baseline_sql_select_lists_all_schema_features() -> None:
    """The baseline SQL construction pattern used in
    `check_data_drift()`:

        SELECT {', '.join(TRAINING_FEATURES)}
        FROM {from_clause}
        WHERE is_fraud IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 5000

    ...must produce a SELECT list that contains every feature name,
    delimited by commas, in the order schema declares them. Test the
    pattern directly rather than invoking check_data_drift() (which
    reaches out to Athena / MLflow / SageMaker).
    """
    select_clause = ", ".join(ldm.TRAINING_FEATURES)
    for feature in schema.feature_names():
        assert feature in select_clause, (
            f"baseline SELECT clause is missing feature {feature!r}; "
            f"drift comparison would then have a shape mismatch between "
            f"baseline and current-window DataFrames."
        )

    # Column count must be exact — a superset introduces an unbound
    # column into the SQL that Athena will reject.
    parts = [p.strip() for p in select_clause.split(",")]
    assert len(parts) == len(schema.feature_names())
    assert parts == list(schema.feature_names()), (
        "The SELECT column ORDER must match schema.feature_names() — "
        "Evidently identifies columns by name so a reorder is harmless "
        "in theory, but keeping SQL stable across runs is easier to "
        "reason about at debug time."
    )


# --- 14.1 (behavioral): JSON-parse loop keys off schema features -----------


def test_input_features_parse_extracts_only_schema_features() -> None:
    """The parse loop in check_data_drift():

        parsed = {}
        for feat in TRAINING_FEATURES:
            if feat in features:
                parsed[feat] = float(features[feat])

    ...must extract exactly `schema.feature_names()` from a JSON
    payload, silently ignoring extra unknown keys. Test the loop's
    contract by exercising the same pattern.

    The custom inference handler emits everything it knows about a
    prediction (identifier, timestamp, derived features, raw JSON
    metadata). The drift monitor only cares about the model-input
    columns — anything else is noise.
    """
    import json

    # Payload with a superset of keys: all schema features + unknowns.
    payload = {feat: 1.5 for feat in schema.feature_names()}
    payload["some_unknown_extra_key"] = "irrelevant"
    payload["another_extra"] = 999
    payload_json = json.dumps(payload)

    features = json.loads(payload_json)
    parsed = {}
    for feat in ldm.TRAINING_FEATURES:
        if feat in features:
            parsed[feat] = float(features[feat])

    assert set(parsed.keys()) == set(schema.feature_names()), (
        "The parse loop must yield exactly schema.feature_names() — "
        "extras get dropped, no unexpected keys leak into the DataFrame."
    )
    assert "some_unknown_extra_key" not in parsed
    assert "another_extra" not in parsed


def test_input_features_parse_tolerates_missing_keys() -> None:
    """Missing feature keys must NOT crash the parse loop — the endpoint
    handler occasionally ships partial payloads (missing categoricals,
    schema evolution mid-migration) and the drift monitor still needs
    to progress on whatever samples it can parse. The final DataFrame
    will just have NaN for missing columns.
    """
    import json

    # Payload missing several features.
    partial = {feat: 1.0 for feat in schema.feature_names()[:5]}
    features = json.loads(json.dumps(partial))

    parsed = {}
    for feat in ldm.TRAINING_FEATURES:
        if feat in features:
            parsed[feat] = float(features[feat])

    # Only the first 5 features made it through.
    assert set(parsed.keys()) == set(schema.feature_names()[:5])
    # No exception was raised.


# --- 14.3: Dockerfile.lambda bundles schema + YAML -------------------------


_DOCKERFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "drift_monitoring" / "Dockerfile.lambda"
)


@pytest.fixture(scope="module")
def dockerfile_lines() -> list[str]:
    """The raw Dockerfile.lambda text, split by line for pattern matches."""
    return _DOCKERFILE_PATH.read_text().splitlines()


def test_dockerfile_copies_schema_module(dockerfile_lines: list[str]) -> None:
    """`schema.py` must be bundled into the Lambda image — without it,
    `import schema` inside the Lambda would fail at cold-start.
    """
    schema_copy = [ln for ln in dockerfile_lines
                   if re.search(r"COPY\s+.*schema\.py", ln)]
    assert schema_copy, (
        "Dockerfile.lambda must contain a COPY instruction for schema.py — "
        "otherwise the Lambda cold-start raises ModuleNotFoundError and "
        "every scheduled drift run fails."
    )


def test_dockerfile_copies_dataset_schema_yaml(dockerfile_lines: list[str]) -> None:
    """`dataset_schema.yaml` must be bundled — schema.py opens the YAML
    at accessor time, so shipping the module without the YAML would
    surface as `FileNotFoundError: dataset_schema.yaml` deep in the
    Lambda handler, not at cold-start.
    """
    yaml_copy = [ln for ln in dockerfile_lines
                 if re.search(r"COPY\s+.*dataset_schema\.yaml", ln)]
    assert yaml_copy, (
        "Dockerfile.lambda must contain a COPY instruction for "
        "dataset_schema.yaml — without it schema.feature_names() raises "
        "FileNotFoundError at first accessor call inside the Lambda."
    )


def test_dockerfile_still_copies_lambda_source(dockerfile_lines: list[str]) -> None:
    """Sanity: the pre-existing COPY for lambda_drift_monitor.py must
    survive alongside the new schema/YAML COPY lines. Guards against a
    "replaced" (rather than "added alongside") diff.
    """
    handler_copy = [ln for ln in dockerfile_lines
                    if re.search(r"COPY\s+.*lambda_drift_monitor\.py", ln)]
    assert handler_copy, (
        "Dockerfile.lambda must still COPY lambda_drift_monitor.py — the "
        "schema-bundling change is additive; if this COPY got removed the "
        "Lambda has no handler."
    )
