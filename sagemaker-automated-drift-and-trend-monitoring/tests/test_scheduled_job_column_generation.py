"""Unit tests for scheduled-job Lambda column generation (Requirement 15,
Task 18.3).

`setup_scheduled_inference.py` and `setup_scheduled_batch_transform.py`
generate the Lambda source code they will deploy to AWS by splicing
schema-derived column lists into a `LAMBDA_CODE_TEMPLATE` string at
setup-script run time. The rendered `LAMBDA_CODE` string is then zipped
and uploaded to Lambda — the deployed function itself never imports
`src.config.schema` (Requirement 15.4 / design change 8), so those
column values must be:

  1. Present and correct in the rendered source at setup time.
  2. Baked in as plain Python literals (a `repr()` of the feature list)
     rather than a runtime import — otherwise the deployed Lambda's
     cold-start fails with ModuleNotFoundError on `src.config.schema`.
  3. Include the identifier and timestamp columns for the inference
     Lambda's Athena SELECT (schema.identifier_column() and
     schema.timestamp_column() are intentionally NOT part of
     feature_names() — see dataset_schema.yaml).

These tests inspect the rendered `LAMBDA_CODE` string module-level
attribute (already generated at import time) — no real Lambda deploy,
no zip write, no boto3.

Validates: Requirements 15.1, 15.2, 15.3, 15.4
"""

from __future__ import annotations

import re

import pytest

from src.config import schema
from src.setup import setup_scheduled_inference as ssi
from src.setup import setup_scheduled_batch_transform as ssb


# --- setup_scheduled_inference.py -----------------------------------------


def test_inference_lambda_code_contains_all_schema_features() -> None:
    """The rendered Lambda source must contain every feature name from
    schema.feature_names(). If a feature is missing, the deployed
    Lambda's endpoint invocation gets fed a truncated feature vector
    and returns bogus predictions.
    """
    features = schema.feature_names()
    for feat in features:
        assert feat in ssi.LAMBDA_CODE, (
            f"setup_scheduled_inference: Lambda source is missing "
            f"feature {feat!r}. The deployed Lambda would then invoke "
            f"the endpoint with a wrong-shape feature vector."
        )


def test_inference_lambda_code_contains_identifier_and_timestamp_columns() -> None:
    """The Athena SELECT built by setup_scheduled_inference.py must
    include the identifier and timestamp columns (in addition to the
    feature list). Those two aren't part of feature_names() by design
    but are needed for row-level joining and lookback filtering.
    """
    assert schema.identifier_column() in ssi.LAMBDA_CODE, (
        "Lambda source must SELECT the identifier column so predictions "
        "can be joined back to the source table."
    )
    assert schema.timestamp_column() in ssi.LAMBDA_CODE, (
        "Lambda source must SELECT the timestamp column — the lookback "
        "filter compares against it."
    )


def test_inference_lambda_code_bakes_feature_list_as_python_literal() -> None:
    """`feature_columns` inside the rendered Lambda must be a plain
    Python list literal — matching `repr(schema.feature_names())`.

    If a future refactor tries `from schema import feature_names` inside
    the Lambda, that import fails at cold-start because the deployed zip
    doesn't ship schema.py or dataset_schema.yaml (design change 8).
    Baking the value in as a literal is what makes the Lambda
    dependency-free.
    """
    expected_literal = repr(schema.feature_names())
    assert expected_literal in ssi.LAMBDA_CODE, (
        f"setup_scheduled_inference must splice repr(schema.feature_names()) "
        f"into the Lambda source; got no match for the expected literal."
    )


def test_inference_lambda_code_has_no_schema_import() -> None:
    """The deployed Lambda must not import `schema` — the zip that gets
    uploaded doesn't ship schema.py or dataset_schema.yaml. Any
    reintroduction of a schema import silently breaks cold-start on the
    next redeploy.
    """
    assert not re.search(r"^\s*import\s+schema\b", ssi.LAMBDA_CODE, re.MULTILINE), (
        "The rendered Lambda source contains `import schema` — but the "
        "deployed zip has no schema.py to import from. Bake the values "
        "in via string splicing at setup-script run time instead."
    )
    assert not re.search(
        r"from\s+src\.config\s+import\s+schema", ssi.LAMBDA_CODE
    ), (
        "The rendered Lambda source contains `from src.config import "
        "schema` — same problem: the deployed zip has no src.config "
        "package."
    )
    assert not re.search(
        r"from\s+src\.config\.schema\s+import", ssi.LAMBDA_CODE
    )


def test_inference_lambda_code_placeholders_were_replaced() -> None:
    """Both placeholders in LAMBDA_CODE_TEMPLATE must have been replaced
    when LAMBDA_CODE was rendered — a stray placeholder means the
    deploy shipped a Lambda that still contains the token
    `__..._PLACEHOLDER__` at the SELECT / feature-list site, and the
    Lambda would then fail at first execution.
    """
    assert "__SELECT_COLUMNS_PLACEHOLDER__" not in ssi.LAMBDA_CODE, (
        "SELECT columns placeholder was never spliced out."
    )
    assert "__FEATURE_COLUMNS_PLACEHOLDER__" not in ssi.LAMBDA_CODE, (
        "FEATURE_COLUMNS placeholder was never spliced out."
    )


# --- setup_scheduled_batch_transform.py -----------------------------------


def test_batch_lambda_code_contains_all_schema_features() -> None:
    """The batch-transform Lambda's CTAS SELECT must contain every
    schema feature name. Missing features would leak into the batch
    transform input file as either a wrong-shape row or a NULL column.
    """
    features = schema.feature_names()
    for feat in features:
        assert feat in ssb.LAMBDA_CODE, (
            f"setup_scheduled_batch_transform: Lambda source is missing "
            f"feature {feat!r} — the CTAS query would then omit it "
            f"from the batch input CSV."
        )


def test_batch_lambda_code_bakes_feature_list_as_python_literal() -> None:
    """Same invariant as the inference Lambda: feature_columns must be
    a repr()-produced literal, not a runtime import.
    """
    expected_literal = repr(schema.feature_names())
    assert expected_literal in ssb.LAMBDA_CODE


def test_batch_lambda_code_has_no_schema_import() -> None:
    """Deployed batch Lambda must not import schema."""
    assert not re.search(r"^\s*import\s+schema\b", ssb.LAMBDA_CODE, re.MULTILINE)
    assert not re.search(r"from\s+src\.config\s+import\s+schema", ssb.LAMBDA_CODE)
    assert not re.search(r"from\s+src\.config\.schema\s+import", ssb.LAMBDA_CODE)


def test_batch_lambda_code_placeholder_was_replaced() -> None:
    assert "__FEATURE_COLUMNS_PLACEHOLDER__" not in ssb.LAMBDA_CODE


# --- Consistency between the two setup scripts ----------------------------


def test_both_lambdas_agree_on_feature_list() -> None:
    """A future refactor mustn't let inference and batch drift out of
    sync on the feature list they bake in — both must use the same
    `repr(schema.feature_names())` literal.
    """
    lit = repr(schema.feature_names())
    assert lit in ssi.LAMBDA_CODE
    assert lit in ssb.LAMBDA_CODE
    # The exact string being present in both is the invariant. If
    # someone rewrote just one script to hardcode a different list, one
    # of these two assertions would fail.


def test_module_level_feature_columns_literal_is_repr_of_schema() -> None:
    """The module-level constants that get spliced in must equal
    repr(schema.feature_names()). Guards against a refactor that leaves
    the LAMBDA_CODE right but breaks the intermediate literal (which is
    also imported by callers wanting the same feature list).
    """
    expected = repr(schema.feature_names())
    assert ssi.FEATURE_COLUMNS_LITERAL == expected
    assert ssb.FEATURE_COLUMNS_LITERAL == expected
