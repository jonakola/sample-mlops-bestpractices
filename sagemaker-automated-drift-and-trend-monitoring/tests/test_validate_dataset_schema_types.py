"""Unit tests for Schema_Validator's sample-type validation and range-read
sampling (Requirement 4, Task 5.2).

Covers `src/setup/validate_dataset_schema.py`'s:

1. `_validate_sample_types()` -- numeric-typed columns must have every
   sampled value parse as `float`; boolean-typed columns must have every
   sampled value case-insensitively match one of
   `{true, false, 1, 0, yes, no}`.
2. `_read_header_and_sample()` -- sampling issues a single S3 `get_object`
   range request for `bytes=0-1048576` (the first 1 MB) rather than
   downloading the full object, and defaults `sample_rows` to 20.

All S3/boto3 calls are mocked; no real network access occurs.

**Validates: Requirements 4.3, 4.4, 4.5, 4.7**
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from src.config.schema import Feature
from src.setup import validate_dataset_schema as vds


# ---------------------------------------------------------------------------
# _validate_sample_types(): numeric columns must parse as float
# ---------------------------------------------------------------------------


def _use_features(monkeypatch, features):
    """Point `schema.features()` (as seen by validate_dataset_schema.py) at
    a fixed, test-controlled feature list instead of the checked-in YAML."""
    monkeypatch.setattr(vds.schema, "features", lambda: features)


def test_numeric_column_with_non_numeric_sample_value_raises_error(monkeypatch):
    _use_features(monkeypatch, [Feature("amount", "double")])
    header = ["amount"]
    rows = [["12.5"], ["not-a-number"]]

    errors = vds._validate_sample_types(header, rows)

    assert len(errors) == 1
    assert "amount" in errors[0]
    assert "not-a-number" in errors[0]


@pytest.mark.parametrize("value", ["12.5", "-3", "0", "1e10", "3.0"])
def test_numeric_column_with_float_parseable_values_passes(monkeypatch, value):
    _use_features(monkeypatch, [Feature("amount", "double")])
    header = ["amount"]
    rows = [[value]]

    errors = vds._validate_sample_types(header, rows)

    assert errors == []


def test_numeric_column_with_int_type_also_requires_float_parseable_value(monkeypatch):
    """Requirement 4.3 applies to numeric types generally (`double`/`int`),
    not just `double`."""
    _use_features(monkeypatch, [Feature("count", "int")])
    header = ["count"]
    rows = [["abc"]]

    errors = vds._validate_sample_types(header, rows)

    assert len(errors) == 1
    assert "count" in errors[0]


# ---------------------------------------------------------------------------
# _validate_sample_types(): boolean columns, case-insensitive bool-like match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["true", "TRUE", "True", "false", "FALSE", "False", "1", "0", "yes", "YES", "no", "No"],
)
def test_boolean_column_accepts_case_insensitive_bool_like_values(monkeypatch, value):
    _use_features(monkeypatch, [Feature("is_fraud", "boolean")])
    header = ["is_fraud"]
    rows = [[value]]

    errors = vds._validate_sample_types(header, rows)

    assert errors == []


def test_boolean_column_with_non_bool_like_value_raises_error(monkeypatch):
    _use_features(monkeypatch, [Feature("is_fraud", "boolean")])
    header = ["is_fraud"]
    rows = [["maybe"]]

    errors = vds._validate_sample_types(header, rows)

    assert len(errors) == 1
    assert "is_fraud" in errors[0]
    assert "maybe" in errors[0]


def test_empty_sample_value_is_skipped_for_both_numeric_and_boolean(monkeypatch):
    _use_features(
        monkeypatch,
        [Feature("amount", "double"), Feature("is_fraud", "boolean")],
    )
    header = ["amount", "is_fraud"]
    rows = [["", ""]]

    errors = vds._validate_sample_types(header, rows)

    assert errors == []


def test_no_sample_rows_produces_no_errors(monkeypatch):
    _use_features(monkeypatch, [Feature("amount", "double")])

    errors = vds._validate_sample_types(["amount"], [])

    assert errors == []


# ---------------------------------------------------------------------------
# _read_header_and_sample(): range-read sampling and default sample_rows
# ---------------------------------------------------------------------------


def _fake_s3_client(csv_bytes: bytes) -> MagicMock:
    """A mocked boto3 S3 client whose `get_object()` returns `csv_bytes`
    regardless of the Range requested, so tests can assert on *what Range
    was requested* rather than relying on real partial-content semantics."""
    s3 = MagicMock()
    s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    body = MagicMock()
    body.read.return_value = csv_bytes
    s3.get_object.return_value = {"Body": body}
    return s3


def _make_csv_bytes(num_data_rows: int) -> bytes:
    lines = ["id,amount"]
    lines += [f"row{i},{i}.0" for i in range(num_data_rows)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_sampling_reads_only_first_1mb_via_range_request(monkeypatch):
    csv_bytes = _make_csv_bytes(5)
    fake_s3 = _fake_s3_client(csv_bytes)
    monkeypatch.setattr(vds.boto3, "client", lambda *a, **k: fake_s3)

    vds._read_header_and_sample("s3://my-bucket/my-key.csv")

    fake_s3.get_object.assert_called_once()
    call_kwargs = fake_s3.get_object.call_args.kwargs
    assert call_kwargs["Bucket"] == "my-bucket"
    assert call_kwargs["Key"] == "my-key.csv"
    assert call_kwargs["Range"] == "bytes=0-1048576"


def test_default_sample_rows_is_20(monkeypatch):
    # 25 data rows available; with the default sample_rows, only the first
    # 20 should be returned.
    csv_bytes = _make_csv_bytes(25)
    fake_s3 = _fake_s3_client(csv_bytes)
    monkeypatch.setattr(vds.boto3, "client", lambda *a, **k: fake_s3)

    header, rows = vds._read_header_and_sample("s3://my-bucket/my-key.csv")

    assert header == ["id", "amount"]
    assert len(rows) == 20
    assert rows[0] == ["row0", "0.0"]
    assert rows[-1] == ["row19", "19.0"]


def test_explicit_sample_rows_overrides_default(monkeypatch):
    csv_bytes = _make_csv_bytes(25)
    fake_s3 = _fake_s3_client(csv_bytes)
    monkeypatch.setattr(vds.boto3, "client", lambda *a, **k: fake_s3)

    _, rows = vds._read_header_and_sample("s3://my-bucket/my-key.csv", sample_rows=3)

    assert len(rows) == 3
