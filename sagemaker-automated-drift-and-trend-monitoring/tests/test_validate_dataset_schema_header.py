"""Unit tests for Schema_Validator's header validation in
`src/setup/validate_dataset_schema.py`.

Covers `validate()`:
  - Raises `SchemaValidationError` listing every missing column, every
    extra column, and every order mismatch when the source CSV's header
    differs from `schema.csv_column_order()`.
  - Returns normally (no exception) when the header matches exactly.

All S3 access is mocked — no real network/AWS calls are made. Each test
supplies a header-only (or single-row) CSV body via a fake S3 client so
`_validate_sample_types()` has nothing to flag; only header-validation
behavior is under test here.

**Validates: Requirements 4.1, 4.2, 4.6**
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.setup import validate_dataset_schema as vds

_TEST_CSV_S3_URI = "s3://test-bucket/test-prefix/data.csv"
_EXPECTED_ORDER = ["txn_id", "amount", "merchant", "aux_score", "is_fraud"]


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class _FakeS3Client:
    """Stands in for a boto3 S3 client. `get_object()` always returns the
    CSV text it was constructed with, regardless of the requested Range."""

    def __init__(self, csv_text: str):
        self._csv_text = csv_text
        self.exceptions = SimpleNamespace(NoSuchKey=Exception)

    def get_object(self, Bucket, Key, Range=None):
        return {"Body": _FakeBody(self._csv_text.encode("utf-8"))}


def _mock_s3(monkeypatch, csv_text: str):
    fake_client = _FakeS3Client(csv_text)
    monkeypatch.setattr(vds.boto3, "client", lambda *a, **k: fake_client)
    return fake_client


def _mock_expected_order(monkeypatch, expected=_EXPECTED_ORDER):
    monkeypatch.setattr(vds.schema, "csv_column_order", lambda: list(expected))


def _header_only_csv(columns) -> str:
    return ",".join(columns) + "\n"


def test_validate_returns_normally_when_header_matches_exactly(monkeypatch):
    _mock_expected_order(monkeypatch)
    _mock_s3(monkeypatch, _header_only_csv(_EXPECTED_ORDER))

    # Should not raise.
    vds.validate(csv_s3_uri=_TEST_CSV_S3_URI)


def test_validate_raises_and_lists_every_missing_column(monkeypatch):
    _mock_expected_order(monkeypatch)
    # Header is missing "merchant" and "aux_score".
    header = ["txn_id", "amount", "is_fraud"]
    _mock_s3(monkeypatch, _header_only_csv(header))

    with pytest.raises(vds.SchemaValidationError) as exc_info:
        vds.validate(csv_s3_uri=_TEST_CSV_S3_URI)

    message = str(exc_info.value)
    assert "merchant" in message
    assert "aux_score" in message
    assert "missing" in message.lower()


def test_validate_raises_and_lists_every_extra_column(monkeypatch):
    _mock_expected_order(monkeypatch)
    # Header has two columns not declared in the schema.
    header = _EXPECTED_ORDER + ["unexpected_col_1", "unexpected_col_2"]
    _mock_s3(monkeypatch, _header_only_csv(header))

    with pytest.raises(vds.SchemaValidationError) as exc_info:
        vds.validate(csv_s3_uri=_TEST_CSV_S3_URI)

    message = str(exc_info.value)
    assert "unexpected_col_1" in message
    assert "unexpected_col_2" in message
    assert "not declared" in message.lower()


def test_validate_raises_and_reports_order_mismatch_when_set_matches_but_order_differs(monkeypatch):
    _mock_expected_order(monkeypatch)
    # Same set of columns, but not in the expected order.
    header = list(reversed(_EXPECTED_ORDER))
    assert set(header) == set(_EXPECTED_ORDER)
    _mock_s3(monkeypatch, _header_only_csv(header))

    with pytest.raises(vds.SchemaValidationError) as exc_info:
        vds.validate(csv_s3_uri=_TEST_CSV_S3_URI)

    message = str(exc_info.value)
    assert "order differs" in message.lower()
    assert str(_EXPECTED_ORDER) in message
    assert str(header) in message


def test_validate_raises_and_lists_missing_extra_and_order_together(monkeypatch):
    _mock_expected_order(monkeypatch)
    # Drop "merchant" (missing), add "bonus_col" (extra), and reorder the
    # remaining columns so all three error categories fire at once.
    header = ["is_fraud", "bonus_col", "amount", "txn_id", "aux_score"]
    _mock_s3(monkeypatch, _header_only_csv(header))

    with pytest.raises(vds.SchemaValidationError) as exc_info:
        vds.validate(csv_s3_uri=_TEST_CSV_S3_URI)

    message = str(exc_info.value)
    assert "merchant" in message
    assert "missing" in message.lower()
    assert "bonus_col" in message
    assert "not declared" in message.lower()
