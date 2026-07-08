"""Unit tests for `--verify-only` and `--skip-s3` in
`src/setup/create_athena_tables.py`.

Covers:
  - `--verify-only` reports each table's existence state without creating
    or dropping any table (Requirement 3.4).
  - `--skip-s3` skips S3 bucket creation (Requirement 3.5).

All boto3 clients are mocked/monkeypatched — no real AWS calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.setup import create_athena_tables as cat


# ---------------------------------------------------------------------------
# Requirement 3.4: --verify-only reports each table's existence state
# without creating or dropping any table.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "existing_tables,expected_exit_code",
    [
        (set(cat.ALL_TABLE_NAMES), 0),  # all tables exist -> success
        (set(), 1),  # no tables exist -> failure
        ({cat.ALL_TABLE_NAMES[0]}, 1),  # partial -> failure
    ],
)
def test_verify_only_reports_state_without_mutating(
    monkeypatch, capsys, existing_tables, expected_exit_code
):
    """--verify-only must call verify_all_tables() and print each table's
    existence state, and must never call create_s3_bucket, create_database,
    create_all_tables, or drop_all_tables."""
    verify_result = {
        name: {"exists": name in existing_tables} for name in cat.ALL_TABLE_NAMES
    }
    verify_mock = MagicMock(return_value=verify_result)
    create_s3_mock = MagicMock()
    create_database_mock = MagicMock()
    create_all_tables_mock = MagicMock()
    drop_all_tables_mock = MagicMock()

    monkeypatch.setattr(cat, "verify_all_tables", verify_mock)
    monkeypatch.setattr(cat, "create_s3_bucket", create_s3_mock)
    monkeypatch.setattr(cat, "create_database", create_database_mock)
    monkeypatch.setattr(cat, "create_all_tables", create_all_tables_mock)
    monkeypatch.setattr(cat, "drop_all_tables", drop_all_tables_mock)
    monkeypatch.setattr("sys.argv", ["create_athena_tables.py", "--verify-only"])

    exit_code = cat.main()

    assert exit_code == expected_exit_code
    verify_mock.assert_called_once()
    create_s3_mock.assert_not_called()
    create_database_mock.assert_not_called()
    create_all_tables_mock.assert_not_called()
    drop_all_tables_mock.assert_not_called()

    captured = capsys.readouterr()
    for name in cat.ALL_TABLE_NAMES:
        expected_status = "✓ EXISTS" if name in existing_tables else "✗ MISSING"
        assert f"{expected_status}  {name}" in captured.out


# ---------------------------------------------------------------------------
# Requirement 3.5: --skip-s3 skips S3 bucket creation.
# ---------------------------------------------------------------------------


def test_skip_s3_flag_skips_bucket_creation(monkeypatch):
    """--skip-s3 must prevent create_s3_bucket() from being called, while
    still proceeding with database creation and table setup."""
    create_s3_mock = MagicMock()
    monkeypatch.setattr(cat, "create_s3_bucket", create_s3_mock)
    monkeypatch.setattr(cat, "create_database", MagicMock(return_value=True))
    monkeypatch.setattr(
        cat, "create_all_tables",
        MagicMock(return_value={t: True for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr(
        cat, "verify_all_tables",
        MagicMock(return_value={t: {"exists": True} for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr("sys.argv", ["create_athena_tables.py", "--skip-s3"])

    exit_code = cat.main()

    assert exit_code == 0
    create_s3_mock.assert_not_called()


def test_without_skip_s3_flag_creates_bucket(monkeypatch):
    """Without --skip-s3, create_s3_bucket() must be called."""
    create_s3_mock = MagicMock(return_value=True)
    monkeypatch.setattr(cat, "create_s3_bucket", create_s3_mock)
    monkeypatch.setattr(cat, "create_database", MagicMock(return_value=True))
    monkeypatch.setattr(
        cat, "create_all_tables",
        MagicMock(return_value={t: True for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr(
        cat, "verify_all_tables",
        MagicMock(return_value={t: {"exists": True} for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr("sys.argv", ["create_athena_tables.py"])

    exit_code = cat.main()

    assert exit_code == 0
    create_s3_mock.assert_called_once()
