"""Unit tests for `main()`'s exit codes in `src/setup/create_athena_tables.py`
(Requirement 3.8, Task 4.4).

`main()` returns 0 when every table is created/verified successfully and 1
when at least one table fails to create or fails verification. All boto3
usage is mocked/monkeypatched at the module-function level (`create_s3_bucket`,
`create_database`, `create_all_tables`, `verify_all_tables`) — no real AWS
calls are made.

Validates: Requirements 3.8
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.setup import create_athena_tables as cat


def _patch_common(monkeypatch, *, create_ok=True, create_results=None, verify_results=None):
    monkeypatch.setattr(cat, "create_s3_bucket", MagicMock(return_value=True))
    monkeypatch.setattr(cat, "create_database", MagicMock(return_value=create_ok))
    if create_results is None:
        create_results = {t: True for t in cat.ALL_TABLE_NAMES}
    if verify_results is None:
        verify_results = {t: {"exists": True} for t in cat.ALL_TABLE_NAMES}
    monkeypatch.setattr(cat, "create_all_tables", MagicMock(return_value=create_results))
    monkeypatch.setattr(cat, "verify_all_tables", MagicMock(return_value=verify_results))
    monkeypatch.setattr(cat, "drop_all_tables", MagicMock())
    monkeypatch.setattr("sys.argv", ["create_athena_tables.py", "--skip-s3"])


def test_main_returns_0_when_all_tables_created_and_verified(monkeypatch):
    """Requirement 3.8: main() returns 0 when every table is
    created/verified successfully."""
    _patch_common(monkeypatch)

    exit_code = cat.main()

    assert exit_code == 0


def test_main_returns_1_when_a_table_creation_fails(monkeypatch):
    """Requirement 3.8: main() returns 1 when at least one table fails to
    create, even if every table is later reported as existing."""
    create_results = {t: True for t in cat.ALL_TABLE_NAMES}
    create_results["drifted_data"] = False
    _patch_common(monkeypatch, create_results=create_results)

    exit_code = cat.main()

    assert exit_code == 1


def test_main_returns_1_when_a_table_fails_verification(monkeypatch):
    """Requirement 3.8: main() returns 1 when at least one table is
    reported missing during verification, even if create_all_tables()
    reported success for every table."""
    verify_results = {t: {"exists": True} for t in cat.ALL_TABLE_NAMES}
    verify_results["monitoring_responses"] = {"exists": False}
    _patch_common(monkeypatch, verify_results=verify_results)

    exit_code = cat.main()

    assert exit_code == 1


def test_main_returns_1_when_database_creation_fails(monkeypatch):
    """Requirement 3.8: main() returns 1 (and aborts before creating
    tables) when create_database() itself fails."""
    _patch_common(monkeypatch, create_ok=False)

    exit_code = cat.main()

    assert exit_code == 1
    cat.create_all_tables.assert_not_called()
