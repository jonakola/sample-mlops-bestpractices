"""Unit tests for `--force-recreate` and `drop_all_tables()` in
`src/setup/create_athena_tables.py`.

Covers:
  - `--force-recreate` logs a destructive-action warning naming the table
    count before dropping anything (Requirements 3.6, 3.7).
  - `drop_all_tables()` always issues a Glue `delete_table` force-purge
    after `DROP TABLE IF EXISTS`, regardless of the reported outcome of
    the `DROP TABLE` query (Requirement 3.3).

All boto3 clients are mocked/monkeypatched — no real AWS calls are made.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.setup import create_athena_tables as cat


# ---------------------------------------------------------------------------
# Requirement 3.7 / 3.3: --force-recreate logs a destructive-action warning
# naming the table count before dropping anything.
# ---------------------------------------------------------------------------


def test_force_recreate_logs_destructive_warning_before_drop(monkeypatch, caplog):
    """`main()` with --force-recreate must log a warning naming the table
    count, and that warning must already be present in the logs by the time
    `drop_all_tables()` is invoked (i.e. before any table is dropped)."""
    warning_already_logged_when_dropping = []

    def fake_drop_all_tables():
        # At the moment drop_all_tables() runs, the destructive warning
        # must already have been emitted.
        warning_already_logged_when_dropping.append(
            any(
                "force-recreate" in record.getMessage().lower()
                and str(len(cat.ALL_TABLE_NAMES)) in record.getMessage()
                for record in caplog.records
            )
        )

    monkeypatch.setattr(cat, "create_s3_bucket", MagicMock(return_value=True))
    monkeypatch.setattr(cat, "create_database", MagicMock(return_value=True))
    monkeypatch.setattr(cat, "drop_all_tables", fake_drop_all_tables)
    monkeypatch.setattr(
        cat, "create_all_tables",
        MagicMock(return_value={t: True for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr(
        cat, "verify_all_tables",
        MagicMock(return_value={t: {"exists": True} for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr(
        "sys.argv", ["create_athena_tables.py", "--force-recreate", "--skip-s3"]
    )

    with caplog.at_level(logging.WARNING, logger=cat.logger.name):
        exit_code = cat.main()

    assert exit_code == 0
    # drop_all_tables() was actually invoked exactly once.
    assert warning_already_logged_when_dropping == [True]

    warning_records = [
        r for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert warning_records, "Expected a destructive-action warning to be logged"
    combined = " ".join(r.getMessage() for r in warning_records)
    assert str(len(cat.ALL_TABLE_NAMES)) in combined
    assert "force-recreate" in combined.lower()


def test_no_force_recreate_never_logs_destructive_warning(monkeypatch, caplog):
    """Without --force-recreate, no destructive warning is logged and
    drop_all_tables() is never called."""
    drop_mock = MagicMock()
    monkeypatch.setattr(cat, "create_s3_bucket", MagicMock(return_value=True))
    monkeypatch.setattr(cat, "create_database", MagicMock(return_value=True))
    monkeypatch.setattr(cat, "drop_all_tables", drop_mock)
    monkeypatch.setattr(
        cat, "create_all_tables",
        MagicMock(return_value={t: True for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr(
        cat, "verify_all_tables",
        MagicMock(return_value={t: {"exists": True} for t in cat.ALL_TABLE_NAMES}),
    )
    monkeypatch.setattr("sys.argv", ["create_athena_tables.py", "--skip-s3"])

    with caplog.at_level(logging.WARNING, logger=cat.logger.name):
        cat.main()

    drop_mock.assert_not_called()
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("force-recreate" in r.getMessage().lower() for r in warning_records)


# ---------------------------------------------------------------------------
# Requirement 3.3: drop_all_tables() always issues a Glue delete_table
# force-purge after DROP TABLE IF EXISTS, regardless of the reported
# outcome of the DROP TABLE query.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_outcome", [True, False])
def test_drop_all_tables_always_force_purges_via_glue(monkeypatch, query_outcome):
    """Whether the DROP TABLE query is reported as succeeding or failing,
    drop_all_tables() must still call Glue's delete_table for every table,
    and must call it only after issuing the DROP TABLE IF EXISTS query for
    that same table."""
    call_order = []

    def fake_run_athena_query(sql, *, timeout=300):
        assert "DROP TABLE IF EXISTS" in sql
        call_order.append(("drop_table_sql", sql))
        return query_outcome

    glue_mock = MagicMock()
    glue_mock.exceptions.EntityNotFoundException = RuntimeError

    def fake_delete_table(DatabaseName, Name):
        call_order.append(("delete_table", Name))

    glue_mock.delete_table.side_effect = fake_delete_table

    def fake_boto3_client(service_name, region_name=None):
        assert service_name == "glue"
        return glue_mock

    monkeypatch.setattr(cat, "_run_athena_query", fake_run_athena_query)
    monkeypatch.setattr(cat.boto3, "client", fake_boto3_client)

    cat.drop_all_tables()

    # A DROP TABLE IF EXISTS query + a Glue delete_table force-purge for
    # every table this script owns, regardless of the reported DROP TABLE
    # outcome.
    assert glue_mock.delete_table.call_count == len(cat.ALL_TABLE_NAMES)
    drop_calls = [c for c in call_order if c[0] == "drop_table_sql"]
    delete_calls = [c for c in call_order if c[0] == "delete_table"]
    assert len(drop_calls) == len(cat.ALL_TABLE_NAMES)
    assert len(delete_calls) == len(cat.ALL_TABLE_NAMES)

    # For each table, the DROP TABLE query must precede the Glue delete_table
    # force-purge.
    for i in range(len(cat.ALL_TABLE_NAMES)):
        assert call_order[2 * i][0] == "drop_table_sql"
        assert call_order[2 * i + 1][0] == "delete_table"


def test_drop_all_tables_swallows_glue_entity_not_found(monkeypatch):
    """If Glue reports the table doesn't exist (EntityNotFoundException),
    drop_all_tables() must not propagate that error — the force-purge is
    best-effort per table."""

    class FakeEntityNotFoundException(Exception):
        pass

    glue_mock = MagicMock()
    glue_mock.exceptions.EntityNotFoundException = FakeEntityNotFoundException
    glue_mock.delete_table.side_effect = FakeEntityNotFoundException("missing")

    monkeypatch.setattr(cat, "_run_athena_query", MagicMock(return_value=True))
    monkeypatch.setattr(
        cat.boto3, "client", lambda service_name, region_name=None: glue_mock
    )

    # Should not raise.
    cat.drop_all_tables()

    assert glue_mock.delete_table.call_count == len(cat.ALL_TABLE_NAMES)
