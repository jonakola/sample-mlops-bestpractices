"""Unit tests for `create_athena_tables.py`'s maintenance helpers
(Requirement 11, Task 13.3).

These helpers migrated in from the deleted `iceberg_manager.py` module:

  - `get_table_stats()`   — row count + Glue metadata + is_iceberg /
                             is_partitioned flags
  - `verify_all_tables()` — presence + location + type for every table
                             in `ALL_TABLE_NAMES`
  - `optimize_table()`    — issues `OPTIMIZE ... REWRITE DATA USING BIN_PACK`
                             on Iceberg tables; no-ops on non-Iceberg
  - `vacuum_table()`      — issues Iceberg `VACUUM`; no-ops on non-Iceberg
  - `expire_snapshots()`  — calls the Iceberg `expire_snapshots` procedure;
                             no-ops on non-Iceberg
  - `table_exists()`      — Glue `get_table` wrapped in
                             EntityNotFoundException handling

The maintenance helpers are best-effort by design — they log warnings
and return False on failure rather than raising, so an operator running
"optimize the whole warehouse" gets a partial-success dict instead of
the first-error abort. The tests below pin that contract so a future
"just raise on any failure" refactor breaks loudly.

All boto3 clients are mocked — no real AWS calls, no dependency on a
running Athena warehouse.

Validates: Requirements 11.2, 11.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.setup import create_athena_tables as cat


# --- Shared mock helpers ---------------------------------------------------


class _EntityNotFoundException(Exception):
    """Stand-in for `glue.exceptions.EntityNotFoundException`."""


def _make_mock_glue(existing_tables: set[str] | None = None) -> MagicMock:
    """Glue client where `get_table` raises for anything not in
    ``existing_tables`` (defaults to "all tables exist").
    """
    glue = MagicMock()
    glue.exceptions.EntityNotFoundException = _EntityNotFoundException

    def get_table(DatabaseName: str, Name: str):
        if existing_tables is not None and Name not in existing_tables:
            raise _EntityNotFoundException()
        return {
            "Table": {
                "StorageDescriptor": {"Location": f"s3://bucket/{Name}/"},
                "TableType": "EXTERNAL_TABLE",
            }
        }

    glue.get_table.side_effect = get_table
    return glue


def _make_mock_athena(*, succeed: bool = True, count_value: int = 42) -> MagicMock:
    """Athena client whose `start_query_execution` + polling loop returns
    SUCCEEDED (or FAILED if ``succeed=False``) and whose
    `get_query_results` returns a single scalar `count_value`.
    """
    athena = MagicMock()
    athena.start_query_execution.return_value = {"QueryExecutionId": "q-1"}
    athena.get_query_execution.return_value = {
        "QueryExecution": {
            "Status": {
                "State": "SUCCEEDED" if succeed else "FAILED",
                "StateChangeReason": "" if succeed else "mock failure",
            }
        }
    }
    athena.get_query_results.return_value = {
        "ResultSet": {
            "Rows": [
                {"Data": [{"VarCharValue": "row_count"}]},   # header row
                {"Data": [{"VarCharValue": str(count_value)}]},
            ]
        }
    }
    return athena


@pytest.fixture
def mock_boto3_clients():
    """Patch `boto3.client(...)` to return the right mock per service.

    Every maintenance helper constructs its own client, so the patch has
    to route by service name.
    """
    athena = _make_mock_athena()
    glue = _make_mock_glue()

    def client(service: str, *args, **kwargs):
        if service == "athena":
            return athena
        if service == "glue":
            return glue
        raise AssertionError(f"unexpected boto3.client({service!r}) call")

    with patch.object(cat.boto3, "client", side_effect=client):
        yield {"athena": athena, "glue": glue}


# --- table_exists() --------------------------------------------------------


def test_table_exists_true_when_glue_returns_metadata() -> None:
    glue = _make_mock_glue()
    assert cat.table_exists(glue, "training_data") is True
    glue.get_table.assert_called_once()


def test_table_exists_false_when_glue_raises_not_found() -> None:
    glue = _make_mock_glue(existing_tables=set())
    assert cat.table_exists(glue, "training_data") is False


# --- verify_all_tables() ---------------------------------------------------


def test_verify_all_tables_reports_presence_per_table(mock_boto3_clients) -> None:
    """Every table in ALL_TABLE_NAMES gets a `{'exists': True, ...}`
    entry when Glue returns metadata for it.
    """
    result = cat.verify_all_tables()

    assert set(result.keys()) == set(cat.ALL_TABLE_NAMES), (
        "verify_all_tables must return a status entry for every table in "
        "ALL_TABLE_NAMES — missing keys would silently skip maintenance "
        "on real tables."
    )
    for table, info in result.items():
        assert info["exists"] is True, f"{table} should be reported as exists=True"
        assert info["location"].startswith("s3://")
        assert "table_type" in info


def test_verify_all_tables_reports_missing_tables() -> None:
    """Tables that Glue can't find must be reported as `exists=False`
    without their metadata fields — the caller uses this to trigger a
    create-if-missing path.
    """
    # Only training_data exists; everything else raises.
    only_training = _make_mock_glue(existing_tables={"training_data"})
    with patch.object(cat.boto3, "client", return_value=only_training):
        result = cat.verify_all_tables()

    assert result["training_data"]["exists"] is True
    for table in cat.ALL_TABLE_NAMES:
        if table == "training_data":
            continue
        assert result[table] == {"exists": False}, (
            f"{table} should report exists=False with no metadata fields"
        )


# --- get_table_stats() -----------------------------------------------------


def test_get_table_stats_returns_row_count_and_metadata(mock_boto3_clients) -> None:
    """The stats dict must include row_count from Athena and location /
    table_type from Glue, plus the is_iceberg/is_partitioned flags that
    let maintenance code decide whether Iceberg-specific commands apply.
    """
    stats = cat.get_table_stats("training_data")

    assert stats["table_name"] == "training_data"
    assert stats["database"] == cat.ATHENA_DATABASE
    assert stats["row_count"] == 42  # from _make_mock_athena default
    assert stats["location"] == "s3://bucket/training_data/"
    assert stats["table_type"] == "EXTERNAL_TABLE"
    assert stats["is_iceberg"] is True    # training_data is in ICEBERG_TABLES
    assert isinstance(stats["is_partitioned"], bool)


def test_get_table_stats_flags_non_iceberg_correctly(mock_boto3_clients) -> None:
    """The is_iceberg flag lets the maintenance code no-op Iceberg-only
    commands (OPTIMIZE/VACUUM/expire_snapshots) on non-Iceberg tables.
    Verifies the flag is derived from the ICEBERG_TABLES membership
    check, not hardcoded to True.
    """
    # Pick a table name that isn't in ICEBERG_TABLES if any exist,
    # otherwise this is a smoke test of the flag existing.
    non_iceberg = [t for t in cat.ALL_TABLE_NAMES if t not in cat.ICEBERG_TABLES]
    if not non_iceberg:
        pytest.skip("All configured tables are Iceberg; no non-Iceberg case to test")

    stats = cat.get_table_stats(non_iceberg[0])
    assert stats["is_iceberg"] is False


# --- optimize_table() ------------------------------------------------------


def test_optimize_table_no_ops_on_non_iceberg_table() -> None:
    """optimize_table must return False (not raise) when called on a
    table that isn't in ICEBERG_TABLES — protecting non-Iceberg tables
    from an OPTIMIZE that would fail at Athena parse time.
    """
    with patch.object(cat.boto3, "client") as mock_client:
        result = cat.optimize_table("some_non_iceberg_table_that_doesnt_exist")
    assert result is False
    # Must NOT issue an OPTIMIZE query when the guard rejects the table.
    mock_client.assert_not_called()


def test_optimize_table_issues_bin_pack_rewrite_on_iceberg_table(mock_boto3_clients) -> None:
    """The Iceberg OPTIMIZE ... REWRITE DATA USING BIN_PACK command
    must be issued exactly once and returns True on Athena success.
    """
    iceberg_table = cat.ICEBERG_TABLES[0]
    result = cat.optimize_table(iceberg_table)

    assert result is True
    athena = mock_boto3_clients["athena"]
    calls = athena.start_query_execution.call_args_list
    assert len(calls) == 1
    submitted_sql = calls[0].kwargs["QueryString"]
    assert "OPTIMIZE" in submitted_sql
    assert "REWRITE DATA USING BIN_PACK" in submitted_sql
    assert iceberg_table in submitted_sql


def test_optimize_table_returns_false_on_athena_failure() -> None:
    """When Athena reports FAILED, optimize_table must return False and
    NOT raise — it's best-effort maintenance and an operator running a
    warehouse-wide sweep gets a partial-success dict.
    """
    athena = _make_mock_athena(succeed=False)
    with patch.object(cat.boto3, "client", return_value=athena):
        result = cat.optimize_table(cat.ICEBERG_TABLES[0])
    assert result is False


# --- vacuum_table() --------------------------------------------------------


def test_vacuum_table_no_ops_on_non_iceberg_table() -> None:
    non_iceberg = [t for t in cat.ALL_TABLE_NAMES if t not in cat.ICEBERG_TABLES]
    if not non_iceberg:
        pytest.skip("All configured tables are Iceberg")
    with patch.object(cat.boto3, "client") as mock_client:
        assert cat.vacuum_table(non_iceberg[0]) is False
    mock_client.assert_not_called()


def test_vacuum_table_issues_vacuum_command(mock_boto3_clients) -> None:
    """Verifies the SQL contains VACUUM plus the older_than window."""
    iceberg_table = cat.ICEBERG_TABLES[0]
    result = cat.vacuum_table(iceberg_table, older_than_days=14)

    assert result is True
    submitted_sql = mock_boto3_clients["athena"].start_query_execution.call_args.kwargs["QueryString"]
    assert "VACUUM" in submitted_sql
    assert iceberg_table in submitted_sql
    assert "14 days ago" in submitted_sql, (
        "The older_than window must be spliced into the VACUUM SQL — "
        "regressing to a hardcoded default would silently ignore the "
        "caller's requested retention."
    )


# --- expire_snapshots() ----------------------------------------------------


def test_expire_snapshots_no_ops_on_non_iceberg_table() -> None:
    non_iceberg = [t for t in cat.ALL_TABLE_NAMES if t not in cat.ICEBERG_TABLES]
    if not non_iceberg:
        pytest.skip("All configured tables are Iceberg")
    with patch.object(cat.boto3, "client") as mock_client:
        assert cat.expire_snapshots(non_iceberg[0]) is False
    mock_client.assert_not_called()


def test_expire_snapshots_calls_iceberg_procedure(mock_boto3_clients) -> None:
    """The command must call the Iceberg `expire_snapshots` system
    procedure, not DROP or DELETE — those would destroy the metadata,
    not just prune it.
    """
    iceberg_table = cat.ICEBERG_TABLES[0]
    result = cat.expire_snapshots(iceberg_table, older_than_days=30)

    assert result is True
    submitted_sql = mock_boto3_clients["athena"].start_query_execution.call_args.kwargs["QueryString"]
    assert "expire_snapshots" in submitted_sql
    assert f"table_name => '{iceberg_table}'" in submitted_sql
    assert "30 days ago" in submitted_sql


# --- Iceberg / partitioned table catalogs ---------------------------------


def test_iceberg_and_partitioned_lists_are_subsets_of_all_tables() -> None:
    """Sanity: every table name in ICEBERG_TABLES / PARTITIONED_TABLES
    must also appear in ALL_TABLE_NAMES. A missed entry means the
    maintenance helpers would try to optimize a table that
    `create_all_tables()` doesn't manage.
    """
    all_names = set(cat.ALL_TABLE_NAMES)
    assert set(cat.ICEBERG_TABLES).issubset(all_names), (
        f"ICEBERG_TABLES contains entries not in ALL_TABLE_NAMES: "
        f"{set(cat.ICEBERG_TABLES) - all_names}"
    )
    assert set(cat.PARTITIONED_TABLES).issubset(all_names), (
        f"PARTITIONED_TABLES contains entries not in ALL_TABLE_NAMES: "
        f"{set(cat.PARTITIONED_TABLES) - all_names}"
    )


def test_get_iceberg_tables_and_get_partitioned_tables_return_module_lists() -> None:
    """The accessor functions must return the module-level lists, not a
    hardcoded literal. Guards against a future refactor that forgets to
    update the accessor when adding a table.
    """
    assert cat.get_iceberg_tables() == cat.ICEBERG_TABLES
    assert cat.get_partitioned_tables() == cat.PARTITIONED_TABLES
