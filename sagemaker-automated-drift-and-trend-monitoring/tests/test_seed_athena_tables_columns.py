"""Unit tests for `seed_athena_tables.py`'s schema-driven column ordering
and integrity-check behavior (Requirement 5, Task 6.1).

Covers:

- The `INSERT ... SELECT` staging table's column list is derived from
  `schema.csv_column_order()` rather than a hardcoded constant (5.1).
- The script imports its bundled sibling `schema` module (via
  `sys.path.insert` on its own directory) rather than importing
  `src.config.schema` by package path (5.2).
- The post-seed integrity check identifies rows using
  `schema.identifier_column()`/`schema.target_column()` rather than
  dataset-specific hardcoded column names (5.3).
- A failed integrity check raises `RuntimeError` directing the operator
  to inspect the predictions CSV in S3, exercised end-to-end through
  `main()` against a fully mocked Athena client (5.6).

`seed_athena_tables.py` is designed to run standalone inside a SageMaker
Processing container where `schema.py` has been bundled as a sibling file
(see `pipeline.py::_build_seed_source_dir`). To exercise that same
bundled-import path in a test environment without physically copying
`schema.py` into `pipeline_steps/`, we pre-populate `sys.modules['schema']`
with the real `src.config.schema` module before importing
`seed_athena_tables` — Python's import system consults `sys.modules`
before touching `sys.path`, so the module's `import schema` statement
resolves to it without needing a copy on disk.

_Requirements: 5.1, 5.2, 5.3, 5.6_
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.config import schema

# Make the bundled-sibling `import schema` statement in seed_athena_tables.py
# resolve to the real src.config.schema module without needing a physical
# copy on disk (see module docstring above).
sys.modules.setdefault("schema", schema)

from src.train_pipeline.pipeline_steps import seed_athena_tables as seed  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_schema_cache():
    """Keep schema.py's parsed-document cache cold across tests, mirroring
    the other schema-driven test modules in this suite."""
    schema._load.cache_clear()
    yield
    schema._load.cache_clear()


# ---------------------------------------------------------------------------
# 5.1 — CSV/staging column order is derived from schema.csv_column_order()
# ---------------------------------------------------------------------------


def test_csv_column_order_matches_schema_csv_column_order():
    """CSV_COLUMN_ORDER must equal schema.csv_column_order(), not a
    hardcoded constant."""
    assert seed.CSV_COLUMN_ORDER == schema.csv_column_order()
    assert seed.CSV_COLUMN_ORDER == (
        [schema.identifier_column()]
        + schema.feature_names()
        + [c.name for c in schema.auxiliary_columns()]
        + [schema.target_column()]
    )


def test_seed_tables_staging_table_uses_csv_column_order(monkeypatch):
    """The staging `CREATE EXTERNAL TABLE` DDL declares columns in exactly
    `CSV_COLUMN_ORDER`'s order."""
    cfg = _make_config()
    submitted: List[str] = []

    def _fake_run_query(cfg_arg, query, **kwargs):
        submitted.append(query)
        return []

    monkeypatch.setattr(seed, "_run_athena_query", _fake_run_query)

    seed.seed_tables(cfg)

    create_stmts = [q for q in submitted if "CREATE EXTERNAL TABLE" in q]
    assert len(create_stmts) == 1
    ddl = create_stmts[0]

    # Columns appear as "STRING, " separated declarations in CSV_COLUMN_ORDER.
    last_idx = -1
    for col in seed.CSV_COLUMN_ORDER:
        idx = ddl.find(f"{col} STRING")
        assert idx != -1, f"expected column {col!r} declared as STRING in staging DDL"
        assert idx > last_idx, f"column {col!r} out of CSV_COLUMN_ORDER position"
        last_idx = idx


# ---------------------------------------------------------------------------
# 5.2 — bundled sibling `schema` import, not `src.config.schema`
# ---------------------------------------------------------------------------


def test_script_imports_bundled_sibling_schema_not_src_config_schema():
    """The script's source imports a bare sibling `schema` module (staged
    alongside it in the container) rather than `src.config.schema` by
    package path."""
    source = Path(seed.__file__).read_text()

    assert "import schema" in source
    assert "from src.config import schema" not in source
    assert "import src.config.schema" not in source
    # It resolves the sibling via its own directory before importing.
    assert "sys.path.insert(0, str(Path(__file__).resolve().parent))" in source


def test_seed_module_schema_object_is_the_bundled_sibling():
    """The `schema` name the module actually calls into at runtime is the
    one resolved via the bundled-sibling import path (sys.modules lookup),
    and behaves identically to src.config.schema."""
    assert seed.schema is sys.modules["schema"]
    assert seed.schema.csv_column_order() == schema.csv_column_order()


# ---------------------------------------------------------------------------
# 5.3 — post-seed integrity check uses identifier_column()/target_column()
# ---------------------------------------------------------------------------


def test_id_and_target_column_constants_come_from_schema():
    assert seed.ID_COLUMN == schema.identifier_column()
    assert seed.TARGET_COLUMN == schema.target_column()


def test_check_one_table_queries_by_target_column(monkeypatch):
    """`_check_one_table()`'s GROUP BY integrity query groups by
    `schema.target_column()`, not a hardcoded column name."""
    cfg = _make_config()
    submitted: List[str] = []

    def _fake_run_query(cfg_arg, query, **kwargs):
        submitted.append(query)
        return [
            {seed.TARGET_COLUMN: "true", "n": "5"},
            {seed.TARGET_COLUMN: "false", "n": "10"},
        ]

    monkeypatch.setattr(seed, "_run_athena_query", _fake_run_query)

    result = seed._check_one_table(cfg, "training_data")

    assert len(submitted) == 1
    assert f"SELECT {seed.TARGET_COLUMN}, COUNT(*)" in submitted[0]
    assert f"GROUP BY {seed.TARGET_COLUMN}" in submitted[0]
    assert result == {"passed": True, "fraud_count": 5, "non_fraud_count": 10}


def test_verify_integrity_checks_both_training_and_evaluation_tables(monkeypatch):
    cfg = _make_config()
    checked_tables: List[str] = []

    def _fake_check(cfg_arg, table):
        checked_tables.append(table)
        return {"passed": True, "fraud_count": 1, "non_fraud_count": 1}

    monkeypatch.setattr(seed, "_check_one_table", _fake_check)

    result = seed.verify_integrity(cfg)

    assert checked_tables == [cfg.training_table, cfg.evaluation_table]
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# 5.6 — failed integrity check raises RuntimeError pointing at the S3 CSV
# ---------------------------------------------------------------------------


def test_main_raises_runtime_error_on_failed_post_seed_integrity(monkeypatch):
    """End-to-end through `main()` against a mocked Athena client: when the
    post-seed integrity check still fails, `main()` raises `RuntimeError`
    directing the operator to inspect the predictions CSV in S3."""
    _set_required_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["seed_athena_tables.py", "--force"])

    mock_athena = _make_mock_athena_client(
        # Only the "false" class is ever present -> integrity check always fails.
        group_by_rows=[{seed.TARGET_COLUMN: "false", "n": "3"}]
    )
    monkeypatch.setattr(seed.boto3, "client", lambda service_name, **kw: mock_athena)

    with pytest.raises(RuntimeError) as exc_info:
        seed.main()

    message = str(exc_info.value)
    assert "predictions" in message.lower()
    assert "s3" in message.lower() or "S3" in message


def test_main_succeeds_when_post_seed_integrity_passes(monkeypatch):
    """Sanity check complementing the failure case: a passing integrity
    check does not raise."""
    _set_required_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["seed_athena_tables.py", "--force"])

    mock_athena = _make_mock_athena_client(
        group_by_rows=[
            {seed.TARGET_COLUMN: "true", "n": "5"},
            {seed.TARGET_COLUMN: "false", "n": "5"},
        ]
    )
    monkeypatch.setattr(seed.boto3, "client", lambda service_name, **kw: mock_athena)

    seed.main()  # should not raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> "seed.Config":
    cfg = seed.Config.__new__(seed.Config)
    cfg.region = "us-east-1"
    cfg.database = "fraud_detection"
    cfg.output_s3 = "s3://bucket/athena-results/"
    cfg.training_table = "training_data"
    cfg.evaluation_table = "evaluation_data"
    cfg.bucket = "bucket"
    cfg.prefix = "fraud-detection/"
    return cfg


def _set_required_env(monkeypatch) -> None:
    env = {
        "AWS_DEFAULT_REGION": "us-east-1",
        "ATHENA_DATABASE": "fraud_detection",
        "ATHENA_OUTPUT_S3": "s3://bucket/athena-results/",
        "ATHENA_TRAINING_TABLE": "training_data",
        "ATHENA_EVALUATION_TABLE": "evaluation_data",
        "DATA_S3_BUCKET": "bucket",
        "DATA_S3_PREFIX": "fraud-detection/",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _make_mock_athena_client(group_by_rows: List[Dict[str, str]]) -> MagicMock:
    """A mocked boto3 Athena client where every query succeeds; queries
    that look like the integrity check's `GROUP BY` return
    `group_by_rows`, and every other query (DESCRIBE, DELETE, CREATE,
    DROP, INSERT) returns no rows.
    """
    athena = MagicMock()
    athena.start_query_execution.side_effect = lambda **kwargs: {
        "QueryExecutionId": "qid",
        "_query": kwargs["QueryString"],
    }

    def _get_query_execution(QueryExecutionId):
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    athena.get_query_execution.side_effect = _get_query_execution

    last_query_by_id: Dict[str, str] = {}

    real_start = athena.start_query_execution.side_effect

    def _start(**kwargs):
        result = real_start(**kwargs)
        last_query_by_id[result["QueryExecutionId"]] = kwargs["QueryString"]
        return result

    athena.start_query_execution.side_effect = _start

    def _get_paginator(name):
        assert name == "get_query_results"
        paginator = MagicMock()

        def _paginate(QueryExecutionId):
            query = last_query_by_id.get(QueryExecutionId, "")
            if "GROUP BY" in query:
                header = [seed.TARGET_COLUMN, "n"]
                rows = [{"Data": [{"VarCharValue": c} for c in header]}]
                for r in group_by_rows:
                    rows.append(
                        {"Data": [{"VarCharValue": r[seed.TARGET_COLUMN]}, {"VarCharValue": r["n"]}]}
                    )
                return [{"ResultSet": {"Rows": rows}}]
            return [{"ResultSet": {"Rows": []}}]

        paginator.paginate.side_effect = _paginate
        return paginator

    athena.get_paginator.side_effect = _get_paginator
    return athena
