"""Unit tests for `seed_athena_tables.py`'s per-column CAST expressions
(Requirement 5, Task 6.2).

Covers:

- `schema.cast_expr()` is applied to every feature column, every
  auxiliary column, and the target column in the generated
  `INSERT ... SELECT` list (5.5).

See `tests/test_seed_athena_tables_columns.py`'s module docstring for why
`sys.modules['schema']` is pre-populated with `src.config.schema` before
importing `seed_athena_tables` — the module under test does a
bundled-sibling `import schema` that would otherwise fail outside a
staged container directory.
"""

from __future__ import annotations

import sys
from typing import List

import pytest

from src.config import schema

# Make the bundled-sibling `import schema` statement in seed_athena_tables.py
# resolve to the real src.config.schema module without needing a physical
# copy on disk (see test_seed_athena_tables_columns.py's module docstring).
sys.modules.setdefault("schema", schema)

from src.train_pipeline.pipeline_steps import seed_athena_tables as seed  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_schema_cache():
    """Keep schema.py's parsed-document cache cold across tests, mirroring
    the other schema-driven test modules in this suite."""
    schema._load.cache_clear()
    yield
    schema._load.cache_clear()


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


def _run_seed_and_capture_inserts(monkeypatch) -> List[str]:
    """Runs `seed.seed_tables()` against a fully mocked
    `_run_athena_query` and returns the submitted `INSERT INTO ... SELECT`
    statements (one per target table)."""
    cfg = _make_config()
    submitted: List[str] = []

    def _fake_run_query(cfg_arg, query, **kwargs):
        submitted.append(query)
        return []

    monkeypatch.setattr(seed, "_run_athena_query", _fake_run_query)

    seed.seed_tables(cfg)

    inserts = [q for q in submitted if q.startswith("INSERT INTO")]
    assert len(inserts) == 2  # training_data + evaluation_data
    return inserts


def test_cast_expr_applied_to_every_feature_column(monkeypatch):
    """Every feature column's SELECT fragment matches
    `schema.cast_expr()` for that feature, not a bare column reference."""
    inserts = _run_seed_and_capture_inserts(monkeypatch)

    for insert_sql in inserts:
        for feature in seed.FEATURES:
            expected = schema.cast_expr(feature)
            assert expected in insert_sql, (
                f"expected cast_expr fragment {expected!r} for feature "
                f"{feature.name!r} in INSERT SELECT list"
            )


def test_cast_expr_applied_to_every_auxiliary_column(monkeypatch):
    """Every auxiliary column's SELECT fragment matches
    `schema.cast_expr()` for that column."""
    inserts = _run_seed_and_capture_inserts(monkeypatch)

    aux_columns = schema.auxiliary_columns()
    for insert_sql in inserts:
        for aux in aux_columns:
            expected = schema.cast_expr(aux)
            assert expected in insert_sql, (
                f"expected cast_expr fragment {expected!r} for auxiliary "
                f"column {aux.name!r} in INSERT SELECT list"
            )


def test_cast_expr_applied_to_target_column(monkeypatch):
    """The target column's SELECT fragment matches `schema.cast_expr()`
    applied to a Feature built from `TARGET_COLUMN`/`schema.target_type()`,
    not a hardcoded cast."""
    inserts = _run_seed_and_capture_inserts(monkeypatch)

    expected = schema.cast_expr(schema.Feature(seed.TARGET_COLUMN, schema.target_type()))
    for insert_sql in inserts:
        assert expected in insert_sql, (
            f"expected cast_expr fragment {expected!r} for target column "
            f"{seed.TARGET_COLUMN!r} in INSERT SELECT list"
        )


def test_select_list_has_no_uncasted_typed_column_reference(monkeypatch):
    """Sanity check: every non-string feature/aux/target column appears in
    the SELECT list only via its CAST(...) expression — never as a bare,
    uncasted column reference (which would silently keep it as STRING)."""
    inserts = _run_seed_and_capture_inserts(monkeypatch)

    typed_columns = [
        f for f in list(seed.FEATURES) + list(schema.auxiliary_columns())
        if f.type != "string"
    ]
    for insert_sql in inserts:
        for col in typed_columns:
            bare_ref = f", {col.name}, "
            assert bare_ref not in insert_sql, (
                f"column {col.name!r} appears uncasted in INSERT SELECT list"
            )
