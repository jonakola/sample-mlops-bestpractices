"""Unit tests for `download_kaggle_dataset.py`'s column renaming and output
column order (Requirement 6, Task 7.1).

Covers:

- `KAGGLE_COLUMN_MAP` renames every raw Kaggle `creditcardfraud` column
  (`Time`, `V1`..`V28`, `Amount`, `Class`) to the project's business-friendly
  schema names (6.1).
- The transformed output CSV's column order equals
  `schema.csv_column_order()` — derived from `dataset_schema.yaml` — rather
  than a hardcoded list (6.2).

No real network/Kaggle calls are made: `kagglehub.dataset_download` is
monkeypatched to point at a small in-memory-generated temp CSV fixture that
mimics the raw Kaggle file's shape.

_Requirements: 6.1, 6.2_
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import schema
from src.setup import download_kaggle_dataset as dk


@pytest.fixture(autouse=True)
def _reset_schema_cache():
    """Keep schema.py's parsed-document cache cold across tests."""
    schema._load.cache_clear()
    yield
    schema._load.cache_clear()


def _raw_kaggle_columns() -> list[str]:
    """The exact raw column names present in the Kaggle creditcardfraud CSV."""
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def _make_raw_kaggle_df(n: int = 5) -> pd.DataFrame:
    """A small DataFrame shaped like the real Kaggle creditcardfraud CSV."""
    rng = np.random.default_rng(0)
    data = {"Time": np.arange(n, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Amount"] = rng.uniform(1, 500, size=n)
    data["Class"] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 6.1 — KAGGLE_COLUMN_MAP renames source columns correctly
# ---------------------------------------------------------------------------


def test_kaggle_column_map_keys_are_exactly_the_raw_kaggle_columns():
    """KAGGLE_COLUMN_MAP has one entry per raw Kaggle column, no more, no
    fewer."""
    assert set(dk.KAGGLE_COLUMN_MAP.keys()) == set(_raw_kaggle_columns())


def test_kaggle_column_map_renames_identifier_timestamp_and_target_sources():
    """The Kaggle columns that back the identifier/timestamp/target roles
    rename to exactly the names schema.py expects."""
    assert dk.KAGGLE_COLUMN_MAP["Time"] == schema.timestamp_column()
    assert dk.KAGGLE_COLUMN_MAP["Amount"] == "transaction_amount"
    assert dk.KAGGLE_COLUMN_MAP["Class"] == schema.target_column()


def test_kaggle_column_map_renames_v_columns_to_schema_feature_names():
    """Every V1..V28 anonymized column renames to a name that is a real
    feature in dataset_schema.yaml (e.g. V14 -> num_transactions_24h)."""
    feature_names = set(schema.feature_names())
    v_columns = [f"V{i}" for i in range(1, 29)]
    for v_col in v_columns:
        renamed = dk.KAGGLE_COLUMN_MAP[v_col]
        assert renamed in feature_names, (
            f"{v_col} renamed to {renamed!r}, which is not a schema feature"
        )
    assert dk.KAGGLE_COLUMN_MAP["V14"] == "num_transactions_24h"


def test_applying_column_map_produces_expected_renamed_columns():
    """Renaming an actual raw-shaped DataFrame with KAGGLE_COLUMN_MAP
    produces exactly the mapped column names."""
    raw = _make_raw_kaggle_df()
    renamed = raw.rename(columns=dk.KAGGLE_COLUMN_MAP)

    expected = {dk.KAGGLE_COLUMN_MAP[c] for c in raw.columns}
    assert set(renamed.columns) == expected
    # Values are unchanged by the rename (Time -> transaction_timestamp).
    assert list(renamed[schema.timestamp_column()]) == list(raw["Time"])


# ---------------------------------------------------------------------------
# 6.2 — output CSV column order equals schema.csv_column_order()
# ---------------------------------------------------------------------------


def test_csv_column_order_constant_matches_schema_not_a_hardcoded_list():
    """CSV_COLUMN_ORDER must equal schema.csv_column_order(), not a
    hardcoded constant."""
    assert dk.CSV_COLUMN_ORDER == schema.csv_column_order()
    assert dk.CSV_COLUMN_ORDER == (
        [schema.identifier_column()]
        + schema.feature_names()
        + [c.name for c in schema.auxiliary_columns()]
        + [schema.target_column()]
    )


def test_download_and_transform_writes_csv_in_schema_column_order(tmp_path, monkeypatch):
    """End-to-end (no network): download_and_transform() writes a local CSV
    whose header equals schema.csv_column_order(), using a fake kagglehub
    module and a small temp-file fixture standing in for the raw Kaggle
    download."""
    raw_df = _make_raw_kaggle_df(n=10)
    kaggle_download_dir = tmp_path / "kaggle_download"
    kaggle_download_dir.mkdir()
    raw_csv_path = kaggle_download_dir / "creditcard.csv"
    raw_df.to_csv(raw_csv_path, index=False)

    # Stand in for `import kagglehub` (imported lazily inside the function
    # under test) so no real network/Kaggle call is ever made.
    fake_kagglehub = types.ModuleType("kagglehub")
    fake_kagglehub.dataset_download = lambda handle: str(kaggle_download_dir)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    local_csv_path = tmp_path / "output" / "creditcard_predictions_final.csv"
    monkeypatch.setattr(dk, "_DATA_DIR", local_csv_path.parent)
    monkeypatch.setattr(dk, "LOCAL_CSV", local_csv_path)

    result_path = dk.download_and_transform()

    assert result_path == local_csv_path
    assert local_csv_path.exists()

    written = pd.read_csv(local_csv_path)
    assert list(written.columns) == schema.csv_column_order()
    assert list(written.columns) == dk.CSV_COLUMN_ORDER
    assert len(written) == len(raw_df)
