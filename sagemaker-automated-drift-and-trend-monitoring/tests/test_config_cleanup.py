"""Guardrail tests for the config-cleanup step (Requirement 13, Task 16.2).

`TRAINING_FEATURES` and `TARGET_COLUMN` were removed from
`src.config.config` when the codebase switched to being driven by
`src/config/dataset_schema.yaml` (Requirements 13.1, 13.2). These tests
assert those attributes stay gone — a regression here would silently
re-introduce dual sources of truth for the feature list and target
column and let code drift back to hardcoded values.

The tests also assert the *replacement* accessors are wired up correctly
(`src.config.schema.feature_names()` / `schema.target_column()`), so a
would-be re-adder gets an explicit "use the schema module instead"
signal from the failing test rather than a mysterious NameError deep in
downstream code.

Validates: Requirements 13.1, 13.2, 13.3
"""

from __future__ import annotations

import pytest

from src.config import config, schema


def test_config_has_no_training_features_attribute() -> None:
    """The retired `TRAINING_FEATURES` constant must not reappear."""
    assert not hasattr(config, "TRAINING_FEATURES"), (
        "config.TRAINING_FEATURES was removed as part of Requirement 13.1 — "
        "downstream code should call src.config.schema.feature_names() "
        "instead. Reintroducing this constant re-creates the dual-source-of-"
        "truth problem the schema-driven refactor exists to solve."
    )


def test_config_has_no_target_column_attribute() -> None:
    """The retired `TARGET_COLUMN` constant must not reappear."""
    assert not hasattr(config, "TARGET_COLUMN"), (
        "config.TARGET_COLUMN was removed as part of Requirement 13.2 — "
        "downstream code should call src.config.schema.target_column() "
        "instead."
    )


def test_replacement_accessors_are_wired_up() -> None:
    """The schema-driven accessors are the intended replacement path.

    Guards against a "removed the constant but forgot to add the
    replacement" regression that would leave callers without a working
    substitute.
    """
    features = schema.feature_names()
    assert isinstance(features, list) and len(features) > 0, (
        "schema.feature_names() must return a non-empty list — this is the "
        "replacement for the removed config.TRAINING_FEATURES."
    )
    assert all(isinstance(f, str) and f for f in features), (
        "schema.feature_names() entries must all be non-empty strings."
    )

    target = schema.target_column()
    assert isinstance(target, str) and target, (
        "schema.target_column() must return a non-empty string — this is "
        "the replacement for the removed config.TARGET_COLUMN."
    )


@pytest.mark.parametrize("attribute", ["TRAINING_FEATURES", "TARGET_COLUMN"])
def test_direct_import_from_config_raises_importerror(attribute: str) -> None:
    """`from src.config.config import TRAINING_FEATURES` must fail loudly.

    This is the specific failure mode a user gets if they re-add code
    written against the old API. The test pins it so we can be certain
    the ImportError happens at import time (not silently at
    attribute-access time deep inside a Lambda).
    """
    with pytest.raises(ImportError):
        exec(f"from src.config.config import {attribute}", {})
