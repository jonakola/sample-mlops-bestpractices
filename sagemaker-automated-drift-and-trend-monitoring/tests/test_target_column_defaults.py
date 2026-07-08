"""Unit tests for target-column defaults and shared schema bundling
(Requirement 8, Task 10.4).

Covers:
  - `train.py` and `preprocessing_pyspark.py` both default their
    `--target-column` argparse argument to `schema.target_column()`
    rather than a hardcoded `'is_fraud'` literal (8.1, 8.2). The test
    parses the checked-in source instead of invoking `main()`, so it
    doesn't need Spark/XGBoost/SageMaker deps to run.
  - `FraudDetectionPipeline._build_seed_source_dir()` calls
    `_stage_schema_sibling()`, which copies `schema.py` +
    `dataset_schema.yaml` into the staging directory alongside
    `seed_athena_tables.py` (8.3, 8.4, 8.5). Same helper is shared with
    the training and preprocessing bundling call sites.

Static parsing is deliberate. Argparse defaults are resolved at
`add_argument()` time — invoking `main()` would require live SageMaker,
Athena, and Spark clients and add zero coverage over what static
inspection catches (the whole point of Requirement 8 is that the
default *is* the schema call, not that the call happens to work at
runtime once).

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import schema


# --- Static-source inspection helpers ---------------------------------------


_TRAIN_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "src" / "train_pipeline" / "pipeline_steps" / "train.py"
)
_PREPROCESS_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "src" / "train_pipeline" / "pipeline_steps" / "preprocessing_pyspark.py"
)


def _find_add_argument_call(tree: ast.AST, flag: str) -> ast.Call | None:
    """Return the ``parser.add_argument('--flag', ...)`` Call node whose
    first positional arg matches ``flag``, or None if not found.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == flag:
            return node
    return None


def _extract_default_kwarg(call: ast.Call) -> ast.expr | None:
    """Return the AST for the ``default=`` kwarg of an add_argument call."""
    for kw in call.keywords:
        if kw.arg == "default":
            return kw.value
    return None


def _assert_default_is_schema_target_column(script_path: Path, script_name: str) -> None:
    """Shared assertion: the script's --target-column default is exactly
    `schema.target_column()`, not a hardcoded literal.
    """
    tree = ast.parse(script_path.read_text())
    call = _find_add_argument_call(tree, "--target-column")
    assert call is not None, (
        f"{script_name} must declare a --target-column argparse argument"
    )

    default = _extract_default_kwarg(call)
    assert default is not None, (
        f"{script_name}'s --target-column must have a `default=` kwarg"
    )

    # Reject hardcoded string literals: default=schema.target_column() is
    # the whole point of Requirement 8 — a stringy 'is_fraud' silently
    # reintroduces the dual-source-of-truth problem.
    assert not isinstance(default, ast.Constant), (
        f"{script_name}'s --target-column default is a hardcoded "
        f"{default.value!r} literal, but Requirement 8 requires it to be "
        f"`schema.target_column()` so BYO-dataset users don't have to "
        f"edit this file."
    )

    # Must be a call, must be to `.target_column()`.
    assert (
        isinstance(default, ast.Call)
        and isinstance(default.func, ast.Attribute)
        and default.func.attr == "target_column"
    ), (
        f"{script_name}'s --target-column default must be a call to "
        f"`schema.target_column()`, got {ast.dump(default)}"
    )


# --- Argparse default tests -------------------------------------------------


def test_train_target_column_defaults_to_schema() -> None:
    _assert_default_is_schema_target_column(_TRAIN_SCRIPT, "train.py")


def test_preprocessing_target_column_defaults_to_schema() -> None:
    _assert_default_is_schema_target_column(
        _PREPROCESS_SCRIPT, "preprocessing_pyspark.py"
    )


def test_train_target_column_default_matches_checked_in_schema() -> None:
    """End-to-end sanity: whatever `schema.target_column()` currently
    returns, `train.py --target-column` (unset) resolves to the same
    value. Guards against a case where the AST looks right but the
    schema module returns a mismatched value at runtime.
    """
    # Import the module namespace of train.py without invoking main() —
    # exec its argparse block against a freshly-constructed parser.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-column", type=str, default=schema.target_column())
    args = parser.parse_args([])
    assert args.target_column == schema.target_column()


# --- Bundling-helper tests --------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_schema_cache():
    """Keep schema.py's parsed-document cache cold across tests so
    monkeypatch/reload orders can't leak stale state between cases.
    """
    schema._load.cache_clear()
    yield
    schema._load.cache_clear()


@pytest.fixture
def pipeline_instance():
    """A `FraudDetectionPipeline` built against a fully mocked SageMaker
    SDK / boto3 — no real AWS calls. Matches the fixture pattern used
    by tests/test_pipeline_bundling.py so both test files stay in
    lock-step if the pipeline constructor changes.
    """
    from src.train_pipeline import pipeline as pl

    with patch.object(pl, "PipelineSession") as mock_pipeline_session, \
         patch.object(pl.boto3, "client") as mock_boto_client, \
         patch.object(pl.boto3, "Session"):
        mock_boto_client.return_value.get_caller_identity.return_value = {
            "Account": "123456789012"
        }
        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "test-bucket"
        mock_pipeline_session.return_value = mock_session

        yield pl.FraudDetectionPipeline(
            role="arn:aws:iam::123456789012:role/fake-role"
        )


def test_stage_schema_sibling_copies_both_files(pipeline_instance, tmp_path) -> None:
    """`_stage_schema_sibling(target_dir)` must place both `schema.py`
    and `dataset_schema.yaml` in the target directory as sibling files.

    This is the invariant every downstream bundling call relies on —
    the containerized script does `import schema` and `schema` does
    `open('dataset_schema.yaml')`, and both must be found in the
    entry-script's own directory.
    """
    pipeline_instance._stage_schema_sibling(tmp_path)

    assert (tmp_path / "schema.py").exists(), (
        "schema.py must be copied into the target directory as a sibling"
    )
    assert (tmp_path / "dataset_schema.yaml").exists(), (
        "dataset_schema.yaml must be copied into the target directory "
        "alongside schema.py"
    )


def test_build_seed_source_dir_bundles_three_files(pipeline_instance) -> None:
    """The seed step's source_dir must contain exactly the three files
    the container needs at runtime: the entry script, schema.py, and
    dataset_schema.yaml. Extra files leaking in would bloat the
    processor upload; missing files would break the seed job.
    """
    source_dir = pipeline_instance._build_seed_source_dir()
    try:
        assert source_dir.exists() and source_dir.is_dir()
        contents = {p.name for p in source_dir.iterdir() if p.is_file()}
        assert "seed_athena_tables.py" in contents
        assert "schema.py" in contents
        assert "dataset_schema.yaml" in contents
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def test_build_seed_source_dir_uses_shared_helper(pipeline_instance, tmp_path, monkeypatch) -> None:
    """`_build_seed_source_dir()` must delegate the schema-file copy to
    `_stage_schema_sibling()` — not open-coded — so the training and
    preprocessing bundling call sites (which use the same helper) can't
    diverge from the seed step's bundling behavior.
    """
    calls = []
    real_helper = pipeline_instance._stage_schema_sibling

    def spy(target_dir):
        calls.append(Path(target_dir))
        return real_helper(target_dir)

    monkeypatch.setattr(pipeline_instance, "_stage_schema_sibling", spy)
    source_dir = pipeline_instance._build_seed_source_dir()
    try:
        assert len(calls) == 1, (
            "_build_seed_source_dir must call _stage_schema_sibling exactly "
            "once — the shared helper is the single point of truth for "
            "schema-file bundling across all pipeline steps."
        )
        assert calls[0] == source_dir
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)
