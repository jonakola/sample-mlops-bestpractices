"""Unit tests for `pipeline.py`'s schema bundling for the seed step and
target-column parameter wiring (Requirement 7, Task 8.1).

Covers:

- `_build_seed_source_dir()` copies `seed_athena_tables.py`, `schema.py`,
  and `dataset_schema.yaml` into a fresh temp dir (7.1).
- The seed step is built with a `FrameworkProcessor`, not a
  `ScriptProcessor` (7.2).
- The pipeline's `target_column` parameter default comes from
  `schema.target_column()` (7.3).
- The `ModelTrainer` hyperparameter resolves the target-column value from
  the pipeline parameter at execution time rather than a hardcoded
  literal (7.4).

The SageMaker SDK is mocked throughout — no real AWS calls are made.
`FraudDetectionPipeline.__init__` calls `boto3.client('sts')`,
`boto3.Session(...)`, and `PipelineSession(...).default_bucket()`, so
those are patched in every test via the `pipeline_instance` fixture.

_Requirements: 7.1, 7.2, 7.3, 7.4_
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import schema
from src.train_pipeline import pipeline as pl

# Importing pipeline.py transitively imports sagemaker's workflow pipeline
# module, which imports the unrelated third-party `schema` PyPI package
# (a data-validation library) as a side effect — polluting `sys.modules`
# under the bare name "schema". Other test modules in this suite (e.g.
# test_seed_athena_tables_columns.py) rely on `sys.modules.setdefault
# ("schema", <src.config.schema>)` to simulate seed_athena_tables.py's
# bundled-sibling import; if the third-party package claims that slot
# first, setdefault becomes a no-op. Pop it here so it doesn't leak into
# later-collected test modules regardless of collection order.
sys.modules.pop("schema", None)


@pytest.fixture(autouse=True)
def _reset_schema_cache():
    """Keep schema.py's parsed-document cache cold across tests."""
    schema._load.cache_clear()
    yield
    schema._load.cache_clear()


@pytest.fixture
def pipeline_instance():
    """A `FraudDetectionPipeline` built against a fully mocked SageMaker
    SDK / boto3 — no real AWS calls."""
    with patch.object(pl, "PipelineSession") as mock_pipeline_session, patch.object(
        pl.boto3, "client"
    ) as mock_boto_client, patch.object(pl.boto3, "Session"):
        mock_boto_client.return_value.get_caller_identity.return_value = {
            "Account": "123456789012"
        }
        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "test-bucket"
        mock_pipeline_session.return_value = mock_session

        yield pl.FraudDetectionPipeline(role="arn:aws:iam::123456789012:role/fake-role")


# ---------------------------------------------------------------------------
# 7.1 — _build_seed_source_dir() copies the three files into a fresh temp dir
# ---------------------------------------------------------------------------


def test_build_seed_source_dir_copies_expected_files(pipeline_instance):
    staging_dir = pipeline_instance._build_seed_source_dir()
    try:
        assert staging_dir.is_dir()
        staged_names = {p.name for p in staging_dir.iterdir()}
        assert staged_names == {
            "seed_athena_tables.py",
            "schema.py",
            "dataset_schema.yaml",
        }
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def test_build_seed_source_dir_returns_a_fresh_dir_each_call(pipeline_instance):
    """Each call stages into its own new temp dir, not a shared/reused one."""
    dir_a = pipeline_instance._build_seed_source_dir()
    dir_b = pipeline_instance._build_seed_source_dir()
    try:
        assert dir_a != dir_b
        assert dir_a.is_dir()
        assert dir_b.is_dir()
    finally:
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)


def test_build_seed_source_dir_contents_match_real_source_files(pipeline_instance):
    """Staged files are byte-for-byte copies of the real repo files."""
    staging_dir = pipeline_instance._build_seed_source_dir()
    try:
        pipeline_steps_dir = Path(pl.__file__).parent / "pipeline_steps"
        config_dir = Path(pl.__file__).parent.parent / "config"

        assert (staging_dir / "seed_athena_tables.py").read_bytes() == (
            pipeline_steps_dir / "seed_athena_tables.py"
        ).read_bytes()
        assert (staging_dir / "schema.py").read_bytes() == (
            config_dir / "schema.py"
        ).read_bytes()
        assert (staging_dir / "dataset_schema.yaml").read_bytes() == (
            config_dir / "dataset_schema.yaml"
        ).read_bytes()
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7.2 — the seed step is built with FrameworkProcessor, not ScriptProcessor
# ---------------------------------------------------------------------------


def test_seed_step_uses_framework_processor_not_script_processor(pipeline_instance):
    with patch.object(pl, "FrameworkProcessor") as mock_fp, patch.object(
        pl, "ScriptProcessor"
    ) as mock_sp, patch.object(pl, "ProcessingStep") as mock_step, patch.object(
        pl, "retrieve_image_uri", return_value="fake-image-uri"
    ):
        pipeline_instance._create_seed_athena_step()

        mock_fp.assert_called_once()
        mock_sp.assert_not_called()

        # The step's step_args must come from the FrameworkProcessor's run(),
        # confirming FrameworkProcessor (not ScriptProcessor) built the step.
        mock_step.assert_called_once()
        assert (
            mock_step.call_args.kwargs["step_args"]
            is mock_fp.return_value.run.return_value
        )


def test_seed_step_run_called_with_source_dir(pipeline_instance):
    """FrameworkProcessor.run() is invoked with a source_dir (uploading the
    whole staged directory), not just a single `code=` file."""
    with patch.object(pl, "FrameworkProcessor") as mock_fp, patch.object(
        pl, "ProcessingStep"
    ), patch.object(pl, "retrieve_image_uri", return_value="fake-image-uri"):
        pipeline_instance._create_seed_athena_step()

        run_call_kwargs = mock_fp.return_value.run.call_args.kwargs
        assert run_call_kwargs["code"] == "seed_athena_tables.py"
        assert "source_dir" in run_call_kwargs
        assert run_call_kwargs["source_dir"]  # non-empty path string


# ---------------------------------------------------------------------------
# 7.3 — target_column pipeline parameter default comes from schema.target_column()
# ---------------------------------------------------------------------------


def test_target_column_parameter_default_comes_from_schema(pipeline_instance):
    params = pipeline_instance._define_parameters()

    assert params["target_column"].name == "TargetColumn"
    assert params["target_column"].default_value == schema.target_column()


# ---------------------------------------------------------------------------
# 7.4 — ModelTrainer's target-column hyperparameter resolves from the
# pipeline parameter at execution time, not a hardcoded literal
# ---------------------------------------------------------------------------


def test_model_trainer_target_column_hyperparameter_wraps_the_pipeline_parameter(
    pipeline_instance,
):
    """The 'target-column' hyperparameter passed to ModelTrainer must be a
    Join() wrapping the pipeline's target_column ParameterString — i.e. its
    value resolves at pipeline-execution time from the parameter, not a
    literal string baked in at pipeline-definition time."""
    params = pipeline_instance._define_parameters()

    preprocessing_step = MagicMock()
    preprocessing_step.properties.ProcessingOutputConfig.Outputs.__getitem__.return_value.S3Output.S3Uri = (
        "s3://bucket/preprocessing"
    )

    with patch.object(pl, "ModelTrainer") as mock_model_trainer, patch.object(
        pl, "TrainingStep"
    ), patch.object(pl, "retrieve_image_uri", return_value="fake-image-uri"):
        pipeline_instance._create_training_step(params, preprocessing_step)

        hyperparameters = mock_model_trainer.call_args.kwargs["hyperparameters"]
        target_column_hp = hyperparameters["target-column"]

        # Must be a Join(...) — i.e. resolved via the workflow graph at
        # execution time — not a plain hardcoded string literal.
        assert isinstance(target_column_hp, pl.Join)
        assert target_column_hp.values == [params["target_column"]]

        # And it must be wired to the SAME parameter object _define_parameters()
        # defaulted from schema.target_column() — not some other literal.
        assert target_column_hp.values[0] is params["target_column"]
        assert params["target_column"].default_value == schema.target_column()
