"""Unit tests for `src/train_pipeline/deploy_endpoint.py`.

Covers:
  - `select_latest_approved_model()` picks the requested version override,
    or defaults to the latest approved version.
  - `create_model_from_package()` raises RuntimeError when
    code/inference.py is missing from the model artifact tarball.
  - `deploy_endpoint()` distinguishes delete-then-create vs. update-in-place
    based on `redeploy_clean` and the existing endpoint's status.
  - `deploy_endpoint()` raises on 'Failed' terminal status, including the
    FailureReason.
  - `get_endpoint_status()` / `delete_endpoint()` handle not-found gracefully.

All boto3 clients are mocked — no real AWS calls are made.
"""

from __future__ import annotations

import io
import tarfile
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.train_pipeline import deploy_endpoint as de


def _client_error(message: str, operation: str = "DescribeEndpoint") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": message}},
        operation_name=operation,
    )


def _make_tarball_bytes(names):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in names:
            info = tarfile.TarInfo(name=name)
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# select_latest_approved_model()
# ---------------------------------------------------------------------------


def _model_summary(version, arn, created):
    return {
        "ModelPackageArn": arn,
        "ModelPackageVersion": version,
        "ModelApprovalStatus": "Approved",
        "CreationTime": created,
    }


def test_select_latest_approved_model_defaults_to_latest():
    """Without a version override, models[0] (the most recent) is returned."""
    sm_client = MagicMock()
    sm_client.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            _model_summary(3, "arn:...:3", datetime(2026, 6, 25, 16, 50, 20)),
            _model_summary(2, "arn:...:2", datetime(2026, 6, 25, 16, 23, 40)),
            _model_summary(1, "arn:...:1", datetime(2026, 6, 25, 5, 13, 35)),
        ]
    }

    result = de.select_latest_approved_model("fraud-detection", sm_client=sm_client)

    assert result["arn"] == "arn:...:3"
    assert result["version"] == 3
    assert result["created_time"] == datetime(2026, 6, 25, 16, 50, 20).isoformat()


def test_select_latest_approved_model_honors_version_override():
    """When model_version is given, that specific version is selected
    instead of defaulting to models[0]."""
    sm_client = MagicMock()
    sm_client.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            _model_summary(3, "arn:...:3", datetime(2026, 6, 25, 16, 50, 20)),
            _model_summary(2, "arn:...:2", datetime(2026, 6, 25, 16, 23, 40)),
            _model_summary(1, "arn:...:1", datetime(2026, 6, 25, 5, 13, 35)),
        ]
    }

    result = de.select_latest_approved_model(
        "fraud-detection", model_version=1, sm_client=sm_client
    )

    assert result["arn"] == "arn:...:1"
    assert result["version"] == 1


def test_select_latest_approved_model_raises_when_none_approved():
    sm_client = MagicMock()
    sm_client.list_model_packages.return_value = {"ModelPackageSummaryList": []}

    with pytest.raises(RuntimeError):
        de.select_latest_approved_model("fraud-detection", sm_client=sm_client)


def test_select_latest_approved_model_raises_when_version_not_found():
    sm_client = MagicMock()
    sm_client.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            _model_summary(3, "arn:...:3", datetime(2026, 6, 25, 16, 50, 20)),
        ]
    }

    with pytest.raises(RuntimeError):
        de.select_latest_approved_model(
            "fraud-detection", model_version=99, sm_client=sm_client
        )


# ---------------------------------------------------------------------------
# create_model_from_package()
# ---------------------------------------------------------------------------


def test_create_model_from_package_raises_when_inference_py_missing():
    """When code/inference.py is absent from the model artifact tarball,
    create_model_from_package() must raise RuntimeError before ever calling
    sm_client.create_model()."""
    sm_client = MagicMock()
    sm_client.describe_model_package.return_value = {
        "InferenceSpecification": {
            "Containers": [
                {
                    "ModelDataUrl": "s3://my-bucket/models/model.tar.gz",
                    "Image": "xgboost-image-uri",
                }
            ]
        }
    }

    s3_client = MagicMock()
    tar_bytes = _make_tarball_bytes(["feature_names.json", "xgboost-model"])
    s3_client.get_object.return_value = {"Body": io.BytesIO(tar_bytes)}

    with pytest.raises(RuntimeError, match="code/inference.py is MISSING"):
        de.create_model_from_package(
            model_package_arn="arn:...:3",
            model_version=3,
            endpoint_name="fraud-detector-endpoint",
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            region="us-west-2",
            inference_env={},
            sm_client=sm_client,
            s3_client=s3_client,
        )

    sm_client.create_model.assert_not_called()


def test_create_model_from_package_succeeds_when_inference_py_present():
    sm_client = MagicMock()
    sm_client.describe_model_package.return_value = {
        "InferenceSpecification": {
            "Containers": [
                {
                    "ModelDataUrl": "s3://my-bucket/models/model.tar.gz",
                    "Image": "xgboost-image-uri",
                }
            ]
        }
    }
    sm_client.create_model.return_value = {"ModelArn": "arn:aws:sagemaker:model/foo"}

    s3_client = MagicMock()
    tar_bytes = _make_tarball_bytes(["code/inference.py", "xgboost-model"])
    s3_client.get_object.return_value = {"Body": io.BytesIO(tar_bytes)}

    inference_env = {"ENABLE_ATHENA_LOGGING": "true"}

    model_name = de.create_model_from_package(
        model_package_arn="arn:...:3",
        model_version=3,
        endpoint_name="fraud-detector-endpoint",
        role="arn:aws:iam::123456789012:role/SageMakerRole",
        region="us-west-2",
        inference_env=inference_env,
        sm_client=sm_client,
        s3_client=s3_client,
    )

    assert model_name.startswith("fraud-detector-endpoint-model-")
    assert inference_env["MODEL_VERSION"] == "v3"
    assert inference_env["MLFLOW_RUN_ID"] == "model-registry-v3"
    assert inference_env["SAGEMAKER_PROGRAM"] == "inference.py"
    assert inference_env["SAGEMAKER_SUBMIT_DIRECTORY"] == "/opt/ml/model/code"
    sm_client.create_model.assert_called_once()
    _, kwargs = sm_client.create_model.call_args
    assert kwargs["Containers"][0]["ModelPackageName"] == "arn:...:3"


# ---------------------------------------------------------------------------
# deploy_endpoint()
# ---------------------------------------------------------------------------


def test_deploy_endpoint_creates_when_no_existing_endpoint():
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = [
        _client_error("Could not find endpoint"),  # initial existence check
        {"EndpointArn": "arn:endpoint", "EndpointStatus": "InService"},  # poll
    ]
    sm_client.create_endpoint.return_value = {"EndpointArn": "arn:endpoint"}

    result = de.deploy_endpoint(
        "fraud-detector-endpoint", "config-1", sm_client=sm_client, poll_interval=0
    )

    sm_client.create_endpoint.assert_called_once()
    sm_client.update_endpoint.assert_not_called()
    sm_client.delete_endpoint.assert_not_called()
    assert result["status"] == "InService"


def test_deploy_endpoint_updates_in_place_when_redeploy_clean_false():
    """redeploy_clean=False with an existing (non-Failed) endpoint should
    update_endpoint() rather than delete-then-create."""
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = [
        {"EndpointArn": "arn:endpoint", "EndpointStatus": "InService"},  # existence check
        {"EndpointArn": "arn:endpoint", "EndpointStatus": "InService"},  # poll
    ]

    result = de.deploy_endpoint(
        "fraud-detector-endpoint",
        "config-2",
        redeploy_clean=False,
        sm_client=sm_client,
        poll_interval=0,
    )

    sm_client.update_endpoint.assert_called_once_with(
        EndpointName="fraud-detector-endpoint", EndpointConfigName="config-2"
    )
    sm_client.delete_endpoint.assert_not_called()
    sm_client.create_endpoint.assert_not_called()
    assert result["status"] == "InService"


def test_deploy_endpoint_deletes_then_creates_when_redeploy_clean_true():
    """redeploy_clean=True with an existing endpoint should delete it, wait
    for deletion, then create_endpoint()."""
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = [
        {"EndpointArn": "arn:endpoint", "EndpointStatus": "InService"},  # existence check
        _client_error("Could not find endpoint"),  # deletion-poll: now gone
        {"EndpointArn": "arn:endpoint2", "EndpointStatus": "InService"},  # terminal poll
    ]
    sm_client.create_endpoint.return_value = {"EndpointArn": "arn:endpoint2"}

    result = de.deploy_endpoint(
        "fraud-detector-endpoint",
        "config-3",
        redeploy_clean=True,
        sm_client=sm_client,
        poll_interval=0,
    )

    sm_client.delete_endpoint.assert_called_once_with(EndpointName="fraud-detector-endpoint")
    sm_client.create_endpoint.assert_called_once()
    sm_client.update_endpoint.assert_not_called()
    assert result["status"] == "InService"


def test_deploy_endpoint_deletes_then_creates_when_existing_status_failed():
    """Even with redeploy_clean=False, a Failed existing endpoint must be
    deleted and recreated rather than updated."""
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = [
        {"EndpointArn": "arn:endpoint", "EndpointStatus": "Failed"},  # existence check
        _client_error("Could not find endpoint"),  # deletion-poll: now gone
        {"EndpointArn": "arn:endpoint2", "EndpointStatus": "InService"},  # terminal poll
    ]
    sm_client.create_endpoint.return_value = {"EndpointArn": "arn:endpoint2"}

    result = de.deploy_endpoint(
        "fraud-detector-endpoint",
        "config-4",
        redeploy_clean=False,
        sm_client=sm_client,
        poll_interval=0,
    )

    sm_client.delete_endpoint.assert_called_once()
    sm_client.create_endpoint.assert_called_once()
    sm_client.update_endpoint.assert_not_called()
    assert result["status"] == "InService"


def test_deploy_endpoint_raises_on_failed_status_with_failure_reason():
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = [
        _client_error("Could not find endpoint"),  # existence check
        {
            "EndpointArn": "arn:endpoint",
            "EndpointStatus": "Failed",
            "FailureReason": "The model container failed to load the model.",
        },  # poll
    ]
    sm_client.create_endpoint.return_value = {"EndpointArn": "arn:endpoint"}

    with pytest.raises(RuntimeError, match="The model container failed to load the model."):
        de.deploy_endpoint(
            "fraud-detector-endpoint", "config-5", sm_client=sm_client, poll_interval=0
        )


def test_deploy_endpoint_skips_polling_when_wait_false():
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = [
        _client_error("Could not find endpoint"),  # existence check
        {"EndpointArn": "arn:endpoint", "EndpointStatus": "Creating"},  # immediate status
    ]
    sm_client.create_endpoint.return_value = {"EndpointArn": "arn:endpoint"}

    result = de.deploy_endpoint(
        "fraud-detector-endpoint", "config-6", sm_client=sm_client, wait=False
    )

    assert result["status"] == "Creating"
    # Only 2 describe_endpoint calls: the existence check + the immediate
    # status fetch. No polling loop should have run.
    assert sm_client.describe_endpoint.call_count == 2


def test_deploy_endpoint_reraises_unexpected_client_error():
    """A ClientError that isn't the 'not found' message must propagate."""
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = _client_error("AccessDenied")

    with pytest.raises(ClientError):
        de.deploy_endpoint("fraud-detector-endpoint", "config-7", sm_client=sm_client)


# ---------------------------------------------------------------------------
# get_endpoint_status() / delete_endpoint()
# ---------------------------------------------------------------------------


def test_get_endpoint_status_returns_not_found_gracefully():
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = _client_error("Could not find endpoint")

    result = de.get_endpoint_status("missing-endpoint", sm_client=sm_client)

    assert result == {"endpoint_name": "missing-endpoint", "status": "NotFound"}


def test_get_endpoint_status_returns_details_when_found():
    sm_client = MagicMock()
    sm_client.describe_endpoint.return_value = {
        "EndpointStatus": "InService",
        "EndpointConfigName": "cfg-1",
        "EndpointArn": "arn:endpoint",
    }

    result = de.get_endpoint_status("fraud-detector-endpoint", sm_client=sm_client)

    assert result == {
        "endpoint_name": "fraud-detector-endpoint",
        "status": "InService",
        "endpoint_config_name": "cfg-1",
        "endpoint_arn": "arn:endpoint",
    }


def test_get_endpoint_status_reraises_unexpected_client_error():
    sm_client = MagicMock()
    sm_client.describe_endpoint.side_effect = _client_error("AccessDenied")

    with pytest.raises(ClientError):
        de.get_endpoint_status("fraud-detector-endpoint", sm_client=sm_client)


def test_delete_endpoint_returns_not_found_gracefully():
    sm_client = MagicMock()
    sm_client.delete_endpoint.side_effect = _client_error("Could not find endpoint")

    result = de.delete_endpoint("missing-endpoint", sm_client=sm_client)

    assert result == {"endpoint_name": "missing-endpoint", "status": "not_found"}


def test_delete_endpoint_returns_deleted_on_success():
    sm_client = MagicMock()
    sm_client.delete_endpoint.return_value = {}

    result = de.delete_endpoint("fraud-detector-endpoint", sm_client=sm_client)

    assert result == {"endpoint_name": "fraud-detector-endpoint", "status": "deleted"}


def test_delete_endpoint_reraises_unexpected_client_error():
    sm_client = MagicMock()
    sm_client.delete_endpoint.side_effect = _client_error("AccessDenied")

    with pytest.raises(ClientError):
        de.delete_endpoint("fraud-detector-endpoint", sm_client=sm_client)
