"""
Library of SageMaker endpoint deployment operations.

Functions:
- resolve_inference_sqs_queue_url — look up the inference-logging SQS queue URL
- build_inference_env             — build the custom inference handler's env dict
- select_latest_approved_model    — pick a model version from the Model Registry
- create_model_from_package       — create a SageMaker Model from a registered package
- create_endpoint_config          — create a serverless endpoint configuration
- deploy_endpoint                 — create/update/delete-and-recreate an endpoint
- deploy                          — top-level orchestrator wiring the above together
- get_endpoint_status             — describe an endpoint (not-found-safe)
- delete_endpoint                 — delete an endpoint (not-found-safe)

Ported from `notebooks/2_deployment.ipynb` so the deployment logic is
importable and testable outside a notebook. Consumed by the project's
user-facing CLI in `main.py` (subcommand `deploy`). This module is
intentionally library-only — no `__main__` block — so there is one entry
point (`main.py`) for command-line use, matching `pipeline_cli.py`.
"""

import io
import logging
import os
import tarfile
import time
from datetime import datetime
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from src.config import config
from src.config.config import (
    AWS_DEFAULT_REGION,
    ATHENA_DATABASE,
    ATHENA_OUTPUT_S3,
    DATA_S3_BUCKET,
    MLFLOW_MODEL_NAME,
    SAGEMAKER_EXEC_ROLE,
    SERVERLESS_MAX_CONCURRENCY,
    SERVERLESS_MEMORY_SIZE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize boto3 clients
sagemaker_client = boto3.client('sagemaker', region_name=AWS_DEFAULT_REGION)

# Name of the SQS queue CFN creates for inference-response logging (resource
# `InferenceLoggerQueue`). The custom inference handler reads SQS_QUEUE_URL
# from its container env and sends a message per prediction; a downstream
# Lambda drains the queue and INSERTs into Athena `inference_responses`.
INFERENCE_SQS_QUEUE_NAME = os.getenv(
    'INFERENCE_SQS_QUEUE_NAME',
    'fraud-detection-monitoring-inference-logging',
)


def resolve_inference_sqs_queue_url(
    queue_name: str = INFERENCE_SQS_QUEUE_NAME,
    sqs_client: Optional[Any] = None,
) -> str:
    """
    Resolve the inference-logging SQS queue URL by name.

    Args:
        queue_name: SQS queue name to resolve
        sqs_client: Optional boto3 SQS client (constructed if not provided)

    Returns:
        The queue URL, or empty string if the queue could not be resolved.
        On failure the endpoint can still be deployed — inference logging
        to Athena is simply disabled (the handler logs "SQS send failed:
        NonExistentQueue" and skips the SQS send).
    """
    if sqs_client is None:
        sqs_client = boto3.client('sqs', region_name=AWS_DEFAULT_REGION)

    try:
        queue_url = sqs_client.get_queue_url(QueueName=queue_name)['QueueUrl']
        logger.info(f"✓ Resolved SQS queue: {queue_url}")
        return queue_url
    except ClientError as e:
        logger.warning(
            f"Could not resolve SQS queue '{queue_name}': {e}. "
            "Endpoint will deploy but inference logging to Athena will be disabled."
        )
        return ''


def build_inference_env(
    endpoint_name: str,
    inference_sqs_queue_url: str,
    region: str = AWS_DEFAULT_REGION,
) -> Dict[str, str]:
    """
    Build the environment dict passed to the custom inference handler.

    Args:
        endpoint_name: Name of the endpoint being deployed
        inference_sqs_queue_url: Resolved SQS queue URL (empty string disables logging)
        region: AWS region

    Returns:
        Dictionary of environment variables for the SageMaker Model container.
        Note: MODEL_VERSION defaults to 'v1' here and is overwritten by
        `create_model_from_package()` once the actual model version is known.
    """
    return {
        'ENABLE_ATHENA_LOGGING': 'true',
        'ATHENA_DATABASE': ATHENA_DATABASE,
        'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
        'DATA_S3_BUCKET': DATA_S3_BUCKET,
        'ENDPOINT_NAME': endpoint_name,
        'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI', ''),
        'MLFLOW_MODEL_NAME': MLFLOW_MODEL_NAME,
        'MLFLOW_RUN_ID': 'pipeline',
        'MODEL_VERSION': 'v1',
        'SQS_QUEUE_URL': inference_sqs_queue_url,
        # AWS_REGION is set explicitly because SageMaker doesn't always set
        # SAGEMAKER_REGION on containers created via raw boto3.create_model()
        # (only via ModelBuilder) — without it, boto3 clients in the handler
        # default to us-east-1 and fail with NonExistentQueue against a
        # queue in the actual deployment region.
        'AWS_REGION': region,
    }


def select_latest_approved_model(
    model_package_group: str,
    model_version: Optional[int] = None,
    sm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Select an approved model from the SageMaker Model Registry.

    Lists the most recent 10 approved model package versions in
    `model_package_group` and returns the requested one.

    Args:
        model_package_group: Name of the SageMaker Model Package Group
        model_version: Optional specific version to select. If omitted, the
            latest approved version (most recently created) is returned.
            Only the most recent 10 approved versions are searched — if the
            requested version isn't among them, a RuntimeError is raised
            rather than paginating further.
        sm_client: Optional boto3 SageMaker client (constructed if not provided)

    Returns:
        Dict with keys: 'arn', 'version', 'created_time' (isoformat string)

    Raises:
        RuntimeError: If the model package group has zero approved models,
            or if `model_version` was requested but not found among the
            most recent 10 approved versions.
        ClientError: sm_client.exceptions.ResourceNotFound propagates
            unchanged if the model package group itself doesn't exist.
    """
    if sm_client is None:
        sm_client = boto3.client('sagemaker', region_name=AWS_DEFAULT_REGION)

    logger.info(f"Listing approved models in package group: {model_package_group}")

    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=10,
        )
    except sm_client.exceptions.ResourceNotFound:
        logger.error(
            f"Model package group '{model_package_group}' not found — "
            "run the training pipeline first"
        )
        raise

    models = response.get('ModelPackageSummaryList', [])

    if not models:
        raise RuntimeError(
            f"No approved models found in model package group "
            f"'{model_package_group}'. Run the training pipeline first."
        )

    if model_version is not None:
        selected = next(
            (m for m in models if m.get('ModelPackageVersion') == model_version),
            None,
        )
        if selected is None:
            raise RuntimeError(
                f"Model version {model_version} not found among the most recent "
                f"{len(models)} approved versions in '{model_package_group}'."
            )
    else:
        selected = models[0]

    result = {
        'arn': selected['ModelPackageArn'],
        'version': selected.get('ModelPackageVersion'),
        'created_time': selected['CreationTime'].isoformat(),
    }

    logger.info(f"✓ Selected model version {result['version']}: {result['arn']}")
    return result


def create_model_from_package(
    model_package_arn: str,
    model_version: int,
    endpoint_name: str,
    role: str,
    region: str,
    inference_env: Dict[str, str],
    sm_client: Optional[Any] = None,
    s3_client: Optional[Any] = None,
) -> str:
    """
    Create a SageMaker Model from a registered model package.

    Validates the model package's artifact tarball contains
    `code/inference.py`, finalizes the inference-handler environment
    (model version + script-mode vars), and creates the model FROM the
    model package (not raw Image+ModelDataUrl) so it carries
    ModelPackageName for the drift Lambda's `describe_model ->
    ModelPackageName` lookup to succeed.

    Args:
        model_package_arn: ARN of the model package to deploy
        model_version: Model package version (used to tag MODEL_VERSION/MLFLOW_RUN_ID)
        endpoint_name: Name of the endpoint this model will serve (used to name the model)
        role: SageMaker execution role ARN
        region: AWS region (unused directly here, kept for caller symmetry with build_inference_env)
        inference_env: Environment dict (from `build_inference_env`) to attach to the container.
            Mutated in place with the resolved MODEL_VERSION/MLFLOW_RUN_ID/script-mode vars.
        sm_client: Optional boto3 SageMaker client (constructed if not provided)
        s3_client: Optional boto3 S3 client (constructed if not provided)

    Returns:
        The created model's name.

    Raises:
        RuntimeError: If `code/inference.py` is missing from the model artifact tarball.
    """
    if sm_client is None:
        sm_client = boto3.client('sagemaker', region_name=AWS_DEFAULT_REGION)
    if s3_client is None:
        s3_client = boto3.client('s3', region_name=AWS_DEFAULT_REGION)

    model_desc = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    model_data_url = model_desc['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    image_uri = model_desc['InferenceSpecification']['Containers'][0]['Image']

    logger.info(f"Model package ARN  : {model_package_arn}")
    logger.info(f"Model artifact     : {model_data_url}")
    logger.info(f"Container image    : {image_uri}")

    # Validate the tarball has code/inference.py — without it the stock
    # XGBoost container runs in algorithm mode and ignores our custom
    # handler. The training pipeline's train.py save_model() bakes this in.
    bucket = model_data_url.replace('s3://', '').split('/', 1)[0]
    key = model_data_url.replace(f's3://{bucket}/', '')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    with tarfile.open(fileobj=io.BytesIO(obj['Body'].read()), mode='r:gz') as tar:
        names = tar.getnames()

    if 'code/inference.py' not in names:
        raise RuntimeError(
            'code/inference.py is MISSING from the registered model package. '
            'Re-run the training pipeline — it bakes inference.py into the '
            'tarball during the TrainModel step.'
        )
    logger.info("✓ code/inference.py present in model artifact")

    # Update inference_env with the actual model version.
    inference_env['MODEL_VERSION'] = f'v{model_version}'
    inference_env['MLFLOW_RUN_ID'] = f'model-registry-v{model_version}'

    # Script-mode env vars — REQUIRED. The XGBoost container only honors
    # code/inference.py when SAGEMAKER_PROGRAM is set; without it the
    # container scans /opt/ml/model alphabetically and picks
    # feature_names.json as the model file, and the endpoint goes Failed.
    inference_env['SAGEMAKER_PROGRAM'] = 'inference.py'
    inference_env['SAGEMAKER_SUBMIT_DIRECTORY'] = '/opt/ml/model/code'

    model_name = f'{endpoint_name}-model-{int(datetime.now().timestamp())}'

    logger.info(f"Creating SageMaker model: {model_name}...")
    create_model_response = sm_client.create_model(
        ModelName=model_name,
        Containers=[{
            'ModelPackageName': model_package_arn,
            'Environment': inference_env,
        }],
        ExecutionRoleArn=role,
    )
    logger.info(f"✓ Model created: {model_name}")
    logger.info(f"  ARN: {create_model_response['ModelArn']}")

    return model_name


def create_endpoint_config(
    endpoint_name: str,
    model_name: str,
    memory_size_mb: int = SERVERLESS_MEMORY_SIZE,
    max_concurrency: int = SERVERLESS_MAX_CONCURRENCY,
    sm_client: Optional[Any] = None,
) -> str:
    """
    Create a serverless SageMaker endpoint configuration.

    Args:
        endpoint_name: Name of the endpoint this config will be used for (used to name the config)
        model_name: Name of the SageMaker Model to serve
        memory_size_mb: Serverless memory size in MB (1024/2048/3072/4096/5120/6144)
        max_concurrency: Max concurrent invocations
        sm_client: Optional boto3 SageMaker client (constructed if not provided)

    Returns:
        The created endpoint config's name.
    """
    if sm_client is None:
        sm_client = boto3.client('sagemaker', region_name=AWS_DEFAULT_REGION)

    endpoint_config_name = f'{endpoint_name}-config-{int(datetime.now().timestamp())}'

    logger.info(f"Creating endpoint configuration: {endpoint_config_name}...")
    create_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'ServerlessConfig': {
                    'MemorySizeInMB': memory_size_mb,
                    'MaxConcurrency': max_concurrency,
                },
            }
        ],
    )
    logger.info(f"✓ Endpoint configuration created: {endpoint_config_name}")
    logger.info(f"  ARN: {create_config_response['EndpointConfigArn']}")

    return endpoint_config_name


def deploy_endpoint(
    endpoint_name: str,
    endpoint_config_name: str,
    redeploy_clean: bool = True,
    sm_client: Optional[Any] = None,
    poll_interval: int = 30,
    max_wait_seconds: int = 1200,
    wait: bool = True,
) -> Dict[str, Any]:
    """
    Create, update, or delete-and-recreate a SageMaker endpoint.

    Args:
        endpoint_name: Name of the endpoint to deploy
        endpoint_config_name: Name of the endpoint config to deploy
        redeploy_clean: If True, delete any existing endpoint first for a
            clean slate. If False, update the existing endpoint's config
            in place (still delete-and-recreate if the existing endpoint's
            status is 'Failed', since a Failed endpoint can't be updated).
        sm_client: Optional boto3 SageMaker client (constructed if not provided)
        poll_interval: Seconds between status checks while waiting for the
            endpoint to reach a terminal state
        max_wait_seconds: Maximum total seconds to wait for the endpoint to
            reach InService/Failed (and, separately, the same bound applies
            to waiting for a delete-then-recreate's deletion to complete)
        wait: If False, skip polling for a terminal status — return
            immediately after create_endpoint/update_endpoint is called
            (status will be 'Creating'/'Updating', not a terminal state).

    Returns:
        Dict with keys: 'endpoint_name', 'endpoint_arn', 'status', 'endpoint_config_name'

    Raises:
        TimeoutError: If the endpoint (or its deletion) doesn't reach a
            terminal state within `max_wait_seconds`.
        RuntimeError: If the endpoint reaches 'Failed' status, including the
            FailureReason if available.
        ClientError: Any describe_endpoint error other than "not found" propagates.
    """
    if sm_client is None:
        sm_client = boto3.client('sagemaker', region_name=AWS_DEFAULT_REGION)

    # Check existing state.
    endpoint_exists = False
    endpoint_status = None
    try:
        existing = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        endpoint_status = existing['EndpointStatus']
        logger.info(f"Endpoint {endpoint_name} already exists (status: {endpoint_status})")
    except ClientError as e:
        if 'Could not find endpoint' in str(e):
            logger.info(f"Endpoint {endpoint_name} does not exist yet")
        else:
            raise

    # Delete-and-recreate path — used when redeploy_clean is True or status is Failed.
    if endpoint_exists and (redeploy_clean or endpoint_status == 'Failed'):
        logger.info(f"Deleting existing endpoint {endpoint_name}...")
        sm_client.delete_endpoint(EndpointName=endpoint_name)

        delete_deadline = time.time() + max_wait_seconds
        while True:
            try:
                sm_client.describe_endpoint(EndpointName=endpoint_name)
            except ClientError:
                logger.info(f"✓ Endpoint {endpoint_name} deleted")
                endpoint_exists = False
                break

            if time.time() > delete_deadline:
                raise TimeoutError(
                    f"Timed out after {max_wait_seconds}s waiting for endpoint "
                    f"{endpoint_name} to finish deleting"
                )
            time.sleep(5)

    # Create or update the endpoint.
    if endpoint_exists:
        logger.info(f"Updating endpoint config: {endpoint_name} -> {endpoint_config_name}...")
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        logger.info("✓ Endpoint update initiated")
    else:
        logger.info(f"Creating endpoint: {endpoint_name}...")
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        logger.info("✓ Endpoint creation initiated")
        logger.info(f"  ARN: {create_endpoint_response['EndpointArn']}")

    if not wait:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        return {
            'endpoint_name': endpoint_name,
            'endpoint_arn': response['EndpointArn'],
            'status': response['EndpointStatus'],
            'endpoint_config_name': endpoint_config_name,
        }

    # Poll for a terminal status.
    logger.info("Monitoring deployment (this takes 5-10 minutes)...")
    start_time = time.time()
    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        elapsed = int(time.time() - start_time)
        logger.info(f"[{elapsed}s] Endpoint status: {status}")

        if status == 'InService':
            logger.info(f"✓ Endpoint is live and ready for inference: {endpoint_name}")
            return {
                'endpoint_name': endpoint_name,
                'endpoint_arn': response['EndpointArn'],
                'status': status,
                'endpoint_config_name': endpoint_config_name,
            }

        if status == 'Failed':
            failure_reason = response.get('FailureReason', 'Unknown')
            raise RuntimeError(
                f"Endpoint {endpoint_name} deployment failed: {failure_reason}"
            )

        if elapsed > max_wait_seconds:
            raise TimeoutError(
                f"Timed out after {max_wait_seconds}s waiting for endpoint "
                f"{endpoint_name} to reach a terminal status (last status: {status})"
            )

        time.sleep(poll_interval)


def deploy(
    *,
    endpoint_name: Optional[str] = None,
    model_package_group: Optional[str] = None,
    model_version: Optional[int] = None,
    region: Optional[str] = None,
    role: Optional[str] = None,
    memory_size_mb: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    redeploy_clean: bool = True,
    wait: bool = True,
) -> Dict[str, Any]:
    """
    Deploy the latest (or a specific) approved model to a serverless endpoint.

    Top-level orchestrator wiring together model selection, model/endpoint-config
    creation, and endpoint deployment. Intended to be called directly by CLI code.

    Args:
        endpoint_name: Endpoint name. Defaults to config.ENDPOINT_NAME.
        model_package_group: Model Package Group to select from. Defaults to
            config.MLFLOW_MODEL_NAME (the MLflow registered model name, which
            matches the SageMaker Model Package Group name — see pipeline.py).
        model_version: Specific model version to deploy. Defaults to the latest approved version.
        region: AWS region. Defaults to config.AWS_DEFAULT_REGION.
        role: SageMaker execution role ARN. Resolution order: this param ->
            config.SAGEMAKER_EXEC_ROLE -> get_execution_role() (SageMaker
            environment / caller identity). Raises ValueError if none resolve.
        memory_size_mb: Serverless memory size in MB. Defaults to config.SERVERLESS_MEMORY_SIZE.
        max_concurrency: Max concurrent invocations. Defaults to config.SERVERLESS_MAX_CONCURRENCY.
        redeploy_clean: If True, delete any existing endpoint first for a clean slate.
        wait: If False, return immediately after create/update_endpoint is
            called without waiting for a terminal status.

    Returns:
        Dict with keys: 'endpoint_name', 'endpoint_arn', 'status', 'model_name',
        'model_package_arn', 'model_version', 'endpoint_config_name'.

    Raises:
        ValueError: If no execution role can be resolved.
        RuntimeError: Propagated from `select_latest_approved_model` /
            `create_model_from_package` / `deploy_endpoint` (missing approved
            models, missing inference.py, or Failed endpoint status).
    """
    resolved_endpoint_name = endpoint_name or config.ENDPOINT_NAME
    resolved_model_package_group = model_package_group or config.MLFLOW_MODEL_NAME
    resolved_region = region or config.AWS_DEFAULT_REGION
    resolved_memory_size_mb = memory_size_mb if memory_size_mb is not None else config.SERVERLESS_MEMORY_SIZE
    resolved_max_concurrency = max_concurrency if max_concurrency is not None else config.SERVERLESS_MAX_CONCURRENCY

    # Role resolution mirrors pipeline.py's FraudDetectionPipeline.__init__:
    # explicit param -> config.SAGEMAKER_EXEC_ROLE -> get_execution_role().
    if role:
        resolved_role = role
    elif SAGEMAKER_EXEC_ROLE:
        resolved_role = SAGEMAKER_EXEC_ROLE
    else:
        try:
            from src.utils.aws_utils import get_execution_role
            resolved_role = get_execution_role()
        except Exception:
            raise ValueError(
                "Could not determine execution role. "
                "Please provide role ARN or set SAGEMAKER_EXEC_ROLE"
            )

    sm_client = boto3.client('sagemaker', region_name=resolved_region)
    s3_client = boto3.client('s3', region_name=resolved_region)
    sqs_client = boto3.client('sqs', region_name=resolved_region)

    model_info = select_latest_approved_model(
        resolved_model_package_group,
        model_version=model_version,
        sm_client=sm_client,
    )

    inference_sqs_queue_url = resolve_inference_sqs_queue_url(sqs_client=sqs_client)
    inference_env = build_inference_env(
        resolved_endpoint_name, inference_sqs_queue_url, resolved_region
    )

    model_name = create_model_from_package(
        model_package_arn=model_info['arn'],
        model_version=model_info['version'],
        endpoint_name=resolved_endpoint_name,
        role=resolved_role,
        region=resolved_region,
        inference_env=inference_env,
        sm_client=sm_client,
        s3_client=s3_client,
    )

    endpoint_config_name = create_endpoint_config(
        endpoint_name=resolved_endpoint_name,
        model_name=model_name,
        memory_size_mb=resolved_memory_size_mb,
        max_concurrency=resolved_max_concurrency,
        sm_client=sm_client,
    )

    deploy_result = deploy_endpoint(
        endpoint_name=resolved_endpoint_name,
        endpoint_config_name=endpoint_config_name,
        redeploy_clean=redeploy_clean,
        sm_client=sm_client,
        wait=wait,
    )

    return {
        'endpoint_name': deploy_result['endpoint_name'],
        'endpoint_arn': deploy_result['endpoint_arn'],
        'status': deploy_result['status'],
        'model_name': model_name,
        'model_package_arn': model_info['arn'],
        'model_version': model_info['version'],
        'endpoint_config_name': endpoint_config_name,
    }


def get_endpoint_status(
    endpoint_name: str,
    sm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Get the current status of a SageMaker endpoint.

    Args:
        endpoint_name: Name of the endpoint
        sm_client: Optional boto3 SageMaker client (constructed if not provided)

    Returns:
        Dict with keys: 'endpoint_name', 'status', 'endpoint_config_name', 'endpoint_arn'.
        If the endpoint doesn't exist, returns {'endpoint_name': ..., 'status': 'NotFound'}
        rather than raising. Unexpected errors propagate.
    """
    if sm_client is None:
        sm_client = boto3.client('sagemaker', region_name=AWS_DEFAULT_REGION)

    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        if 'Could not find endpoint' in str(e):
            logger.info(f"Endpoint {endpoint_name} not found")
            return {'endpoint_name': endpoint_name, 'status': 'NotFound'}
        raise

    return {
        'endpoint_name': endpoint_name,
        'status': response['EndpointStatus'],
        'endpoint_config_name': response['EndpointConfigName'],
        'endpoint_arn': response['EndpointArn'],
    }


def delete_endpoint(
    endpoint_name: str,
    sm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Delete a SageMaker endpoint.

    Args:
        endpoint_name: Name of the endpoint to delete
        sm_client: Optional boto3 SageMaker client (constructed if not provided)

    Returns:
        Dict with keys: 'endpoint_name', 'status'. Status is 'deleted' on
        success or 'not_found' if the endpoint didn't exist (logged, not raised).
        Unexpected errors propagate.
    """
    if sm_client is None:
        sm_client = boto3.client('sagemaker', region_name=AWS_DEFAULT_REGION)

    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        if 'Could not find endpoint' in str(e):
            logger.info(f"Endpoint {endpoint_name} not found, nothing to delete")
            return {'endpoint_name': endpoint_name, 'status': 'not_found'}
        raise

    logger.info(f"✓ Deleted endpoint: {endpoint_name}")
    return {'endpoint_name': endpoint_name, 'status': 'deleted'}
