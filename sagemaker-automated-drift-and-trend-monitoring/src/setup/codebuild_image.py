"""
Build + push a container image via AWS CodeBuild from a directory + Dockerfile.

Use case: SageMaker Studio JupyterLab spaces don't have a Docker daemon, so
`docker build` fails. The third-party `sagemaker-studio-image-build`
package (sm-docker) was the standard workaround, but its CLI calls v2-only
APIs (sagemaker.get_execution_role, sagemaker.session) that were removed
in SDK v3. We need v3 for SageMaker Pipelines, so sm-docker can't run.

This module replaces sm-docker. It:
  1. Zips the build context (source directory + Dockerfile)
  2. Uploads the zip to S3
  3. Provisions an ephemeral CodeBuild project
  4. Starts a build that:
       - Pulls the zip from S3
       - Runs `docker build` + `docker push` inside the CodeBuild container
       - Tags the resulting image and pushes to ECR
  5. Streams CloudWatch logs back to stdout
  6. Deletes the CodeBuild project on success/failure (idempotent)

Usage from the command line:
    python -m src.setup.codebuild_image \\
        --source-dir . \\
        --dockerfile src/drift_monitoring/Dockerfile.lambda \\
        --repository fraud-detection-drift-monitor \\
        --role-arn arn:aws:iam::123456789012:role/your-sagemaker-role

Or from another script:
    from src.setup.codebuild_image import build_and_push
    image_uri = build_and_push(
        source_dir=".",
        dockerfile="src/drift_monitoring/Dockerfile.lambda",
        repository="fraud-detection-drift-monitor",
        role_arn=os.environ["SAGEMAKER_EXEC_ROLE"],
        region="us-west-2",
    )
"""
from __future__ import annotations

import argparse
import io
import os
import time
import uuid
import zipfile
from pathlib import Path

import boto3


_BUILDSPEC_TEMPLATE = """\
version: 0.2

phases:
  pre_build:
    commands:
      - echo "Logging in to ECR..."
      - aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
  build:
    commands:
      - echo "Building image $REPOSITORY_URI:$IMAGE_TAG..."
      - docker build --platform linux/amd64 -t $REPOSITORY_URI:$IMAGE_TAG -f $DOCKERFILE .
  post_build:
    commands:
      - echo "Pushing image..."
      - docker push $REPOSITORY_URI:$IMAGE_TAG
      - echo "✓ Pushed $REPOSITORY_URI:$IMAGE_TAG"
"""


def _zip_source(source_dir: Path) -> bytes:
    """Zip the build context into memory. Skips .git, __pycache__, .venv."""
    excluded = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if any(part in excluded for part in path.parts):
                continue
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))
    return buf.getvalue()


def _ensure_bucket(s3, region: str, account_id: str) -> str:
    """Reuse the SageMaker default bucket convention."""
    bucket = f"sagemaker-{region}-{account_id}"
    try:
        s3.head_bucket(Bucket=bucket)
    except s3.exceptions.ClientError:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    return bucket


def build_and_push(
    source_dir: str,
    dockerfile: str,
    repository: str,
    role_arn: str,
    region: str | None = None,
    tag: str = "latest",
) -> str:
    """Build + push a container image to ECR via CodeBuild.

    Args:
        source_dir: Directory to zip and upload as the build context.
        dockerfile: Path to Dockerfile, relative to source_dir.
        repository: ECR repository name (must already exist).
        role_arn: IAM role ARN for the CodeBuild project to assume.
        region: AWS region. Defaults to AWS_DEFAULT_REGION env var.
        tag: Image tag. Defaults to "latest".

    Returns:
        Full image URI (account.dkr.ecr.region.amazonaws.com/repo:tag).
    """
    region = region or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    sts = boto3.client("sts", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    codebuild = boto3.client("codebuild", region_name=region)
    logs = boto3.client("logs", region_name=region)

    account_id = sts.get_caller_identity()["Account"]
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    image_uri = f"{registry}/{repository}:{tag}"

    # 1. Zip + upload build context
    print(f"  Zipping source: {source_dir}")
    bucket = _ensure_bucket(s3, region, account_id)
    key = f"codebuild-staging/{repository}-{uuid.uuid4().hex[:8]}.zip"
    zip_bytes = _zip_source(Path(source_dir))
    print(f"  Uploading {len(zip_bytes) / 1024:.0f} KB → s3://{bucket}/{key}")
    s3.put_object(Bucket=bucket, Key=key, Body=zip_bytes)

    # 2. Provision ephemeral CodeBuild project
    project_name = f"codebuild-{repository}-{uuid.uuid4().hex[:8]}"
    print(f"  Creating CodeBuild project: {project_name}")
    codebuild.create_project(
        name=project_name,
        source={
            "type": "S3",
            "location": f"{bucket}/{key}",
            "buildspec": _BUILDSPEC_TEMPLATE,
        },
        artifacts={"type": "NO_ARTIFACTS"},
        environment={
            "type": "LINUX_CONTAINER",
            "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
            "computeType": "BUILD_GENERAL1_SMALL",
            "privilegedMode": True,  # required for `docker build`
            "environmentVariables": [
                {"name": "AWS_REGION", "value": region},
                {"name": "ECR_REGISTRY", "value": registry},
                {"name": "REPOSITORY_URI", "value": f"{registry}/{repository}"},
                {"name": "IMAGE_TAG", "value": tag},
                {"name": "DOCKERFILE", "value": dockerfile},
            ],
        },
        serviceRole=role_arn,
        timeoutInMinutes=20,
    )

    try:
        # 3. Start the build
        build_id = codebuild.start_build(projectName=project_name)["build"]["id"]
        print(f"  Build started: {build_id}")
        print(f"  Streaming logs (Ctrl+C to detach, build keeps running)...")

        # 4. Poll status + stream logs
        last_token = None
        log_group = None
        log_stream = None
        terminal_states = {"SUCCEEDED", "FAILED", "FAULT", "TIMED_OUT", "STOPPED"}

        while True:
            info = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
            status = info["buildStatus"]
            logs_info = info.get("logs", {})
            log_group = logs_info.get("groupName") or log_group
            log_stream = logs_info.get("streamName") or log_stream

            if log_group and log_stream:
                kwargs = {
                    "logGroupName": log_group,
                    "logStreamName": log_stream,
                    "startFromHead": True,
                }
                if last_token:
                    kwargs["nextToken"] = last_token
                try:
                    resp = logs.get_log_events(**kwargs)
                    for event in resp.get("events", []):
                        print(f"    {event['message'].rstrip()}")
                    last_token = resp.get("nextForwardToken")
                except logs.exceptions.ResourceNotFoundException:
                    pass  # log stream not yet created

            if status in terminal_states:
                if status != "SUCCEEDED":
                    raise RuntimeError(f"CodeBuild build {status}: {build_id}")
                break
            time.sleep(5)

        print(f"  ✓ Image built and pushed: {image_uri}")
        return image_uri
    finally:
        # 5. Clean up the ephemeral CodeBuild project + the S3 zip
        try:
            codebuild.delete_project(name=project_name)
            print(f"  ✓ Cleaned up CodeBuild project")
        except Exception as e:
            print(f"  ⚠ Could not delete CodeBuild project {project_name}: {e}")
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--source-dir", default=".", help="Build context directory")
    parser.add_argument("--dockerfile", required=True, help="Dockerfile path (relative to source-dir)")
    parser.add_argument("--repository", required=True, help="ECR repository name")
    parser.add_argument("--role-arn", required=True, help="CodeBuild service role ARN")
    parser.add_argument("--region", default=None, help="AWS region (defaults to AWS_DEFAULT_REGION)")
    parser.add_argument("--tag", default="latest", help="Image tag")
    args = parser.parse_args()

    build_and_push(
        source_dir=args.source_dir,
        dockerfile=args.dockerfile,
        repository=args.repository,
        role_arn=args.role_arn,
        region=args.region,
        tag=args.tag,
    )


if __name__ == "__main__":
    _cli()
