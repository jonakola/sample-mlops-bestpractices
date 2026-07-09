# scripts/

Shell scripts for operations that CloudFormation and the `main.py` CLI can't handle on their own — primarily anything requiring a live Docker daemon (container-image Lambda deploys), plus small utilities used by the CFN scripts.

Most day-to-day operations should go through `main.py`. See [Running in Production](../README.md#running-in-production) in the main README for the full CLI surface.

## Files

| Script | Purpose |
|--------|---------|
| `deploy_lambda_container.sh` | Build the drift-monitor container image (locally with Docker if available, else via a temporary CodeBuild project), push to ECR, create/update the `fraud-detection-drift-monitor` Lambda, and wire the daily EventBridge rule. Called by `main.py monitoring deploy-lambda` under the hood; you rarely invoke it directly. |
| `delete_infrastructure.sh` | Tears down the drift-monitor Lambda + EventBridge rule (the out-of-band resources CloudFormation doesn't own). Run **before** `aws cloudformation delete-stack` on the drift-monitoring stack, since the Lambda references stack-owned resources. |
| `scrub-account-numbers.sh` | Local dev helper — scrubs AWS account IDs and role ARNs out of notebook outputs before committing. |
| `_read_config.sh` | Internal helper sourced by CFN deploy scripts to load values from `src/config/config.yaml` into shell variables. Not meant to run standalone. |

## `deploy_lambda_container.sh` reference

Full argument reference (the CLI wrapper covers the common case):

```bash
./scripts/deploy_lambda_container.sh <email> [data_drift_threshold] [model_drift_threshold]

# Examples
./scripts/deploy_lambda_container.sh you@example.com
./scripts/deploy_lambda_container.sh you@example.com 0.2 0.05
```

What it does end-to-end:

1. Creates the SNS topic + subscribes the given email (fire-and-forget — you confirm the subscription email that AWS sends).
2. Verifies the drift-monitor IAM role exists (created by the drift-monitoring CFN stack).
3. Creates/reuses the ECR repository.
4. Builds `src/drift_monitoring/Dockerfile.lambda` — tries local `docker build` first, falls back to `src/setup/codebuild_image.py` if no Docker daemon is available (this is the SageMaker Studio path).
5. Pushes the image to ECR and creates/updates the `fraud-detection-drift-monitor` Lambda pointing at it.
6. Attaches the CFN-provisioned drift-monitor role.
7. Creates the daily EventBridge rule with `ScheduleExpression: cron(0 2 * * ? *)`.
8. Sends one synchronous test invocation and prints the response payload.

Requires: AWS CLI configured with permissions for Lambda, ECR, IAM, SNS, EventBridge; either Docker running locally or the CodeBuild fallback prerequisites (roughly, CodeBuild + S3 access — see `src/setup/codebuild_image.py`).

Outputs (all named-resource names are stable across runs):

- SNS topic: `fraud-detection-drift-alerts`
- Lambda function: `fraud-detection-drift-monitor`
- ECR repository: `fraud-detection-drift-monitor`
- EventBridge rule: `fraud-detection-drift-check`

## `delete_infrastructure.sh` reference

```bash
./scripts/delete_infrastructure.sh          # keeps ECR images (default)
./scripts/delete_infrastructure.sh yes      # deletes ECR images too
```

Deletes in order: EventBridge rule + targets → Lambda functions (drift monitor + monitoring writer, if the drift-monitoring stack isn't already tearing the writer down) → SQS queue (if not stack-owned) → SNS topic + subscriptions → IAM role + policies → ECR repo (optional). Skips resources gracefully if they're already gone.

**Always run this before `aws cloudformation delete-stack --stack-name fraud-detection-drift-monitoring`** — the container Lambda references the stack's SQS queue and IAM role, so deleting the stack first leaves an orphan Lambda that then blocks the stack from cleanly recreating.
