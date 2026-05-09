#!/bin/bash
# force-delete-stack.sh
#
# Force-cleanup a stuck CloudFormation stack. Use this when `delete-stack`
# fails (usually on S3 bucket with versioned objects or subnets with
# stuck ENIs) and the stack sits in DELETE_FAILED.
#
# What it does:
#   1. Empties and deletes the S3 data bucket (all versions + delete markers)
#   2. Finds and deletes SageMaker-owned ENIs in the stack's subnets
#   3. Deletes the subnets manually
#   4. Retries `delete-stack`. If still stuck, uses --retain-resources to
#      skip the problem resources and then cleans up the leftover Lambda
#      and IAM role manually.
#
# Usage:
#   ./force-delete-stack.sh [stack-name] [region]
#
# Defaults: stack-name=fraud-detection-monitoring, region=us-west-2
#
# Requires: aws CLI, jq

set -euo pipefail

STACK_NAME="${1:-fraud-detection-monitoring}"
REGION="${2:-${AWS_DEFAULT_REGION:-us-west-2}}"

echo "=============================================="
echo "Force-deleting stuck CloudFormation stack"
echo "  Stack:  $STACK_NAME"
echo "  Region: $REGION"
echo "=============================================="
echo

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: this script requires jq. Install with: brew install jq"
  exit 1
fi

# ------------------------------------------------------------------
# Step 1 — Empty and delete the S3 data bucket
# ------------------------------------------------------------------
echo "[1/5] Finding S3 data bucket..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
# Match the naming convention from the template
BUCKET=$(aws cloudformation describe-stack-resources \
  --stack-name "$STACK_NAME" --region "$REGION" \
  --query "StackResources[?ResourceType=='AWS::S3::Bucket'].PhysicalResourceId" \
  --output text 2>/dev/null || echo "")

# Fallback: derive from project name convention if the resource is gone from the stack
if [ -z "$BUCKET" ] || [ "$BUCKET" = "None" ]; then
  PROJECT_NAME=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Parameters[?ParameterKey=='ProjectName'].ParameterValue" \
    --output text 2>/dev/null || echo "")
  PROJECT_NAME="${PROJECT_NAME:-fraud-detection-monitoring}"
  BUCKET="${PROJECT_NAME}-data-${ACCOUNT_ID}"
fi

echo "    Bucket: $BUCKET"

if aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null; then
  echo "    Emptying versioned objects..."
  while true; do
    VERSIONS=$(aws s3api list-object-versions \
      --bucket "$BUCKET" --region "$REGION" \
      --max-items 1000 \
      --output json 2>/dev/null || echo '{}')

    OBJECTS=$(echo "$VERSIONS" | jq '[.Versions[]?, .DeleteMarkers[]? | {Key, VersionId}]')
    COUNT=$(echo "$OBJECTS" | jq 'length')

    if [ "$COUNT" -eq 0 ]; then
      break
    fi

    echo "      Deleting $COUNT object(s)..."
    aws s3api delete-objects \
      --bucket "$BUCKET" --region "$REGION" \
      --delete "$(echo "$OBJECTS" | jq '{Objects: ., Quiet: true}')" \
      >/dev/null
  done

  echo "    Deleting bucket..."
  aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION" || echo "    (bucket delete failed — may already be gone)"
else
  echo "    Bucket not found or already deleted — skipping."
fi
echo

# ------------------------------------------------------------------
# Step 2 — Find and delete SageMaker-owned ENIs in stack subnets
# ------------------------------------------------------------------
echo "[2/5] Finding subnets in the stack..."
SUBNET_IDS=$(aws cloudformation describe-stack-resources \
  --stack-name "$STACK_NAME" --region "$REGION" \
  --query "StackResources[?ResourceType=='AWS::EC2::Subnet'].PhysicalResourceId" \
  --output text 2>/dev/null || echo "")

if [ -z "$SUBNET_IDS" ]; then
  echo "    No subnets found in stack."
else
  for SUBNET in $SUBNET_IDS; do
    echo "    Subnet: $SUBNET"

    ENIS=$(aws ec2 describe-network-interfaces --region "$REGION" \
      --filters "Name=subnet-id,Values=$SUBNET" \
      --query "NetworkInterfaces[].NetworkInterfaceId" \
      --output text 2>/dev/null || echo "")

    if [ -z "$ENIS" ]; then
      echo "      No ENIs found."
      continue
    fi

    for ENI in $ENIS; do
      STATUS=$(aws ec2 describe-network-interfaces --region "$REGION" \
        --network-interface-ids "$ENI" \
        --query "NetworkInterfaces[0].Status" --output text 2>/dev/null || echo "unknown")
      ATTACHMENT_ID=$(aws ec2 describe-network-interfaces --region "$REGION" \
        --network-interface-ids "$ENI" \
        --query "NetworkInterfaces[0].Attachment.AttachmentId" --output text 2>/dev/null || echo "None")

      echo "      ENI $ENI (status: $STATUS)"

      if [ "$ATTACHMENT_ID" != "None" ] && [ -n "$ATTACHMENT_ID" ]; then
        echo "        Detaching attachment $ATTACHMENT_ID..."
        aws ec2 detach-network-interface --region "$REGION" \
          --attachment-id "$ATTACHMENT_ID" --force 2>/dev/null || echo "        (detach failed — may be owned by SageMaker)"
        # ENI owned by SageMaker can take a few minutes to release
        sleep 5
      fi

      echo "        Deleting ENI..."
      aws ec2 delete-network-interface --region "$REGION" \
        --network-interface-id "$ENI" 2>/dev/null || echo "        (delete failed — will retry after SageMaker releases it)"
    done
  done
fi
echo

# ------------------------------------------------------------------
# Step 3 — Wait briefly for SageMaker to release any ENIs it still owns,
#          then retry subnet deletion manually.
# ------------------------------------------------------------------
echo "[3/5] Waiting 30s for SageMaker to release any remaining ENIs..."
sleep 30

if [ -n "$SUBNET_IDS" ]; then
  for SUBNET in $SUBNET_IDS; do
    REMAINING=$(aws ec2 describe-network-interfaces --region "$REGION" \
      --filters "Name=subnet-id,Values=$SUBNET" \
      --query "length(NetworkInterfaces)" --output text 2>/dev/null || echo "0")

    if [ "$REMAINING" != "0" ]; then
      echo "    $SUBNET still has $REMAINING ENI(s) — attempting delete anyway"
    fi

    echo "    Deleting subnet $SUBNET..."
    aws ec2 delete-subnet --region "$REGION" --subnet-id "$SUBNET" \
      2>/dev/null || echo "      (subnet delete failed — will be handled by stack retry)"
  done
fi
echo

# ------------------------------------------------------------------
# Step 4 — Retry the stack delete. If it still fails, skip the
#          problem custom resources and the bucket (already deleted).
# ------------------------------------------------------------------
echo "[4/5] Retrying stack delete..."
aws cloudformation delete-stack \
  --stack-name "$STACK_NAME" --region "$REGION"

echo "    Waiting up to 10 min for completion..."
if aws cloudformation wait stack-delete-complete \
     --stack-name "$STACK_NAME" --region "$REGION" 2>/dev/null; then
  echo "    ✅ Stack deleted cleanly."
  exit 0
fi

echo "    Stack delete did not complete cleanly. Retrying with --retain-resources for"
echo "    resources that commonly get stuck (S3 bucket cleanup custom resource + bucket)."

# Retain resources that block progress when their underlying AWS resource
# is already gone or in a stuck state.
aws cloudformation delete-stack \
  --stack-name "$STACK_NAME" --region "$REGION" \
  --retain-resources EmptyS3BucketCustomResource DataBucket \
  2>/dev/null || echo "    (retain-resources call may have no effect if stack is already progressing)"

if aws cloudformation wait stack-delete-complete \
     --stack-name "$STACK_NAME" --region "$REGION" 2>/dev/null; then
  echo "    ✅ Stack deleted after retain-resources."
else
  echo "    ⚠️  Stack still not deleted. Manual cleanup needed — see step 5."
fi
echo

# ------------------------------------------------------------------
# Step 5 — Best-effort cleanup of leftover Lambdas and IAM roles
#          that may survive --retain-resources.
# ------------------------------------------------------------------
echo "[5/5] Cleaning up leftover Lambdas and IAM roles..."

PROJECT_NAME=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" --region "$REGION" \
  --query "Stacks[0].Parameters[?ParameterKey=='ProjectName'].ParameterValue" \
  --output text 2>/dev/null || echo "fraud-detection-monitoring")
PROJECT_NAME="${PROJECT_NAME:-fraud-detection-monitoring}"

for FN in \
  "${PROJECT_NAME}-empty-bucket-lambda" \
  "${PROJECT_NAME}-wait-eni-cleanup" \
  "${PROJECT_NAME}-app-cleanup"
do
  if aws lambda get-function --function-name "$FN" --region "$REGION" >/dev/null 2>&1; then
    echo "    Deleting Lambda: $FN"
    aws lambda delete-function --function-name "$FN" --region "$REGION" || true
  fi
done

# IAM roles may have random suffixes (e.g. EmptyS3BucketRole-XXXXX).
# Match by prefix and delete cleanly (detach policies first).
for ROLE_PREFIX in \
  "${PROJECT_NAME}-EmptyS3BucketRole" \
  "${PROJECT_NAME}-WaitForENICleanupRole" \
  "${PROJECT_NAME}-AppCleanupRole" \
  "${PROJECT_NAME}-SageMakerExecutionRole"
do
  ROLES=$(aws iam list-roles \
    --query "Roles[?starts_with(RoleName, \`${ROLE_PREFIX}\`)].RoleName" \
    --output text 2>/dev/null || echo "")

  for ROLE in $ROLES; do
    echo "    Cleaning up IAM role: $ROLE"

    # Detach all managed policies
    for POLICY_ARN in $(aws iam list-attached-role-policies --role-name "$ROLE" \
        --query "AttachedPolicies[].PolicyArn" --output text 2>/dev/null); do
      aws iam detach-role-policy --role-name "$ROLE" --policy-arn "$POLICY_ARN" 2>/dev/null || true
    done

    # Delete all inline policies
    for POLICY_NAME in $(aws iam list-role-policies --role-name "$ROLE" \
        --query "PolicyNames[]" --output text 2>/dev/null); do
      aws iam delete-role-policy --role-name "$ROLE" --policy-name "$POLICY_NAME" 2>/dev/null || true
    done

    aws iam delete-role --role-name "$ROLE" 2>/dev/null || echo "      (role delete failed)"
  done
done

echo
echo "=============================================="
echo "Force-delete complete. Verify with:"
echo "  aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION"
echo "=============================================="
