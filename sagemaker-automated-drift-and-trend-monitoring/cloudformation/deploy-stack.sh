#!/bin/bash
#
# Deploy (create-or-update) the CloudFormation stack.
#
# Uses create-stack / update-stack directly (not `aws cloudformation deploy`),
# because the change-set path runs an AWS::EarlyValidation::PropertyValidation
# hook that rejects this template on update. Direct submission works.
#
# Usage:
#   ./cloudformation/deploy-stack.sh                       # default: fraud-detection-monitoring in us-west-2
#   ./cloudformation/deploy-stack.sh my-other-stack        # override stack name positionally
#   AWS_REGION=us-east-1 ./cloudformation/deploy-stack.sh  # override region via env var
#   TEMPLATE=path/to/other.yaml ./cloudformation/deploy-stack.sh
#
set -euo pipefail

STACK_NAME="${1:-${STACK_NAME:-fraud-detection-monitoring}}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
TEMPLATE="${TEMPLATE:-$(dirname "$0")/sagemaker-mlflow-setup.yaml}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
S3_BUCKET="${S3_BUCKET:-cfn-deploy-${ACCOUNT_ID}-${REGION}}"
TEMPLATE_KEY="$(basename "$TEMPLATE").$(date +%s)"

echo "=== Deploy CloudFormation Stack ==="
echo "Stack:    $STACK_NAME"
echo "Region:   $REGION"
echo "Template: $TEMPLATE"
echo "S3 stage: s3://$S3_BUCKET/$TEMPLATE_KEY"
echo ""

if aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
  ACTION="update"
  echo "Stack exists — running UPDATE"
else
  ACTION="create"
  echo "Stack not found — running CREATE"
fi

if ! aws s3api head-bucket --bucket "$S3_BUCKET" --region "$REGION" >/dev/null 2>&1; then
  echo "Creating staging bucket: $S3_BUCKET"
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
fi

echo "Uploading template..."
aws s3 cp "$TEMPLATE" "s3://$S3_BUCKET/$TEMPLATE_KEY" --region "$REGION" >/dev/null
TEMPLATE_URL="https://s3.${REGION}.amazonaws.com/${S3_BUCKET}/${TEMPLATE_KEY}"
echo ""

if [ "$ACTION" = "update" ]; then
  # Reuse all previous parameter values; the template's defaults take over for
  # any params that didn't exist on the previous deployment.
  PARAM_KEYS="$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].Parameters[].ParameterKey' --output text)"
  PARAM_ARGS=""
  for k in $PARAM_KEYS; do
    # Auto-increment LifecycleConfigVersion with timestamp to force replacement
    if [ "$k" = "LifecycleConfigVersion" ]; then
      NEW_VERSION="v$(date +%s)"
      PARAM_ARGS="$PARAM_ARGS ParameterKey=$k,ParameterValue=$NEW_VERSION"
      echo "Auto-incrementing LifecycleConfigVersion to: $NEW_VERSION"
    else
      PARAM_ARGS="$PARAM_ARGS ParameterKey=$k,UsePreviousValue=true"
    fi
  done

  set +e
  OUT="$(aws cloudformation update-stack \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --template-url "$TEMPLATE_URL" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters $PARAM_ARGS 2>&1)"
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then
    if echo "$OUT" | grep -q "No updates are to be performed"; then
      echo "No updates to perform — stack is already up to date."
      exit 0
    fi
    echo "$OUT"
    exit $RC
  fi
  echo "$OUT"
  echo "Waiting for update to complete..."
  aws cloudformation wait stack-update-complete --stack-name "$STACK_NAME" --region "$REGION"
else
  aws cloudformation create-stack \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --template-url "$TEMPLATE_URL" \
    --capabilities CAPABILITY_NAMED_IAM \
    --on-failure ROLLBACK
  echo "Waiting for create to complete..."
  aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME" --region "$REGION"
fi

echo ""
echo "✅ Done. Inspect with:"
echo "  aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION"
echo "  https://console.aws.amazon.com/cloudformation/home?region=$REGION#/stacks"
