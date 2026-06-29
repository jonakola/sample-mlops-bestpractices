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
#   ./cloudformation/deploy-stack.sh --drop-database       # one-shot wipe of the fraud_detection database
#                                                          # + all 7 table S3 prefixes BEFORE the CFN deploy
#                                                          # runs. Tables get recreated empty by the
#                                                          # lifecycle script on next Space launch.
#   AWS_REGION=us-east-1 ./cloudformation/deploy-stack.sh  # override region via env var
#   TEMPLATE=path/to/other.yaml ./cloudformation/deploy-stack.sh
#
# --drop-database is destructive but ONE-SHOT: it fires once at deploy time and
# does NOT persist as stack state. Subsequent Space restarts won't wipe data.
#
set -euo pipefail

DROP_DATABASE="false"
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --drop-database)
      DROP_DATABASE="true"
      ;;
    --help|-h)
      sed -n '1,/^set -euo pipefail$/p' "$0" | sed '$d'
      exit 0
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done
set -- "${POSITIONAL[@]:-}"

# Stack name + region are read from src/config/config.py (single source of
# truth across CFN, python config, every shell script). Env vars still work
# as overrides via the standard precedence inside config.py.
source "$(dirname "$0")/../scripts/_read_config.sh"
PROJECT_NAME_FROM_CONFIG="$(get_config PROJECT_NAME)"
STACK_NAME="${1:-${STACK_NAME:-$PROJECT_NAME_FROM_CONFIG}}"
REGION="$(get_config AWS_DEFAULT_REGION)"

TEMPLATE="${TEMPLATE:-$(dirname "$0")/sagemaker-mlflow-setup.yaml}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
S3_BUCKET="${S3_BUCKET:-cfn-deploy-${ACCOUNT_ID}-${REGION}}"
TEMPLATE_KEY="$(basename "$TEMPLATE").$(date +%s)"

# Reuse the same names config.py exposes; no second copy of defaults here.
PROJECT_NAME="$PROJECT_NAME_FROM_CONFIG"
DATA_BUCKET="${PROJECT_NAME}-data-${ACCOUNT_ID}"
ATHENA_DATABASE="$(get_config ATHENA_DATABASE)"
# S3 prefix used by the CFN lifecycle script for Athena table data.
# Kept as-is here because the prefix is a CFN-template constant (line 1278
# of sagemaker-mlflow-setup.yaml) — not a config.py knob.
S3_TABLE_PREFIX="fraud-detection"
RESETTABLE_TABLES=(
  training_data evaluation_data inference_responses
  ground_truth ground_truth_updates monitoring_responses drifted_data
)

echo "=== Deploy CloudFormation Stack ==="
echo "Stack:    $STACK_NAME"
echo "Region:   $REGION"
echo "Template: $TEMPLATE"
echo "S3 stage: s3://$S3_BUCKET/$TEMPLATE_KEY"
if [ "$DROP_DATABASE" = "true" ]; then
  echo "Drop database: ENABLED — one-shot wipe will run BEFORE the CFN deploy"
fi
echo ""

# ----------------------------------------------------------------------------
# Pre-deploy: one-shot DROP DATABASE + clear table S3 prefixes if requested.
# Runs ONLY when --drop-database was passed on the command line; nothing in
# the CFN template carries this flag forward, so subsequent Space restarts
# never wipe data.
# ----------------------------------------------------------------------------
if [ "$DROP_DATABASE" = "true" ]; then
  echo "=== Pre-deploy: DROP DATABASE $ATHENA_DATABASE CASCADE ==="
  if aws s3api head-bucket --bucket "$DATA_BUCKET" --region "$REGION" >/dev/null 2>&1; then
    OUTPUT_LOC="s3://${DATA_BUCKET}/athena-results/"
    echo "Dropping Athena database (CASCADE drops all tables in catalog)..."
    QID=$(aws athena start-query-execution \
      --query-string "DROP DATABASE IF EXISTS ${ATHENA_DATABASE} CASCADE" \
      --result-configuration "OutputLocation=${OUTPUT_LOC}" \
      --region "$REGION" \
      --query 'QueryExecutionId' --output text)
    for _ in $(seq 1 60); do
      STATE=$(aws athena get-query-execution --query-execution-id "$QID" --region "$REGION" --query 'QueryExecution.Status.State' --output text)
      case "$STATE" in
        SUCCEEDED) echo "  ✓ Database dropped"; break ;;
        FAILED|CANCELLED)
          REASON=$(aws athena get-query-execution --query-execution-id "$QID" --region "$REGION" --query 'QueryExecution.Status.StateChangeReason' --output text)
          echo "  ⚠ DROP DATABASE $STATE: $REASON"; break ;;
        *) sleep 2 ;;
      esac
    done

    echo "Clearing 7 table S3 prefixes under s3://${DATA_BUCKET}/${S3_TABLE_PREFIX}/ ..."
    for TBL in "${RESETTABLE_TABLES[@]}"; do
      PREFIX="${S3_TABLE_PREFIX}/${TBL}/"
      COUNT=$(aws s3 rm "s3://${DATA_BUCKET}/${PREFIX}" --recursive --region "$REGION" --only-show-errors 2>&1 | wc -l | tr -d ' ')
      echo "  ✓ Cleared ${PREFIX}"
    done
    echo ""
  else
    echo "Data bucket ${DATA_BUCKET} not found — skipping drop (first deploy?)"
    echo ""
  fi
fi

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
    elif [ "$k" = "DropDatabase" ]; then
      # Legacy parameter from earlier versions of the template — silently
      # drop it. The drop logic now lives in this script's pre-deploy step.
      :
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
  # CREATE: pass ProjectName from config.yaml so the template's default
  # doesn't silently win if the user later changes the yaml. All other
  # template parameters use their CFN-template defaults (documented in
  # cloudformation/README.md) — pass more here if you want them yaml-driven.
  CREATE_PARAMS="ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME_FROM_CONFIG"
  aws cloudformation create-stack \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --template-url "$TEMPLATE_URL" \
    --parameters $CREATE_PARAMS \
    --capabilities CAPABILITY_NAMED_IAM \
    --on-failure ROLLBACK
  echo "Waiting for create to complete..."
  aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME" --region "$REGION"
fi

echo ""
echo "✅ Done. Inspect with:"
echo "  aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION"
echo "  https://console.aws.amazon.com/cloudformation/home?region=$REGION#/stacks"
