#!/bin/bash
set -uo pipefail

###############################################################################
# Robust CloudFormation stack deletion for sagemaker-mlflow-setup.yaml.
#
# Run this AFTER scripts/delete_infrastructure.sh — that one removes the
# out-of-band drift-monitor Lambda + EventBridge rule + SNS topic + CloudWatch
# dashboard/alarms. This one drains everything CFN can't drain on its own,
# then issues the delete-stack call and waits.
#
# Why we don't just `aws cloudformation delete-stack`: this stack has 3 known
# resources that block delete unless they're drained first.
#
#   1. S3 buckets with objects (CFN can't delete a non-empty bucket).
#      → empty the bucket via `aws s3 rm --recursive` + clear versioned
#        delete markers if versioning ever ran on it.
#
#   2. SageMaker Studio Space — if a JupyterLab app is RUNNING, the Space
#      can't be deleted. We list apps for the user profile and stop each.
#
#   3. Lake Formation grants on the Athena database — they reference the
#      SageMaker exec role, which can hold up the role delete on stack
#      teardown.
#
# Usage:
#     ./cloudformation/delete_stack.sh                       # dry-run (default) — shows what would happen, deletes nothing
#     ./cloudformation/delete_stack.sh --execute             # actually delete (uses default stack name)
#     ./cloudformation/delete_stack.sh --execute my-stack    # delete a non-default stack
#
# Defaults:
#     stack-name = fraud-detection-monitoring (matches CFN default ProjectName)
#     region     = $AWS_REGION or $AWS_DEFAULT_REGION or us-west-2
###############################################################################

DRY_RUN=true
STACK_NAME=""
for arg in "$@"; do
    case "$arg" in
        --execute|-x) DRY_RUN=false ;;
        --help|-h)
            sed -n '2,/^####/p' "$0" | sed 's/^#//' | head -n -1
            exit 0
            ;;
        --*)
            echo "Unknown flag: $arg" >&2
            exit 1
            ;;
        *)
            STACK_NAME="$arg"
            ;;
    esac
done
# Stack name + region from src/config/config.py (single source of truth).
source "$(dirname "$0")/../scripts/_read_config.sh"
STACK_NAME="${STACK_NAME:-$(get_config PROJECT_NAME)}"
REGION="$(get_config AWS_DEFAULT_REGION)"

if $DRY_RUN; then
    MODE="DRY-RUN (no changes — pass --execute to actually delete)"
else
    MODE="EXECUTE (resources WILL be deleted)"
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  CloudFormation stack delete                                       ║"
echo "║  $MODE"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Stack:   $STACK_NAME"
echo "  Region:  $REGION"
echo ""

# Sanity check: stack actually exists. If not, nothing to do.
STATUS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "MISSING")
if [ "$STATUS" = "MISSING" ]; then
    echo "Stack $STACK_NAME not found in $REGION — nothing to delete."
    exit 0
fi
echo "  Current status: $STATUS"

# Refuse to run (in execute mode) if the out-of-band Lambda still exists —
# it references the CFN-owned IAM role + SQS queue + SNS topic, so its
# presence will fail the stack delete.
_DRIFT_LAMBDA="$(get_config DRIFT_LAMBDA_NAME)"
if aws lambda get-function --function-name "$_DRIFT_LAMBDA" --region "$REGION" >/dev/null 2>&1; then
    echo ""
    echo "  ⚠️  Drift-monitor Lambda still exists."
    if ! $DRY_RUN; then
        echo "  Run scripts/delete_infrastructure.sh --execute first to drain the"
        echo "  out-of-band resources, then re-run this script."
        exit 1
    else
        echo "  In execute mode this script would refuse to run."
        echo "  Resolve by running: scripts/delete_infrastructure.sh --execute"
    fi
fi
echo ""

if ! $DRY_RUN; then
    read -p "Delete stack '$STACK_NAME' and ALL its resources? (type 'yes' to confirm): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""
fi

# Helper: prefix actions with [DRY-RUN] or run them depending on mode.
run() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] would run: $*"
    else
        "$@"
    fi
}

# ────────────────────────────────────────────────────────────────────────────
# 1. Inventory + (if executing) empty S3 buckets owned by the stack.
# ────────────────────────────────────────────────────────────────────────────
echo "[1/5] S3 buckets owned by the stack..."
BUCKETS=$(aws cloudformation list-stack-resources --stack-name "$STACK_NAME" --region "$REGION" \
    --query "StackResourceSummaries[?ResourceType=='AWS::S3::Bucket'].PhysicalResourceId" \
    --output text 2>/dev/null || echo "")
if [ -z "$BUCKETS" ]; then
    echo "  (no S3 buckets in stack)"
else
    for bucket in $BUCKETS; do
        if ! aws s3api head-bucket --bucket "$bucket" --region "$REGION" 2>/dev/null; then
            echo "  - $bucket: already gone, skipping"
            continue
        fi
        OBJECT_COUNT=$(aws s3api list-objects-v2 --bucket "$bucket" --region "$REGION" \
            --query 'KeyCount' --output text 2>/dev/null || echo "0")
        echo "  - $bucket: $OBJECT_COUNT objects"
        if $DRY_RUN; then
            echo "    [DRY-RUN] would empty bucket (aws s3 rm --recursive + sweep versioned markers)"
        else
            aws s3 rm "s3://$bucket" --recursive --region "$REGION" >/dev/null 2>&1 || true
            VERSIONS=$(aws s3api list-object-versions --bucket "$bucket" --region "$REGION" \
                --output json --max-items 1000 2>/dev/null || echo "{}")
            if [ "$(echo "$VERSIONS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get("Versions",[]))+len(d.get("DeleteMarkers",[])))' 2>/dev/null)" != "0" ]; then
                echo "    (cleaning versioned objects + delete markers)"
                python3 - "$bucket" "$REGION" <<'PYEOF' || true
import sys, boto3
bucket, region = sys.argv[1], sys.argv[2]
s3 = boto3.client('s3', region_name=region)
paginator = s3.get_paginator('list_object_versions')
for page in paginator.paginate(Bucket=bucket):
    delete_keys = []
    for obj in page.get('Versions', []) + page.get('DeleteMarkers', []):
        delete_keys.append({'Key': obj['Key'], 'VersionId': obj['VersionId']})
    for i in range(0, len(delete_keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_keys[i:i+1000], 'Quiet': True})
PYEOF
            fi
            echo "    ✓ $bucket emptied"
        fi
    done
fi

# ────────────────────────────────────────────────────────────────────────────
# 2. Stop running SageMaker Studio apps. A Space delete fails while a
#    JupyterLab app is RUNNING.
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/5] SageMaker Studio apps..."
DOMAIN_ID=$(aws cloudformation describe-stack-resource --stack-name "$STACK_NAME" \
    --logical-resource-id SageMakerDomain --region "$REGION" \
    --query 'StackResourceDetail.PhysicalResourceId' --output text 2>/dev/null || echo "")
if [ -z "$DOMAIN_ID" ] || [ "$DOMAIN_ID" = "None" ]; then
    echo "  (no SageMaker domain in stack, skipping)"
else
    echo "  Domain: $DOMAIN_ID"
    APPS=$(aws sagemaker list-apps --domain-id-equals "$DOMAIN_ID" --region "$REGION" \
        --query "Apps[?Status!='Deleted' && Status!='Failed'].[AppName,AppType,SpaceName,UserProfileName,Status]" \
        --output text 2>/dev/null || echo "")
    if [ -z "$APPS" ]; then
        echo "  No running apps."
    else
        echo "$APPS" | while IFS=$'\t' read -r APP_NAME APP_TYPE SPACE_NAME USER_NAME APP_STATUS; do
            [ -z "$APP_NAME" ] && continue
            TARGET_ARG=""
            if [ -n "$SPACE_NAME" ] && [ "$SPACE_NAME" != "None" ]; then
                TARGET_ARG="--space-name $SPACE_NAME"
            elif [ -n "$USER_NAME" ] && [ "$USER_NAME" != "None" ]; then
                TARGET_ARG="--user-profile-name $USER_NAME"
            fi
            echo "    - $APP_NAME (type=$APP_TYPE, status=$APP_STATUS)"
            run aws sagemaker delete-app --domain-id "$DOMAIN_ID" $TARGET_ARG \
                --app-type "$APP_TYPE" --app-name "$APP_NAME" --region "$REGION"
        done
        if ! $DRY_RUN; then
            echo "  Waiting up to 5 minutes for apps to finish deleting..."
            for i in $(seq 1 30); do
                REMAINING=$(aws sagemaker list-apps --domain-id-equals "$DOMAIN_ID" --region "$REGION" \
                    --query "length(Apps[?Status!='Deleted' && Status!='Failed'])" \
                    --output text 2>/dev/null || echo "0")
                if [ "$REMAINING" = "0" ]; then
                    echo "  ✓ All apps deleted."
                    break
                fi
                sleep 10
            done
        fi
    fi
fi

# ────────────────────────────────────────────────────────────────────────────
# 3. Revoke Lake Formation grants on the Athena DB (best-effort).
# ────────────────────────────────────────────────────────────────────────────
echo ""
_ATHENA_DB="$(get_config ATHENA_DATABASE)"
echo "[3/5] Lake Formation grants on Athena database '$_ATHENA_DB' (best-effort)..."
LF_GRANTS=$(aws lakeformation list-permissions --resource "{\"Database\":{\"Name\":\"$_ATHENA_DB\"}}" \
    --region "$REGION" --query 'length(PrincipalResourcePermissions)' \
    --output text 2>/dev/null || echo "0")
echo "  Found $LF_GRANTS grant(s) to revoke"
if [ "$LF_GRANTS" != "0" ] && [ "$LF_GRANTS" != "None" ]; then
    if $DRY_RUN; then
        echo "  [DRY-RUN] would revoke $LF_GRANTS LF grant(s)"
    else
        aws lakeformation list-permissions --resource "{\"Database\":{\"Name\":\"$_ATHENA_DB\"}}" \
            --region "$REGION" --output json 2>/dev/null | \
            python3 - "$REGION" <<'PYEOF' || true
import sys, json, boto3
region = sys.argv[1]
data = json.load(sys.stdin)
lf = boto3.client('lakeformation', region_name=region)
grants = data.get('PrincipalResourcePermissions', [])
revoked, failed = 0, []
for grant in grants:
    try:
        lf.revoke_permissions(
            Principal=grant['Principal'],
            Resource=grant['Resource'],
            Permissions=grant.get('Permissions', []),
            PermissionsWithGrantOption=grant.get('PermissionsWithGrantOption', []),
        )
        revoked += 1
    except Exception as e:
        # Track but don't abort — best-effort cleanup. CFN delete will still
        # attempt the role delete and surface a clear error if a grant
        # references a role being torn down.
        failed.append(f"{grant.get('Principal', {}).get('DataLakePrincipalIdentifier', '?')}: {e.__class__.__name__}")
print(f"  Revoked {revoked}/{len(grants)} LF grant(s)")
if failed:
    print(f"  ⚠ {len(failed)} revoke(s) failed — stack delete may surface IAM role errors:")
    for f in failed[:5]:
        print(f"    - {f}")
PYEOF
    fi
fi

# ────────────────────────────────────────────────────────────────────────────
# 4. Issue delete-stack.
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Stack delete..."
if $DRY_RUN; then
    echo "  [DRY-RUN] would run: aws cloudformation delete-stack --stack-name $STACK_NAME --region $REGION"
    # Show the full inventory of resources that would be deleted.
    RES_COUNT=$(aws cloudformation list-stack-resources --stack-name "$STACK_NAME" --region "$REGION" \
        --query 'length(StackResourceSummaries)' --output text 2>/dev/null || echo "0")
    echo "  Stack contains $RES_COUNT resources that would be deleted, including:"
    aws cloudformation list-stack-resources --stack-name "$STACK_NAME" --region "$REGION" \
        --query "StackResourceSummaries[].[ResourceType,LogicalResourceId]" \
        --output table 2>/dev/null | head -50
else
    aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
fi

# ────────────────────────────────────────────────────────────────────────────
# 5. Wait for delete (only in execute mode). CFN's wait timeout is 3 hours;
#    the MLflow tracking server alone takes ~10 min. We poll every 30s and
#    surface status transitions instead of staring at a silent wait command.
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Wait for delete..."
if $DRY_RUN; then
    echo "  [DRY-RUN] would poll for DELETE_COMPLETE (typically 10-20 min)"
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Dry-run complete. Re-run with --execute to actually delete.       ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    exit 0
fi

LAST_STATUS=""
while true; do
    STATUS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
        --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "DELETE_COMPLETE")
    if [ "$STATUS" = "DELETE_COMPLETE" ]; then
        echo "  ✓ Stack deleted"
        break
    fi
    if [ "$STATUS" = "DELETE_FAILED" ]; then
        echo ""
        echo "  ❌ DELETE_FAILED. Resources that failed to delete:"
        aws cloudformation describe-stack-events --stack-name "$STACK_NAME" --region "$REGION" \
            --query "StackEvents[?ResourceStatus=='DELETE_FAILED'].[LogicalResourceId,ResourceType,ResourceStatusReason]" \
            --output table 2>/dev/null
        echo ""
        echo "  Common fixes:"
        echo "    - 'Bucket not empty' → re-run this script; Step 1 will re-drain it"
        echo "    - 'Role still has attached policies' → manually detach + retry"
        echo "    - 'Network Interface in use' → wait 10 min for ENIs to detach,"
        echo "      then aws cloudformation delete-stack --stack-name $STACK_NAME"
        echo "         --retain-resources <ids>"
        exit 1
    fi
    if [ "$STATUS" != "$LAST_STATUS" ]; then
        echo "  ... $STATUS"
        LAST_STATUS="$STATUS"
    fi
    sleep 30
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Stack deleted                                                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "CloudWatch log groups (/aws/lambda/${STACK_NAME}-*, /aws/sagemaker/*) are"
echo "NOT auto-deleted by CFN — kept for post-mortem debugging. Sweep them with:"
echo ""
echo "  aws logs describe-log-groups --log-group-name-prefix /aws/lambda/${STACK_NAME} \\"
echo "      --region $REGION --query 'logGroups[].logGroupName' --output text | \\"
echo "    xargs -n1 aws logs delete-log-group --region $REGION --log-group-name"
echo ""
echo "ECR repo for the drift-monitor image is also kept by default (so a"
echo "redeploy doesn't re-rebuild). Delete with:"
echo "  scripts/delete_infrastructure.sh --execute --ecr"
