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
# Why we don't just `aws cloudformation delete-stack`: this stack has 4 known
# resources that block delete unless they're drained first.
#
#   1. Glue/Athena Iceberg tables MUST be dropped BEFORE the data bucket
#      empties — Iceberg metadata files live in S3, and if those vanish
#      while the Glue catalog still points at them, the next deploy hits
#      `ICEBERG_MISSING_METADATA: Metadata not found in metadata location`
#      on every SELECT. CFN doesn't clean the catalog on its own (tables
#      are created by a custom resource Lambda; the catalog entries survive).
#
#   2. S3 buckets with objects (CFN can't delete a non-empty bucket).
#      → empty the bucket via `aws s3 rm --recursive` + clear versioned
#        delete markers if versioning ever ran on it.
#
#   3. SageMaker Studio Space — if a JupyterLab app is RUNNING, the Space
#      can't be deleted. We list apps for the user profile and stop each.
#
#   4. Lake Formation grants on the Athena database — they reference the
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

# Special case: stack is already at DELETE_FAILED from a prior run. Tell the
# user what we'll do (sweep EFS orphans, retry delete) and proceed in execute
# mode; in dry-run we just describe the recovery.
IS_RETRY_FROM_DELETE_FAILED=false
if [ "$STATUS" = "DELETE_FAILED" ]; then
    IS_RETRY_FROM_DELETE_FAILED=true
    echo ""
    echo "  Detected DELETE_FAILED — likely orphaned EFS mount targets from"
    echo "  the SageMaker domain (a known AWS race). This script will sweep"
    echo "  them and retry the stack delete."
fi

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
# 1. Drop Glue/Athena tables BEFORE emptying the S3 bucket.
#
# Why this order matters:
#   The Athena tables are Iceberg-format and store their `metadata/` pointer
#   files in the data bucket. If we empty the bucket first, those metadata
#   files vanish and the tables become unreadable with
#       ICEBERG_MISSING_METADATA: Metadata not found in metadata location
#   on the next deploy (when someone tries `SELECT * FROM training_data`).
#
#   The Glue Catalog entries are NOT tracked by CFN directly — they're
#   created by the `SetupAthenaTables` custom resource on stack create.
#   On stack delete CFN tears down the custom resource Lambda but does NOT
#   re-run any cleanup logic on the catalog. So we have to drop the tables
#   ourselves, before we touch the bucket.
# ────────────────────────────────────────────────────────────────────────────
echo "[1/6] Dropping Glue/Athena tables in '$(get_config ATHENA_DATABASE)'..."
_GLUE_DB="$(get_config ATHENA_DATABASE)"
if aws glue get-database --name "$_GLUE_DB" --region "$REGION" >/dev/null 2>&1; then
    GLUE_TABLES=$(aws glue get-tables --database-name "$_GLUE_DB" --region "$REGION" \
        --query 'TableList[].Name' --output text 2>/dev/null || echo "")
    if [ -z "$GLUE_TABLES" ]; then
        echo "  Database exists but contains no tables — skipping."
    else
        for t in $GLUE_TABLES; do
            if $DRY_RUN; then
                echo "  [DRY-RUN] would drop $_GLUE_DB.$t"
            else
                if aws glue delete-table --database-name "$_GLUE_DB" --name "$t" \
                    --region "$REGION" 2>/dev/null; then
                    echo "  ✓ dropped $_GLUE_DB.$t"
                else
                    echo "  ⚠ could not drop $_GLUE_DB.$t (continuing)"
                fi
            fi
        done
    fi
else
    echo "  (database '$_GLUE_DB' does not exist — nothing to drop)"
fi

# ────────────────────────────────────────────────────────────────────────────
# 2. Inventory + (if executing) empty S3 buckets owned by the stack.
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/6] S3 buckets owned by the stack..."
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
# 3. Stop running SageMaker Studio apps. A Space delete fails while a
#    JupyterLab app is RUNNING.
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/6] SageMaker Studio apps..."
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
# Helper: sweep orphaned EFS mount targets in the stack's subnets.
#
# Why this exists:
#   AWS::SageMaker::Domain provisions an EFS file system at create time but
#   leaves it BEHIND on delete (not deterministically — sometimes 10-30 min,
#   sometimes never). The orphaned EFS mount targets hold ENIs in the stack's
#   subnets, so AWS::EC2::Subnet can't delete and the stack lands in
#   DELETE_FAILED with "subnet has dependencies and cannot be deleted".
#
#   The EFS file system itself is NOT in CFN, so CFN can't drain it.
#
# Sweep strategy:
#   1. Find the VPC + subnets the stack owns (works whether stack is
#      mid-delete, DELETE_FAILED, or pre-delete).
#   2. List all EFS mount targets in those subnets.
#   3. Delete each mount target (releases the ENI in ~30-60s).
#   4. Optionally delete the file system once mount targets are gone.
# ────────────────────────────────────────────────────────────────────────────
sweep_orphan_efs() {
    local action_label="${1:-sweep}"
    echo ""
    echo "Sweeping orphaned EFS mount targets ($action_label)..."

    # Get all subnets owned by the stack (works in any stack status as long
    # as the resources still exist — CFN lists them even when DELETE_FAILED).
    local subnets
    subnets=$(aws cloudformation list-stack-resources --stack-name "$STACK_NAME" --region "$REGION" \
        --query "StackResourceSummaries[?ResourceType=='AWS::EC2::Subnet'].PhysicalResourceId" \
        --output text 2>/dev/null || echo "")
    if [ -z "$subnets" ]; then
        echo "  (no subnets in stack — nothing to sweep)"
        return 0
    fi

    # For each subnet, list EFS mount targets attached. EFS doesn't have a
    # by-subnet filter, so we walk all file systems in the account and check
    # if any of their mount targets land in our subnets.
    local subnet_set
    subnet_set=" $(echo "$subnets" | tr '\n\t' '  ') "  # space-padded for substring match

    local fs_ids
    fs_ids=$(aws efs describe-file-systems --region "$REGION" \
        --query 'FileSystems[].FileSystemId' --output text 2>/dev/null || echo "")

    local found=0
    local fs_to_delete=()
    for fs in $fs_ids; do
        local mts
        mts=$(aws efs describe-mount-targets --file-system-id "$fs" --region "$REGION" \
            --query 'MountTargets[].[MountTargetId,SubnetId]' --output text 2>/dev/null || echo "")
        local fs_has_orphan=false
        while IFS=$'\t' read -r mt_id mt_subnet; do
            [ -z "$mt_id" ] && continue
            # Substring check: is mt_subnet in our space-padded subnet set?
            if [[ "$subnet_set" == *" $mt_subnet "* ]]; then
                found=$((found+1))
                fs_has_orphan=true
                echo "  - Mount target $mt_id (fs=$fs, subnet=$mt_subnet)"
                run aws efs delete-mount-target --mount-target-id "$mt_id" --region "$REGION"
            fi
        done <<< "$mts"
        if $fs_has_orphan; then
            fs_to_delete+=("$fs")
        fi
    done

    if [ "$found" = "0" ]; then
        echo "  ✓ No orphan EFS mount targets in stack subnets."
        return 0
    fi

    if $DRY_RUN; then
        echo "  [DRY-RUN] would wait for $found mount target(s) to fully delete, then drop the EFS"
        return 0
    fi

    # Wait for mount targets to fully delete (releases ENIs)
    echo "  Waiting up to 3 min for mount targets to release ENIs..."
    for i in $(seq 1 18); do
        local remaining=0
        for fs in "${fs_to_delete[@]}"; do
            local cnt
            cnt=$(aws efs describe-mount-targets --file-system-id "$fs" --region "$REGION" \
                --query 'length(MountTargets)' --output text 2>/dev/null || echo "0")
            remaining=$((remaining + cnt))
        done
        if [ "$remaining" = "0" ]; then
            echo "  ✓ All mount targets gone."
            break
        fi
        sleep 10
    done

    # Drop the now-empty EFS file systems too (they're orphans — domain owned them).
    for fs in "${fs_to_delete[@]}"; do
        echo "  - Deleting orphan EFS $fs"
        aws efs delete-file-system --file-system-id "$fs" --region "$REGION" 2>/dev/null || \
            echo "    (could not delete $fs — may still have lingering mount targets)"
    done
}

# ────────────────────────────────────────────────────────────────────────────
# Helper: sweep orphaned SageMaker-NFS security groups in the stack's VPC.
#
# Why this exists:
#   AWS::SageMaker::Domain creates two security groups at runtime — named
#   `security-group-for-inbound-nfs-d-<domain-id>` and
#   `security-group-for-outbound-nfs-d-<domain-id>`. These are NOT tracked by
#   CFN. After domain delete they orphan in the VPC and block VPC delete
#   with "has dependencies and cannot be deleted".
#
# Sweep strategy:
#   1. Find the VPC the stack owns.
#   2. List SGs in that VPC matching the SageMaker NFS naming pattern.
#   3. Strip all ingress/egress rules from each (the pair references each
#      other, so deletion fails until rules are revoked from BOTH).
#   4. Delete each SG.
# ────────────────────────────────────────────────────────────────────────────
sweep_orphan_sagemaker_sgs() {
    local action_label="${1:-sweep}"
    echo ""
    echo "Sweeping orphaned SageMaker NFS security groups ($action_label)..."

    # Get the VPC owned by the stack.
    local vpc_id
    vpc_id=$(aws cloudformation list-stack-resources --stack-name "$STACK_NAME" --region "$REGION" \
        --query "StackResourceSummaries[?ResourceType=='AWS::EC2::VPC'].PhysicalResourceId | [0]" \
        --output text 2>/dev/null || echo "")
    if [ -z "$vpc_id" ] || [ "$vpc_id" = "None" ]; then
        echo "  (no VPC in stack — nothing to sweep)"
        return 0
    fi

    # SageMaker creates SG names matching `security-group-for-*-nfs-d-*`.
    local sg_list
    sg_list=$(aws ec2 describe-security-groups --region "$REGION" \
        --filters "Name=vpc-id,Values=$vpc_id" \
                  "Name=group-name,Values=security-group-for-*-nfs-d-*" \
        --query "SecurityGroups[].GroupId" --output text 2>/dev/null || echo "")

    if [ -z "$sg_list" ]; then
        echo "  ✓ No orphan SageMaker NFS security groups in stack VPC."
        return 0
    fi

    echo "  Found orphan security groups:"
    for sg in $sg_list; do
        local name
        name=$(aws ec2 describe-security-groups --group-ids "$sg" --region "$REGION" \
            --query "SecurityGroups[0].GroupName" --output text 2>/dev/null || echo "?")
        echo "    - $sg ($name)"
    done

    if $DRY_RUN; then
        for sg in $sg_list; do
            echo "  [DRY-RUN] would strip rules + delete $sg"
        done
        return 0
    fi

    # Two-pass: strip rules from ALL, then delete ALL. They reference each
    # other so we can't delete the first while the second still has a rule
    # pointing at it.
    for sg in $sg_list; do
        # Revoke ingress (if any)
        local ingress
        ingress=$(aws ec2 describe-security-groups --group-ids "$sg" --region "$REGION" \
            --query "SecurityGroups[0].IpPermissions" --output json 2>/dev/null || echo "[]")
        if [ "$ingress" != "[]" ] && [ -n "$ingress" ]; then
            echo "$ingress" | aws ec2 revoke-security-group-ingress \
                --group-id "$sg" --region "$REGION" --ip-permissions file:///dev/stdin >/dev/null 2>&1 \
                || echo "    ⚠ could not revoke ingress on $sg"
        fi
        # Revoke egress (if any non-default rules)
        local egress
        egress=$(aws ec2 describe-security-groups --group-ids "$sg" --region "$REGION" \
            --query "SecurityGroups[0].IpPermissionsEgress" --output json 2>/dev/null || echo "[]")
        if [ "$egress" != "[]" ] && [ -n "$egress" ]; then
            echo "$egress" | aws ec2 revoke-security-group-egress \
                --group-id "$sg" --region "$REGION" --ip-permissions file:///dev/stdin >/dev/null 2>&1 \
                || echo "    ⚠ could not revoke egress on $sg"
        fi
    done

    # Now delete them.
    for sg in $sg_list; do
        if aws ec2 delete-security-group --group-id "$sg" --region "$REGION" 2>/dev/null; then
            echo "  ✓ Deleted $sg"
        else
            echo "  ⚠ Could not delete $sg — check for other resources still referencing it"
        fi
    done
}

# Combined orphan sweep — EFS first (so subnets can go), then SGs (so VPC can go).
# Called from the DELETE_FAILED recovery path AND from the wait-loop auto-retry.
sweep_all_orphans() {
    local label="${1:-sweep}"
    sweep_orphan_efs "$label"
    sweep_orphan_sagemaker_sgs "$label"
}

# ────────────────────────────────────────────────────────────────────────────
# 4. Revoke Lake Formation grants on the Athena DB (best-effort).
# ────────────────────────────────────────────────────────────────────────────
echo ""
_ATHENA_DB="$(get_config ATHENA_DATABASE)"
echo "[4/6] Lake Formation grants on Athena database '$_ATHENA_DB' (best-effort)..."
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
# 4b. If we're recovering from DELETE_FAILED, sweep orphaned EFS mount
#     targets NOW — they're the most common cause and CFN can't drain them.
# ────────────────────────────────────────────────────────────────────────────
if $IS_RETRY_FROM_DELETE_FAILED; then
    sweep_all_orphans "DELETE_FAILED recovery"
fi

# ────────────────────────────────────────────────────────────────────────────
# 5. Issue delete-stack.
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Stack delete..."
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
# 6. Wait for delete (only in execute mode). CFN's wait timeout is 3 hours;
#    the MLflow tracking server alone takes ~10 min. We poll every 30s and
#    surface status transitions instead of staring at a silent wait command.
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Wait for delete..."
if $DRY_RUN; then
    echo "  [DRY-RUN] would poll for DELETE_COMPLETE (typically 10-20 min)"
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Dry-run complete. Re-run with --execute to actually delete.       ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    exit 0
fi

LAST_STATUS=""
AUTO_RETRIED_EFS=false   # one-shot guard so we don't loop infinitely
while true; do
    STATUS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
        --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "DELETE_COMPLETE")
    if [ "$STATUS" = "DELETE_COMPLETE" ]; then
        echo "  ✓ Stack deleted"
        break
    fi
    if [ "$STATUS" = "DELETE_FAILED" ]; then
        # Check whether the failure is the known "has dependencies" pattern
        # on a subnet (EFS mount target orphan) or VPC (SageMaker SG orphan).
        # If so, sweep + retry ONCE (the auto-retry guard).
        DEP_FAIL=$(aws cloudformation describe-stack-events --stack-name "$STACK_NAME" --region "$REGION" \
            --query "StackEvents[?ResourceStatus=='DELETE_FAILED' && (contains(ResourceStatusReason, 'subnet') || contains(ResourceStatusReason, 'vpc')) && contains(ResourceStatusReason, 'dependencies')] | length(@)" \
            --output text 2>/dev/null || echo "0")
        if [ "$DEP_FAIL" != "0" ] && ! $AUTO_RETRIED_EFS; then
            echo ""
            echo "  ⚠ Subnet/VPC failed to delete (dependencies present)."
            echo "    Sweeping orphan EFS mount targets + SageMaker NFS"
            echo "    security groups, then retrying stack delete once..."
            sweep_all_orphans "auto-retry after subnet/VPC-dependency failure"
            AUTO_RETRIED_EFS=true
            echo ""
            echo "  Re-issuing delete-stack..."
            aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
            LAST_STATUS=""  # reset so the next transition prints
            sleep 30
            continue
        fi

        echo ""
        echo "  ❌ DELETE_FAILED. Resources that failed to delete:"
        aws cloudformation describe-stack-events --stack-name "$STACK_NAME" --region "$REGION" \
            --query "StackEvents[?ResourceStatus=='DELETE_FAILED'].[LogicalResourceId,ResourceType,ResourceStatusReason]" \
            --output table 2>/dev/null
        echo ""
        echo "  Common fixes:"
        echo "    - 'Bucket not empty' → re-run this script; Step 1 will re-drain it"
        echo "    - 'Role still has attached policies' → manually detach + retry"
        echo "    - 'subnet/vpc has dependencies' → re-run this script; the EFS +"
        echo "      SG sweep retry already ran, so check what else holds the VPC:"
        echo "      aws ec2 describe-network-interfaces --filters Name=vpc-id,Values=..."
        echo "      aws ec2 describe-security-groups   --filters Name=vpc-id,Values=..."
        echo "    - Last resort: aws cloudformation delete-stack --stack-name $STACK_NAME \\"
        echo "          --region $REGION --retain-resources <LogicalResourceId(s)>"
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
