"""
Grant Lake Formation permissions on Athena tables used by this project.

Why this is a separate script (not in CFN):
    CFN can model `AWS::LakeFormation::Permissions` resources, BUT the grant
    needs to target the human/SSO role of *whoever is running the notebook*.
    That principal is unknown at stack-deploy time (different developers, CI
    runs, etc.), so the grant must happen at runtime. The notebook calls
    this script in Section 5.1.

Why this is a separate script (not inline in the notebook):
    The original cell was ~100 lines of role-ARN munging + grant-loop
    boilerplate that visually dominated the section. The actual user-facing
    operation is a single line: "grant me + the Lambda role access to the
    catalog if LF is in managed mode". This script encapsulates that.

Three-way behavior depending on the account's Lake Formation mode:
  1. **LF-managed mode** (IAM_ALLOWED_PRINCIPALS NOT in default DB permissions)
     → grant SELECT/DESCRIBE/INSERT/ALTER on every project table to the caller,
       and grant the same on monitoring_responses to LAMBDA_EXEC_ROLE.
  2. **IAM-only mode** (IAM_ALLOWED_PRINCIPALS IS in defaults)
     → skip grants (no-ops; Glue/Athena fall back to IAM).
  3. **Caller lacks lakeformation:GetDataLakeSettings**
     → assume LF is not used; skip grants.
"""
from __future__ import annotations

import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError


PROJECT_TABLES = [
    "training_data",
    "evaluation_data",
    "ground_truth",
    "inference_responses",
    "drifted_data",
    "ground_truth_updates",
    "monitoring_responses",
]
CALLER_PERMISSIONS = ["SELECT", "DESCRIBE", "INSERT", "ALTER"]


def _resolve_iam_role_arn(sts_client) -> tuple[str, str]:
    """Return (caller_arn, account_id). Strips assumed-role session suffix."""
    identity = sts_client.get_caller_identity()
    caller_arn = identity["Arn"]
    account_id = identity["Account"]

    # Lake Formation rejects assumed-role session ARNs (sts::… → iam::… needed).
    if ":assumed-role/" in caller_arn:
        parts = caller_arn.split("/")
        role_name = parts[1]
        if role_name.startswith("AWSReservedSSO_"):
            caller_arn = (
                f"arn:aws:iam::{account_id}:role/aws-reserved/sso.amazonaws.com/{role_name}"
            )
        else:
            caller_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    return caller_arn, account_id


def _is_lf_managed(lf_client) -> Optional[bool]:
    """Detect whether the Glue catalog is in LF-managed mode.

    Returns:
        True  — LF-managed (grants required)
        False — IAM-only or caller lacks LF permissions (grants are no-ops)
        None  — unexpected error (caller should treat as False but log)
    """
    try:
        settings = lf_client.get_data_lake_settings()
        defaults = settings.get("DataLakeSettings", {}).get(
            "CreateDatabaseDefaultPermissions", []
        )
        principals = [
            p.get("Principal", {}).get("DataLakePrincipalIdentifier", "")
            for p in defaults
        ]
        return "IAM_ALLOWED_PRINCIPALS" not in principals
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDeniedException", "AccessDenied"):
            return False
        raise


def _grant(lf_client, principal_arn: str, database: str, table: str, permission: str) -> str:
    """Grant one permission. Returns a status emoji string."""
    try:
        lf_client.grant_permissions(
            Principal={"DataLakePrincipalIdentifier": principal_arn},
            Resource={"Table": {"DatabaseName": database, "Name": table}},
            Permissions=[permission],
        )
        return f"✓ {permission} on {table}"
    except Exception as e:
        msg = str(e).lower()
        if "alreadyexistsexception" in str(type(e).__name__).lower() or "already" in msg:
            return f"✓ {permission} on {table} (already granted)"
        return f"✗ {permission} on {table}: {e}"


def grant_lake_formation_permissions(
    database: str,
    region: str,
    lambda_role_arn: Optional[str] = None,
    monitoring_table: str = "monitoring_responses",
) -> None:
    """Grant LF permissions in LF-managed accounts; skip cleanly otherwise.

    Args:
        database: Athena/Glue database name (e.g. "fraud_detection").
        region: AWS region (e.g. "us-west-2").
        lambda_role_arn: ARN of the Lambda execution role that writes to
            monitoring_responses. Pass an empty string or None to skip the
            Lambda grant.
        monitoring_table: Name of the monitoring_responses table.
    """
    lf = boto3.client("lakeformation", region_name=region)
    sts = boto3.client("sts", region_name=region)

    lf_managed = _is_lf_managed(lf)
    if not lf_managed:
        print("ℹ Lake Formation is not in managed mode for this catalog — grants are no-ops; skipping.")
        return

    caller_arn, _ = _resolve_iam_role_arn(sts)
    print(f"Granting permissions to caller: {caller_arn}")
    print(f"Database: {database}\n")
    for table in PROJECT_TABLES:
        for perm in CALLER_PERMISSIONS:
            print(f"  {_grant(lf, caller_arn, database, table, perm)}")

    if lambda_role_arn:
        print(f"\nGranting Lambda role permissions: {lambda_role_arn}")
        for perm in CALLER_PERMISSIONS:
            print(f"  {_grant(lf, lambda_role_arn, database, monitoring_table, perm)}")
    else:
        print("\n⚠ lambda_role_arn not provided — skipping Lambda role grants.")

    print("\n✓ Lake Formation grants complete")


if __name__ == "__main__":
    from src.config.config import ATHENA_DATABASE, AWS_DEFAULT_REGION, LAMBDA_EXEC_ROLE

    grant_lake_formation_permissions(
        database=ATHENA_DATABASE,
        region=AWS_DEFAULT_REGION,
        lambda_role_arn=LAMBDA_EXEC_ROLE,
        monitoring_table=os.getenv("MONITORING_TABLE_NAME", "monitoring_responses"),
    )
