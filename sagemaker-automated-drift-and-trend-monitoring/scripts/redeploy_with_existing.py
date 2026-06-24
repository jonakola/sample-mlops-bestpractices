#!/usr/bin/env python3
"""
Redeploy CloudFormation stack reusing all existing resources from a failed stack.

This script:
1. Reads the failed stack to extract all resource IDs
2. Deletes the old stack (retaining stuck resources)
3. Creates a new stack with the same name, reusing those resources
4. No cleanup required - just reuse everything as-is

Usage:
    # Redeploy with new name
    python scripts/redeploy_with_existing.py --stack-name <failed-stack-name> --new-stack <new-name>

    # Redeploy with same name (deletes old stack first, retaining resources)
    python scripts/redeploy_with_existing.py --stack-name <failed-stack-name>
"""

import argparse
import boto3
import sys
import time
from typing import Dict, Optional, List


def get_existing_resources(cfn_client, stack_name: str) -> Dict[str, str]:
    """Extract all reusable resource IDs from a stack."""
    params = {}

    print(f"Reading stack: {stack_name}")

    # Get stack status first
    try:
        stack_info = cfn_client.describe_stacks(StackName=stack_name)['Stacks'][0]
        stack_status = stack_info['StackStatus']
        print(f"Stack status: {stack_status}")
    except Exception as e:
        print(f"❌ Cannot read stack: {e}")
        return {}

    print("-" * 70)

    try:
        response = cfn_client.describe_stack_resources(StackName=stack_name)

        for resource in response['StackResources']:
            logical_id = resource['LogicalResourceId']
            physical_id = resource.get('PhysicalResourceId', '')
            resource_type = resource['ResourceType']
            status = resource['ResourceStatus']

            if not physical_id:
                continue

            # Extract resource mappings
            if logical_id == 'VPC':
                params['UseExistingVPC'] = 'true'
                params['ExistingVPCId'] = physical_id
                print(f"✓ VPC: {physical_id} ({status})")

            elif logical_id == 'PublicSubnet1':
                params['ExistingSubnet1Id'] = physical_id
                print(f"✓ Subnet1: {physical_id} ({status})")

            elif logical_id == 'PublicSubnet2':
                params['ExistingSubnet2Id'] = physical_id
                print(f"✓ Subnet2: {physical_id} ({status})")

            elif logical_id == 'SecurityGroup':
                params['ExistingSecurityGroupId'] = physical_id
                print(f"✓ SecurityGroup: {physical_id} ({status})")

            elif logical_id == 'SageMakerExecutionRole':
                params['UseExistingRole'] = 'true'
                # Extract role name from ARN or use as-is
                role_name = physical_id.split('/')[-1] if '/' in physical_id else physical_id
                params['ExistingRoleName'] = role_name
                print(f"✓ IAM Role: {role_name} ({status})")

            elif logical_id == 'DataBucket':
                params['UseExistingBucket'] = 'true'
                params['ExistingBucketName'] = physical_id
                print(f"✓ S3 Bucket: {physical_id} ({status})")

        # Get original parameters to preserve ProjectName, etc.
        for param in stack_info.get('Parameters', []):
            key = param['ParameterKey']
            value = param['ParameterValue']

            # Only copy non-existing parameters
            if key not in params and key not in ['UseExistingVPC', 'UseExistingRole', 'UseExistingBucket']:
                params[key] = value
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error reading stack: {e}")
        return {}

    return params


def get_resources_to_retain(cfn_client, stack_name: str) -> List[str]:
    """Get list of physical resource IDs that should be retained."""
    retain_types = [
        'AWS::EC2::VPC',
        'AWS::EC2::Subnet',
        'AWS::EC2::SecurityGroup',
        'AWS::EC2::InternetGateway',
        'AWS::EC2::RouteTable',
        'AWS::EC2::Route',
        'AWS::IAM::Role',
        'AWS::S3::Bucket',
    ]

    resources_to_retain = []

    try:
        response = cfn_client.describe_stack_resources(StackName=stack_name)
        for resource in response['StackResources']:
            resource_type = resource['ResourceType']
            physical_id = resource.get('PhysicalResourceId', '')
            logical_id = resource['LogicalResourceId']
            status = resource['ResourceStatus']

            if resource_type in retain_types and physical_id:
                resources_to_retain.append(logical_id)
                print(f"  Will retain: {logical_id} ({physical_id}) - {status}")

    except Exception as e:
        print(f"⚠️  Could not get resources to retain: {e}")

    return resources_to_retain


def delete_stack_retain_resources(cfn_client, stack_name: str) -> bool:
    """Delete stack but retain key resources."""
    print("\n" + "=" * 70)
    print(f"Deleting old stack: {stack_name}")
    print("=" * 70)

    # Get resources to retain
    print("\nIdentifying resources to retain...")
    resources_to_retain = get_resources_to_retain(cfn_client, stack_name)

    if not resources_to_retain:
        print("⚠️  No resources identified for retention")
        print("Will attempt regular stack deletion...")

    try:
        if resources_to_retain:
            cfn_client.delete_stack(
                StackName=stack_name,
                RetainResources=resources_to_retain
            )
            print(f"\n✓ Deletion initiated (retaining {len(resources_to_retain)} resources)")
        else:
            cfn_client.delete_stack(StackName=stack_name)
            print(f"\n✓ Deletion initiated")

        # Wait for deletion to complete
        print("Waiting for stack deletion...")
        waiter = cfn_client.get_waiter('stack_delete_complete')

        try:
            waiter.wait(
                StackName=stack_name,
                WaiterConfig={'Delay': 10, 'MaxAttempts': 60}
            )
            print("✓ Stack deleted successfully")
            return True
        except Exception as e:
            # Check if stack is gone
            try:
                cfn_client.describe_stacks(StackName=stack_name)
                print(f"⚠️  Stack deletion may have failed: {e}")
                return False
            except cfn_client.exceptions.ClientError as ce:
                if 'does not exist' in str(ce):
                    print("✓ Stack deleted successfully")
                    return True
                raise

    except Exception as e:
        print(f"❌ Failed to delete stack: {e}")
        return False


def create_stack_with_existing(cfn_client, new_stack_name: str, params: Dict[str, str], template_path: str) -> bool:
    """Create a new CloudFormation stack with existing resources."""

    print("\n" + "=" * 70)
    print(f"Creating new stack: {new_stack_name}")
    print("=" * 70)

    # Read template
    try:
        with open(template_path, 'r') as f:
            template_body = f.read()
    except FileNotFoundError:
        print(f"❌ Template not found: {template_path}")
        return False

    # Convert params dict to CloudFormation parameter format
    cf_params = [
        {'ParameterKey': k, 'ParameterValue': v}
        for k, v in params.items()
    ]

    print("\nParameters:")
    for p in cf_params:
        print(f"  {p['ParameterKey']}: {p['ParameterValue']}")

    try:
        response = cfn_client.create_stack(
            StackName=new_stack_name,
            TemplateBody=template_body,
            Parameters=cf_params,
            Capabilities=['CAPABILITY_NAMED_IAM'],
            OnFailure='ROLLBACK'
        )

        stack_id = response['StackId']
        print(f"\n✓ Stack creation initiated!")
        print(f"  Stack ID: {stack_id}")
        print(f"\nMonitor progress:")
        print(f"  aws cloudformation describe-stack-events --stack-name {new_stack_name}")
        print(f"\nOr in AWS Console:")
        print(f"  https://console.aws.amazon.com/cloudformation/home#/stacks")

        return True

    except Exception as e:
        print(f"❌ Failed to create stack: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Redeploy CloudFormation stack reusing existing resources',
        epilog='''
Examples:
  # Redeploy with same name in us-east-1
  %(prog)s --stack-name my-stack --region us-east-1

  # Redeploy with new name
  %(prog)s --stack-name my-stack --new-stack my-stack-v2 --region us-west-2

  # Dry-run to see what would happen
  %(prog)s --stack-name my-stack --region us-east-1 --dry-run
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--stack-name',
        required=True,
        help='Name of the existing stack (any status: CREATE_COMPLETE, UPDATE_COMPLETE, DELETE_FAILED, etc.)'
    )
    parser.add_argument(
        '--new-stack',
        help='Name for the new stack (if different). If omitted, will delete old stack and recreate with same name.'
    )
    parser.add_argument(
        '--region',
        required=True,
        help='AWS region (e.g., us-east-1, us-west-2)'
    )
    parser.add_argument(
        '--template',
        default='cloudformation/sagemaker-mlflow-setup.yaml',
        help='Path to CloudFormation template (default: cloudformation/sagemaker-mlflow-setup.yaml)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    # Create boto3 client with specified region
    print(f"Using region: {args.region}")
    cfn = boto3.client('cloudformation', region_name=args.region)

    # Determine new stack name
    new_stack_name = args.new_stack if args.new_stack else args.stack_name
    reuse_same_name = (new_stack_name == args.stack_name)

    # Step 1: Extract existing resources
    print("\n" + "=" * 70)
    print("STEP 1: Extract existing resources")
    print("=" * 70)
    params = get_existing_resources(cfn, args.stack_name)

    if not params:
        print("\n❌ No resources found to reuse")
        return 1

    print(f"\n✓ Found {len(params)} parameters")

    if args.dry_run:
        print("\n[DRY RUN] Would perform the following:")
        if reuse_same_name:
            print(f"  1. Delete stack: {args.stack_name} (retaining resources)")
            print(f"  2. Create stack: {new_stack_name} (reusing resources)")
        else:
            print(f"  1. Create stack: {new_stack_name} (reusing resources from {args.stack_name})")
        print("\nParameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        return 0

    # Step 2: If reusing same name, delete old stack first
    if reuse_same_name:
        print("\n" + "=" * 70)
        print("STEP 2: Delete old stack (retaining resources)")
        print("=" * 70)
        print(f"\n⚠️  About to delete stack '{args.stack_name}' and recreate with same name")
        print("Resources will be retained and reused.")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted")
            return 0

        if not delete_stack_retain_resources(cfn, args.stack_name):
            print("\n❌ Failed to delete old stack")
            return 1

        print("\nWaiting 10 seconds before recreating...")
        time.sleep(10)

    else:
        print("\n" + "=" * 70)
        print("STEP 2: Create new stack with different name")
        print("=" * 70)
        print(f"\n⚠️  About to create new stack '{new_stack_name}'")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted")
            return 0

    # Step 3: Create new stack
    print("\n" + "=" * 70)
    print(f"STEP 3: Create stack '{new_stack_name}'")
    print("=" * 70)
    success = create_stack_with_existing(cfn, new_stack_name, params, args.template)

    if success:
        print("\n" + "=" * 70)
        print("✓ SUCCESS!")
        print("=" * 70)
        print(f"\nStack '{new_stack_name}' is being created with existing resources.")
        print("No cleanup or ENI deletion was needed.")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
