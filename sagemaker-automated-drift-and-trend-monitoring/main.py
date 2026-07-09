#!/usr/bin/env python3
"""
Scriptable CLI for the fraud-detection training pipeline.

Designed for CI/CD and headless workflows. The notebooks under `notebooks/`
remain the recommended path for first-time setup; this CLI exposes the
scriptable subset that doesn't need a Jupyter kernel:

  setup               One-shot Athena infrastructure setup
                       (`src/setup/create_athena_tables.py`)
  pipeline            Create, start, list, describe, version, or delete the
                       SageMaker training pipeline
                       (`src/train_pipeline/pipeline_cli.py`)
  deploy              Deploy/update/delete the SageMaker serverless endpoint
                       (`src/train_pipeline/deploy_endpoint.py`)
  test-endpoint       Test a deployed endpoint with sample traffic
                       (`src/train_pipeline/test_endpoint.py`)
  schedule-inference  Manage scheduled real-time inference
                       (`src/setup/setup_scheduled_inference.py`)
  schedule-batch      Manage scheduled batch transform
                       (`src/setup/setup_scheduled_batch_transform.py`)
  monitoring          Deploy/manage the drift-monitor Lambda + EventBridge
                       schedule (`src/drift_monitoring/*`,
                       `scripts/deploy_lambda_container.sh`)
  dashboard           Create/update/delete the QuickSight governance
                       dashboard (`src/governance/create_governance_dashboard.py`)

Usage:
    python main.py setup --force-recreate
    python main.py pipeline create --pipeline-name fraud-detection-pipeline
    python main.py pipeline start  --pipeline-name fraud-detection-pipeline --wait
    python main.py deploy create   --wait
    python main.py monitoring deploy-lambda --alert-email you@example.com
    python main.py dashboard create
"""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables BEFORE any other imports
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_command(args):
    """Run Athena infrastructure setup."""
    from src.setup.create_athena_tables import main as setup_main

    sys.argv = ['create_athena_tables.py']
    if args.verify_only:
        sys.argv.append('--verify-only')
    if args.force_recreate:
        sys.argv.append('--force-recreate')
    if args.skip_s3:
        sys.argv.append('--skip-s3')

    return setup_main()


def schedule_inference_command(args):
    """Manage scheduled real-time inference (EventBridge + Lambda)."""
    from src.setup.setup_scheduled_inference import (
        create_scheduled_inference, delete_scheduled_inference,
    )

    if args.schedule_action == 'delete':
        result = delete_scheduled_inference(
            endpoint_name=args.endpoint_name,
            region=args.region,
        )
    else:
        result = create_scheduled_inference(
            endpoint_name=args.endpoint_name,
            schedule_expression=args.schedule,
            region=args.region,
            athena_database=args.athena_database,
            s3_bucket=args.s3_bucket,
            batch_size=args.batch_size,
            lookback_minutes=args.lookback_minutes,
        )
    print(json.dumps(result, indent=2))
    return 0


def monitoring_command(args):
    """Deploy/manage the drift-monitor Lambda + EventBridge schedule."""
    from src.drift_monitoring.manage_drift_lambda import (
        bootstrap_drift_lambda_role, deploy_drift_lambda_container,
        invoke_drift_lambda, get_drift_lambda_logs,
        update_drift_thresholds, set_drift_schedule_state,
    )
    from src.setup.grant_lake_formation_permissions import grant_lake_formation_permissions
    from src.drift_monitoring.deploy_monitoring_writer import deploy_monitoring_writer
    from src.drift_monitoring.create_cloudwatch_monitoring import create_cloudwatch_monitoring

    action = args.monitoring_action
    if action == 'bootstrap-role':
        result = bootstrap_drift_lambda_role(lambda_exec_role=args.lambda_exec_role, region=args.region)
    elif action == 'grant-lake-formation':
        grant_lake_formation_permissions(
            database=args.athena_database, region=args.region or 'us-east-1',
            lambda_role_arn=args.lambda_role_arn,
        )
        result = {'status': 'done'}
    elif action == 'deploy-writer':
        queue_url = deploy_monitoring_writer(region=args.region or 'us-east-1')
        result = {'queue_url': queue_url}
    elif action == 'deploy-lambda':
        result = deploy_drift_lambda_container(
            alert_email=args.alert_email,
            data_drift_threshold=args.data_drift_threshold,
            model_drift_threshold=args.model_drift_threshold,
            region=args.region,
        )
    elif action == 'deploy-cloudwatch':
        result = create_cloudwatch_monitoring(
            region=args.region or 'us-east-1',
            endpoint_name=args.endpoint_name,
            drift_threshold=args.model_drift_threshold or 0.10,
            psi_threshold=args.data_drift_threshold or 0.2,
        )
    elif action == 'test':
        result = invoke_drift_lambda(lambda_name=args.lambda_name, region=args.region)
    elif action == 'logs':
        result = get_drift_lambda_logs(lambda_name=args.lambda_name, region=args.region, limit=args.limit)
    elif action == 'update-thresholds':
        result = update_drift_thresholds(
            lambda_name=args.lambda_name, region=args.region,
            data_drift_threshold=args.data_drift_threshold,
            model_drift_threshold=args.model_drift_threshold,
            merge_with_existing=not args.no_merge,
        )
    elif action == 'enable-schedule':
        result = set_drift_schedule_state('ENABLED', rule_name=args.rule_name, region=args.region)
    elif action == 'disable-schedule':
        result = set_drift_schedule_state('DISABLED', rule_name=args.rule_name, region=args.region)
    print(json.dumps(result, indent=2, default=str))
    return 0


def dashboard_command(args):
    """Create/update/delete the QuickSight governance dashboard."""
    from src.governance.create_governance_dashboard import (
        create_dashboard, delete_dashboard,
    )

    if args.dashboard_action == 'delete':
        if not args.confirm:
            print("WARNING: this will delete the QuickSight dashboard and its datasets. "
                  "Add --confirm to proceed.")
            return 1
        result = delete_dashboard(region=args.region)
    else:
        result = create_dashboard(region=args.region)
    print(json.dumps(result, indent=2, default=str))
    return 0


def schedule_batch_transform_command(args):
    """Manage scheduled batch transform (EventBridge + Lambda + SageMaker)."""
    from src.setup.setup_scheduled_batch_transform import (
        create_scheduled_batch_transform, delete_scheduled_batch_transform,
    )

    if args.schedule_action == 'delete':
        result = delete_scheduled_batch_transform(
            model_name=args.model_name,
            region=args.region,
        )
    else:
        result = create_scheduled_batch_transform(
            model_name=args.model_name,
            schedule_expression=args.schedule,
            region=args.region,
            sagemaker_role=args.sagemaker_role,
            athena_database=args.athena_database,
            s3_bucket=args.s3_bucket,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            max_concurrent=args.max_concurrent,
            input_table=args.input_table,
            lookback_hours=args.lookback_hours,
            row_limit=args.row_limit,
        )
    print(json.dumps(result, indent=2))
    return 0


def deploy_command(args):
    """Deploy, inspect, or delete the SageMaker serverless endpoint."""
    from src.train_pipeline.deploy_endpoint import (
        deploy, get_endpoint_status, delete_endpoint,
    )

    if args.deploy_action == 'create':
        result = deploy(
            endpoint_name=args.endpoint_name,
            model_package_group=args.model_package_group,
            model_version=args.model_version,
            region=args.region,
            role=args.role,
            memory_size_mb=args.memory_size_mb,
            max_concurrency=args.max_concurrency,
            redeploy_clean=not args.no_clean_redeploy,
            wait=not args.no_wait,
        )
    elif args.deploy_action == 'status':
        result = get_endpoint_status(args.endpoint_name)
    elif args.deploy_action == 'delete':
        if not args.confirm:
            print("WARNING: this will delete the endpoint permanently. Add --confirm to proceed.")
            return 1
        result = delete_endpoint(args.endpoint_name)
    print(json.dumps(result, indent=2, default=str))
    return 0


def test_endpoint_command(args):
    """Test a deployed endpoint with sample traffic."""
    from src.train_pipeline.test_endpoint import test_endpoint

    result = test_endpoint(
        endpoint_name=args.endpoint_name,
        num_samples=args.num_samples,
        data_source=args.data_source,
        test_data_path=args.test_data_path,
        enable_analytics=not args.disable_analytics,
        time_window_minutes=args.time_window,
    )
    print(json.dumps({'summary': result.get('realtime', {})}, indent=2, default=str))
    return 0


def pipeline_command(args):
    """Manage the SageMaker training pipeline."""
    from src.train_pipeline.pipeline_cli import (
        create_pipeline, start_execution, list_executions,
        describe_execution, list_pipeline_versions, delete_pipeline,
    )

    if args.pipeline_action == 'create':
        result = create_pipeline(
            pipeline_name=args.pipeline_name,
            region=args.region,
            role=args.role,
            include_deployment=not args.no_deployment,
        )
        print(json.dumps(result, indent=2))

    elif args.pipeline_action == 'start':
        result = start_execution(
            pipeline_name=args.pipeline_name,
            execution_name=args.execution_name,
            parameters=args.parameters,
            wait=args.wait,
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.pipeline_action == 'list':
        results = list_executions(
            pipeline_name=args.pipeline_name,
            max_results=args.max_results,
        )
        print(json.dumps(results, indent=2))

    elif args.pipeline_action == 'describe':
        result = describe_execution(args.execution_arn)
        print(json.dumps(result, indent=2))

    elif args.pipeline_action == 'versions':
        results = list_pipeline_versions(
            pipeline_name=args.pipeline_name,
            max_results=args.max_results,
        )
        print(json.dumps(results, indent=2))

    elif args.pipeline_action == 'delete':
        if not args.confirm:
            print("WARNING: this will delete the pipeline permanently. Add --confirm to proceed.")
            return 1
        result = delete_pipeline(args.pipeline_name)
        print(json.dumps(result, indent=2))

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SageMaker fraud-detection pipeline — scriptable subset for CI/CD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup --force-recreate
  python main.py pipeline create --pipeline-name fraud-detection-pipeline
  python main.py pipeline start  --pipeline-name fraud-detection-pipeline --wait
  python main.py deploy create --wait
  python main.py deploy status --endpoint-name fraud-detector-endpoint
  python main.py test-endpoint --endpoint-name fraud-detector-endpoint --num-samples 50
  python main.py schedule-inference create --endpoint-name fraud-detector-endpoint
  python main.py schedule-batch create --model-name fraud-detection
  python main.py monitoring deploy-lambda --alert-email you@example.com
  python main.py dashboard create

Note:
  The notebooks under `notebooks/` remain the recommended first-time setup
  path. This CLI is the scriptable subset for CI/CD.
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ---------- setup ----------
    setup_parser = subparsers.add_parser('setup', help='Setup Athena infrastructure')
    setup_parser.add_argument('--verify-only', action='store_true',
                              help='Only verify existing setup; create nothing')
    setup_parser.add_argument('--force-recreate', action='store_true',
                              help='Drop and recreate all tables against the current '
                                   'dataset_schema.yaml (destructive)')
    setup_parser.add_argument('--skip-s3', action='store_true',
                              help='Skip S3 bucket creation')

    # ---------- pipeline ----------
    pipeline_parser = subparsers.add_parser('pipeline', help='Manage SageMaker Pipelines')
    pipeline_parser.add_argument(
        'pipeline_action',
        choices=['create', 'start', 'list', 'describe', 'versions', 'delete'],
        help='Pipeline action',
    )
    pipeline_parser.add_argument('--pipeline-name', default='fraud-detection-pipeline',
                                 help='Pipeline name (default: fraud-detection-pipeline)')
    pipeline_parser.add_argument('--region', default='us-east-1',
                                 help='AWS region (default: us-east-1)')
    pipeline_parser.add_argument('--role',
                                 help='SageMaker execution role ARN')
    pipeline_parser.add_argument('--execution-name',
                                 help='Execution name (for start action)')
    pipeline_parser.add_argument('--execution-arn',
                                 help='Execution ARN (for describe action)')
    pipeline_parser.add_argument('--parameters', type=json.loads,
                                 help='Pipeline parameters as JSON string (for start action)')
    pipeline_parser.add_argument('--wait', action='store_true',
                                 help='Wait for execution to complete (for start action)')
    pipeline_parser.add_argument('--max-results', type=int, default=10,
                                 help='Maximum number of results (for list/versions actions)')
    pipeline_parser.add_argument('--confirm', action='store_true',
                                 help='Confirm deletion (for delete action)')
    pipeline_parser.add_argument('--no-deployment', action='store_true',
                                 help='Exclude deployment + testing steps (for create action)')

    # ---------- deploy ----------
    deploy_parser = subparsers.add_parser(
        'deploy', help='Deploy, inspect, or delete the SageMaker serverless endpoint')
    deploy_parser.add_argument('deploy_action', choices=['create', 'status', 'delete'],
                               help='Deployment action')
    deploy_parser.add_argument('--endpoint-name', help='Endpoint name (default: config.ENDPOINT_NAME)')
    deploy_parser.add_argument('--model-package-group',
                               help='Model Package Group to select from (default: config.MLFLOW_MODEL_NAME)')
    deploy_parser.add_argument('--model-version', type=int,
                               help='Specific approved model version to deploy (default: latest)')
    deploy_parser.add_argument('--region', help='AWS region (default: config.AWS_DEFAULT_REGION)')
    deploy_parser.add_argument('--role', help='SageMaker execution role ARN')
    deploy_parser.add_argument('--memory-size-mb', type=int,
                               help='Serverless memory size in MB (default: config.SERVERLESS_MEMORY_SIZE)')
    deploy_parser.add_argument('--max-concurrency', type=int,
                               help='Max concurrent invocations (default: config.SERVERLESS_MAX_CONCURRENCY)')
    deploy_parser.add_argument('--no-clean-redeploy', action='store_true',
                               help='Update the existing endpoint in place instead of delete-then-recreate '
                                    '(for create action)')
    deploy_parser.add_argument('--no-wait', action='store_true',
                               help="Don't block until the endpoint reaches InService/Failed (for create action)")
    deploy_parser.add_argument('--confirm', action='store_true',
                               help='Confirm deletion (for delete action)')

    # ---------- test-endpoint ----------
    test_ep_parser = subparsers.add_parser('test-endpoint', help='Test a deployed endpoint with sample traffic')
    test_ep_parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    test_ep_parser.add_argument('--num-samples', type=int, default=100, help='Number of test samples (default: 100)')
    test_ep_parser.add_argument('--data-source', choices=['csv', 'athena'], default='csv',
                                help='Test-data source (default: csv)')
    test_ep_parser.add_argument('--test-data-path', help='Path to CSV test data (if data-source=csv)')
    test_ep_parser.add_argument('--disable-analytics', action='store_true',
                                help='Skip querying Athena for aggregated inference analytics')
    test_ep_parser.add_argument('--time-window', type=int, default=60,
                                help='Time window in minutes for Athena analytics (default: 60)')

    # ---------- schedule-inference ----------
    sched_inf_parser = subparsers.add_parser(
        'schedule-inference', help='Manage scheduled real-time inference (EventBridge + Lambda)')
    sched_inf_parser.add_argument('schedule_action', choices=['create', 'delete'],
                                  help='Create or delete the scheduled inference Lambda')
    sched_inf_parser.add_argument('--endpoint-name', required=True,
                                  help='SageMaker endpoint name to invoke')
    sched_inf_parser.add_argument('--schedule', default='rate(1 hour)',
                                  help='EventBridge schedule expression (default: rate(1 hour))')
    sched_inf_parser.add_argument('--region', default='us-east-1', help='AWS region')
    sched_inf_parser.add_argument('--athena-database', default='fraud_detection',
                                  help='Athena database name')
    sched_inf_parser.add_argument('--s3-bucket', help='S3 bucket for data')
    sched_inf_parser.add_argument('--batch-size', type=int, default=100,
                                  help='Transactions per batch (default: 100)')
    sched_inf_parser.add_argument('--lookback-minutes', type=int, default=60,
                                  help='How far back to look for transactions (default: 60)')

    # ---------- schedule-batch ----------
    sched_batch_parser = subparsers.add_parser(
        'schedule-batch', help='Manage scheduled batch transform (EventBridge + Lambda + SageMaker)')
    sched_batch_parser.add_argument('schedule_action', choices=['create', 'delete'],
                                    help='Create or delete the scheduled batch transform Lambda')
    sched_batch_parser.add_argument('--model-name', required=True,
                                    help='SageMaker model or model package group name')
    sched_batch_parser.add_argument('--schedule', default='cron(0 2 * * ? *)',
                                    help='EventBridge schedule expression (default: daily at 2 AM UTC)')
    sched_batch_parser.add_argument('--region', default='us-east-1', help='AWS region')
    sched_batch_parser.add_argument('--sagemaker-role', help='SageMaker execution role ARN')
    sched_batch_parser.add_argument('--athena-database', default='fraud_detection',
                                    help='Athena database name')
    sched_batch_parser.add_argument('--s3-bucket', help='S3 bucket for data')
    sched_batch_parser.add_argument('--instance-type', default='ml.m5.xlarge',
                                    help='EC2 instance type (default: ml.m5.xlarge)')
    sched_batch_parser.add_argument('--instance-count', type=int, default=1,
                                    help='Number of instances (default: 1)')
    sched_batch_parser.add_argument('--max-concurrent', type=int, default=4,
                                    help='Max concurrent transforms (default: 4)')
    sched_batch_parser.add_argument('--input-table', default='training_data',
                                    help='Athena table to read from (default: training_data)')
    sched_batch_parser.add_argument('--lookback-hours', type=int, default=24,
                                    help='Hours of data to process (default: 24)')
    sched_batch_parser.add_argument('--row-limit', type=int, default=10000,
                                    help='Maximum rows per batch (default: 10000)')

    # ---------- monitoring ----------
    monitoring_parser = subparsers.add_parser(
        'monitoring', help='Deploy/manage the drift-monitor Lambda + EventBridge schedule')
    monitoring_parser.add_argument(
        'monitoring_action',
        choices=[
            'bootstrap-role', 'grant-lake-formation', 'deploy-writer', 'deploy-lambda',
            'deploy-cloudwatch', 'test', 'logs', 'update-thresholds',
            'enable-schedule', 'disable-schedule',
        ],
        help='Monitoring infrastructure action',
    )
    monitoring_parser.add_argument('--region', help='AWS region (default: config.AWS_DEFAULT_REGION)')
    monitoring_parser.add_argument('--lambda-name', help='Drift-monitor Lambda name (default: config.DRIFT_LAMBDA_NAME)')
    monitoring_parser.add_argument('--lambda-exec-role', help='Lambda execution role ARN (for bootstrap-role)')
    monitoring_parser.add_argument('--lambda-role-arn', help='Drift Lambda role ARN to grant Lake Formation access to')
    monitoring_parser.add_argument('--athena-database', default='fraud_detection', help='Athena database name')
    monitoring_parser.add_argument('--endpoint-name', help='SageMaker endpoint name (for deploy-cloudwatch)')
    monitoring_parser.add_argument('--alert-email', help='Email to subscribe to drift alerts (for deploy-lambda)')
    monitoring_parser.add_argument('--data-drift-threshold', type=float, help='Data drift (PSI) threshold')
    monitoring_parser.add_argument('--model-drift-threshold', type=float, help='Model drift degradation threshold')
    monitoring_parser.add_argument('--rule-name', help='EventBridge rule name (default: config.EVENTBRIDGE_RULE_NAME)')
    monitoring_parser.add_argument('--limit', type=int, default=50, help='Max log events to fetch (for logs)')
    monitoring_parser.add_argument('--no-merge', action='store_true',
                                   help='Replace the Lambda env vars entirely instead of merging '
                                        '(for update-thresholds) — WARNING: drops unrelated env vars')

    # ---------- dashboard ----------
    dashboard_parser = subparsers.add_parser(
        'dashboard', help='Create/update/delete the QuickSight governance dashboard')
    dashboard_parser.add_argument('dashboard_action', choices=['create', 'delete'],
                                  help='Dashboard action')
    dashboard_parser.add_argument('--region', help='AWS region (default: config.AWS_DEFAULT_REGION)')
    dashboard_parser.add_argument('--confirm', action='store_true',
                                  help='Confirm deletion (for delete action)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'setup':
            return setup_command(args)
        elif args.command == 'pipeline':
            return pipeline_command(args)
        elif args.command == 'schedule-inference':
            return schedule_inference_command(args)
        elif args.command == 'schedule-batch':
            return schedule_batch_transform_command(args)
        elif args.command == 'deploy':
            return deploy_command(args)
        elif args.command == 'test-endpoint':
            return test_endpoint_command(args)
        elif args.command == 'monitoring':
            return monitoring_command(args)
        elif args.command == 'dashboard':
            return dashboard_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
