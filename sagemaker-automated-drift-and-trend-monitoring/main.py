#!/usr/bin/env python3
"""
Scriptable CLI for the fraud-detection training pipeline.

Designed for CI/CD and headless workflows. The full demo flow (training,
deployment, monitoring, dashboards) lives in the notebooks under `notebooks/`;
this CLI exposes just two thin wrappers that don't need a Jupyter kernel:

  setup     One-shot Athena infrastructure setup (`src/setup/setup_athena_tables.py`)
  pipeline  Create, start, list, describe, version, or delete the SageMaker
            training pipeline (`src/train_pipeline/pipeline_cli.py`)

Deployment, endpoint testing, batch transform, drift monitoring, and dashboards
are intentionally NOT in this CLI — they require the lineage, lake-formation,
and configuration logic that only the notebooks reliably produce.

Usage:
    python main.py setup --migrate-data
    python main.py pipeline create --pipeline-name fraud-detection-pipeline
    python main.py pipeline start  --pipeline-name fraud-detection-pipeline --wait
    python main.py pipeline list   --pipeline-name fraud-detection-pipeline
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
    from src.setup.setup_athena_tables import main as setup_main

    sys.argv = ['setup_athena_tables.py']
    if args.migrate_data:
        sys.argv.append('--migrate-data')
    if args.verify_only:
        sys.argv.append('--verify-only')
    if args.skip_s3:
        sys.argv.append('--skip-s3')
    if args.region:
        sys.argv.extend(['--region', args.region])

    return setup_main()


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
  python main.py setup --migrate-data
  python main.py pipeline create --pipeline-name fraud-detection-pipeline
  python main.py pipeline start  --pipeline-name fraud-detection-pipeline --wait
  python main.py pipeline list   --pipeline-name fraud-detection-pipeline

Note:
  Deployment, endpoint testing, monitoring, and dashboards live in the
  notebooks under `notebooks/`. They depend on lineage and configuration
  context that isn't reproducible from the command line.
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ---------- setup ----------
    setup_parser = subparsers.add_parser('setup', help='Setup Athena infrastructure')
    setup_parser.add_argument('--migrate-data', action='store_true',
                              help='Migrate CSV data to Athena tables')
    setup_parser.add_argument('--verify-only', action='store_true',
                              help='Only verify existing setup')
    setup_parser.add_argument('--region', default='us-east-1',
                              help='AWS region (default: us-east-1)')
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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'setup':
            return setup_command(args)
        elif args.command == 'pipeline':
            return pipeline_command(args)
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
