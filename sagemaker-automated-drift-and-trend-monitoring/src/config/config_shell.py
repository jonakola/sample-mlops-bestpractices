"""Shell-friendly accessor for config.py constants.

Usage from shell scripts:
    PROJECT_NAME=$(python3 -m src.config.config_shell PROJECT_NAME)
    REGION=$(python3 -m src.config.config_shell AWS_DEFAULT_REGION)
    LAMBDA_NAME=$(python3 -m src.config.config_shell DRIFT_LAMBDA_NAME)

Why this exists:
    Shell scripts (deploy_lambda_container.sh, delete_infrastructure.sh,
    cloudformation/*.sh) need access to values that live in config.py +
    config.yaml. Earlier versions duplicated the defaults — `${VAR:-literal}` —
    which drifted from config.py over time. This module lets shell call into
    Python, so config.py stays the single source of truth.

    Variables come back EXACTLY as config.py resolved them: env-var override,
    then YAML, then the hardcoded fallback in config.py — never a shell-side
    literal.
"""

from __future__ import annotations

import sys

from src.config import config as _cfg


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 -m src.config.config_shell <CONSTANT_NAME>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    if not hasattr(_cfg, name):
        print(f"config.py has no constant named {name!r}", file=sys.stderr)
        return 3
    value = getattr(_cfg, name)
    # Print scalars verbatim; paths and Path objects render as their string form.
    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
