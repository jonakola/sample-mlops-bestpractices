#!/bin/bash
# ============================================================================
# Read a config constant from src/config/config.py.
#
# This is the SINGLE entry point shell scripts use to learn names, regions,
# schedules, table names, etc. — every default lives in config.py exactly
# once. No shell-side `${VAR:-literal}` fallbacks anywhere in this project.
#
# Precedence (enforced inside config.py, not here):
#   1. Environment variable matching the constant name
#   2. config.yaml value
#   3. Hardcoded fallback in config.py (THE ONLY hardcoded default)
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/_read_config.sh"
#   REGION="$(get_config AWS_DEFAULT_REGION)"
#   LAMBDA_NAME="$(get_config DRIFT_LAMBDA_NAME)"
#   ENDPOINT_NAME="$(get_config ENDPOINT_NAME)"
#
# Requires python3 + pyyaml + python-dotenv (already pulled in by
# pyproject.toml). On a clean machine, `uv pip install -e .` first.
# ============================================================================

_CONFIG_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT_FOR_CONFIG="$(cd "$_CONFIG_HELPER_DIR/.." && pwd)"

# Find a python interpreter that has pyyaml installed. The project venv
# (`.venv/bin/python` via `uv sync` or `pip install -e .`) has it; the system
# python may not. We try, in order: $PYTHON env override → project venv →
# system python3. If none work, the user needs to run `uv pip install -e .`
# from the project root.
_locate_python() {
    if [ -n "${PYTHON:-}" ] && command -v "$PYTHON" >/dev/null 2>&1; then
        echo "$PYTHON"
        return 0
    fi
    if [ -x "$_PROJECT_ROOT_FOR_CONFIG/.venv/bin/python" ]; then
        echo "$_PROJECT_ROOT_FOR_CONFIG/.venv/bin/python"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi
    echo "ERROR: no python interpreter found" >&2
    return 1
}

get_config() {
    # $1 = constant name (e.g. AWS_DEFAULT_REGION, DRIFT_LAMBDA_NAME)
    local name="$1"
    if [ -z "$name" ]; then
        echo "get_config: missing constant name" >&2
        return 2
    fi
    local py
    py="$(_locate_python)" || return $?
    # Run the python accessor from the project root so `src.config.config`
    # imports work without setting PYTHONPATH manually. Errors (unknown
    # constant, missing pyyaml) come through on stderr; success prints the
    # value on stdout.
    (cd "$_PROJECT_ROOT_FOR_CONFIG" && "$py" -m src.config.config_shell "$name")
}
