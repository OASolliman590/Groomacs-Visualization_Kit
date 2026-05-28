#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-gromacs_pipeline_template.yaml}"

if [[ "${1:-}" == "--dry-run" ]]; then
  exec "$PYTHON_BIN" -m gromacs_analysis.cli.pipeline_cli --config "$CONFIG" --dry-run
fi

exec "$PYTHON_BIN" -m gromacs_analysis.cli.pipeline_cli --config "$CONFIG" "$@"
