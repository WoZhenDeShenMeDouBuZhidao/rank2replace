#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec ../.venv/bin/python -m rank2replace.run --experiment 3 "$@"
