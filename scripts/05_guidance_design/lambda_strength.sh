#!/usr/bin/env bash
# Paper recipe: lambda_strength. See scripts/README.md; use --dry-run to inspect.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
run_recipe lambda_strength "$@"
