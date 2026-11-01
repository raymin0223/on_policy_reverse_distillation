#!/usr/bin/env bash
# Paper recipe: alignment_branches. See scripts/README.md; use --dry-run to inspect.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
run_recipe alignment_branches "$@"
