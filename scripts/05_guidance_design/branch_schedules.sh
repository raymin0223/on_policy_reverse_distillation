#!/usr/bin/env bash
# Paper recipe: branch_schedules. See scripts/README.md; use --dry-run to inspect.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
run_recipe branch_schedules "$@"
