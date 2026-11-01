#!/usr/bin/env bash
# Paper recipe: weak_to_strong_code. See scripts/README.md; use --dry-run to inspect.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
run_recipe weak_to_strong_code "$@"
