#!/usr/bin/env bash
# Paper recipe: strong_to_weak_math. See scripts/README.md; use --dry-run to inspect.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
run_recipe strong_to_weak_math "$@"
