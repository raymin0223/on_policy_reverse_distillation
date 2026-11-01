#!/usr/bin/env bash
# Paper recipe: guidance_directions_main_teacher. See scripts/README.md; use --dry-run to inspect.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
run_recipe guidance_directions_main_teacher "$@"
