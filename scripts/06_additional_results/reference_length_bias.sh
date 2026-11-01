#!/usr/bin/env bash
# Paper recipe: reference_length_bias. See scripts/README.md; use --dry-run to inspect.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
run_recipe reference_length_bias "$@"
