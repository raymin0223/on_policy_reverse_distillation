#!/usr/bin/env bash
# Shared entrypoint: resolve the checkout independently of the calling directory.
set -euo pipefail
OPRD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$OPRD_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-on-policy-reverse-distillation}"
run_recipe() {
  "${PYTHON:-python}" -m oprd.experiments "$@"
}
