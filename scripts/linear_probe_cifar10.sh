#!/usr/bin/env bash
set -euo
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# You can override this env var; otherwise we try to pick the most recent SimCLR CIFAR-10 last.ckpt.
if [[ -z "${PRETRAINED_CKPT:-}" ]]; then
  PRETRAINED_CKPT="$(ls -t experiments/simclr_cifar10_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$PRETRAINED_CKPT" || ! -f "$PRETRAINED_CKPT" ]]; then
  echo "ERROR: PRETRAINED_CKPT not set or not found."
  echo "Set it like: PRETRAINED_CKPT=/path/to/last.ckpt ./scripts/linear_probe_cifar10.sh"
  exit 1
fi

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME:-linear_probe_cifar10}_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

export HYDRA_FULL_ERROR=1

OVERRIDES=(
  "--config-name=${CONFIG_NAME:-linear_probe_cifar10}"
  "experiment_name=${EXPERIMENT_NAME:-linear_probe_cifar10}"
  "experiment_dir=${EXPERIMENT_DIR}"
  "hydra.run.dir=${EXPERIMENT_DIR}"
  "trainer.max_epochs=${MAX_EPOCHS:-50}"
  "datamodule.batch_size=${BATCH_SIZE:-256}"
  "datamodule.num_workers=${NUM_WORKERS:-8}"
  "datamodule.data_dir=${DATA_DIR:-/mnt/c/data/CIFAR10}"
  "model.pretrained_ckpt=${PRETRAINED_CKPT}"
  "model.optimizer.lr=${PROBE_LR:-1e-2}"
)

echo "Using pretrained checkpoint: ${PRETRAINED_CKPT}"
echo "Writing logs to: ${EXPERIMENT_DIR}/run.log"
START_SEC="$(date +%s)"

python src/train.py "${OVERRIDES[@]}" > "${EXPERIMENT_DIR}/run.log" 2>&1
python src/test.py  "${OVERRIDES[@]}" >> "${EXPERIMENT_DIR}/run.log" 2>&1

END_SEC="$(date +%s)"
ELAPSED_SEC="$((END_SEC - START_SEC))"
echo "Run completed in ${ELAPSED_SEC}s" >> "${EXPERIMENT_DIR}/run.log"

echo "Experiment dir: ${EXPERIMENT_DIR}"

