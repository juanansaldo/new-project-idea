#!/usr/bin/env bash
set -euo
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME:-resnet50_imagenet}_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

export HYDRA_FULL_ERROR=1

OVERRIDES=(
  "--config-name=${CONFIG_NAME:-resnet50_imagenet}"
  "experiment_name=${EXPERIMENT_NAME:-resnet50_imagenet}"
  "experiment_dir=${EXPERIMENT_DIR}"
  "hydra.run.dir=${EXPERIMENT_DIR}"
  "trainer.max_epochs=${MAX_EPOCHS:-2}"
  "datamodule.batch_size=${BATCH_SIZE:-128}"
  "datamodule.num_workers=${NUM_WORKERS:-8}"
  "datamodule.data_dir=${DATA_DIR:-/mnt/c/data/IMAGENET1K_tar}"
)

echo "Writing logs to: ${EXPERIMENT_DIR}/run.log"
START_SEC="$(date +%s)"

python src/train.py "${OVERRIDES[@]}" > "${EXPERIMENT_DIR}/run.log" 2>&1
python src/test.py  "${OVERRIDES[@]}" >> "${EXPERIMENT_DIR}/run.log" 2>&1

END_SEC="$(date +%s)"
ELAPSED_SEC="$((END_SEC - START_SEC))"
echo "Run completed in ${ELAPSED_SEC}s" >> "${EXPERIMENT_DIR}/run.log"

echo "Experiment dir: ${EXPERIMENT_DIR}"

