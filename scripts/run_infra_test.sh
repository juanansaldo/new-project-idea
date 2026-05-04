#!/usr/bin/env bash
set -euo
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
export HYDRA_FULL_ERROR=1

TOTAL_START_SEC="$(date +%s)"

runs=(
  "config:mnist:${DATA_ROOT:-/mnt/c/data}/MNIST"
  "resnet18_cifar10:resnet18_cifar10:${DATA_ROOT:-/mnt/c/data}/CIFAR10"
  "resnet50_cifar10:resnet50_cifar10:${DATA_ROOT:-/mnt/c/data}/CIFAR10"
  "simclr_cifar10:simclr_cifar10:${DATA_ROOT:-/mnt/c/data}/CIFAR10"
)

for run in "${runs[@]}"; do
  IFS=":" read -r configName experimentName dataDir <<< "$run"
  experimentDir="${PROJECT_ROOT}/experiments/${experimentName}_${TIMESTAMP}"
  mkdir -p "$experimentDir"

  overrides=(
    "--config-name=${configName}"
    "experiment_name=${experimentName}"
    "experiment_dir=${experimentDir}"
    "hydra.run.dir=${experimentDir}"
    "trainer.max_epochs=${MAX_EPOCHS:-2}"
    "datamodule.batch_size=${BATCH_SIZE:-128}"
    "datamodule.num_workers=${NUM_WORKERS:-0}"
    "datamodule.data_dir=${dataDir}"
  )

  echo "--- Running ${configName} ---"
  START_SEC="$(date +%s)"

  python src/train.py "${overrides[@]}" > "${experimentDir}/run.log" 2>&1

  if [[ "${configName}" == simclr* ]]; then
    echo "Test skipped (SimCLR has no test dataloader)." >> "${experimentDir}/run.log"
  else
    python src/test.py "${overrides[@]}" >> "${experimentDir}/run.log" 2>&1
  fi

  END_SEC="$(date +%s)"
  ELAPSED_SEC="$((END_SEC - START_SEC))"
  echo "Run completed in ${ELAPSED_SEC}s" >> "${experimentDir}/run.log"
  echo "  Completed in ${ELAPSED_SEC}s"
done

TOTAL_END_SEC="$(date +%s)"
TOTAL_ELAPSED_SEC="$((TOTAL_END_SEC - TOTAL_START_SEC))"
echo "All runs completed in ${TOTAL_ELAPSED_SEC}s"

