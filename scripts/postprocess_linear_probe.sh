#!/usr/bin/env bash
set -euo
set -o pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/experiments/<run_dir>"
  exit 1
fi

EXPERIMENT_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
  echo "ExperimentDir does not exist: $EXPERIMENT_DIR" >&2
  mkdir -p "$(dirname "$EXPERIMENT_DIR")/$(basename "$EXPERIMENT_DIR")" 2>/dev/null || true
  exit 1
fi

RUN_TAG="$(basename "$EXPERIMENT_DIR")"

METRICS_PATH="${EXPERIMENT_DIR}/linear_probe_metrics.json"
if [[ ! -f "$METRICS_PATH" ]]; then
  echo "Metrics file not found: $METRICS_PATH" >&2
  exit 1
fi

REPORTS_DIR="${EXPERIMENT_DIR}/reports"
FIGURES_DIR="${REPORTS_DIR}/figures"
mkdir -p "$FIGURES_DIR"

FIG_OUT="${FIGURES_DIR}/${RUN_TAG}_cm.png"
SUMMARY_OUT="${REPORTS_DIR}/${RUN_TAG}.md"
MANIFEST_OUT="${EXPERIMENT_DIR}/manifest.json"

echo "Using metrics: $METRICS_PATH"
echo "Generating confusion matrix: $FIG_OUT"

python "${PROJECT_ROOT}/visualizations/reporting.py" \
  "$METRICS_PATH" \
  --out "$FIG_OUT" \
  --title "Linear Probe CIFAR-10 Confusion Matrix (${RUN_TAG})"

python - "$METRICS_PATH" "$SUMMARY_OUT" "$FIG_OUT" "$MANIFEST_OUT" <<'PY'
import json, sys
metrics_path, summary_out, fig_out, manifest_out = sys.argv[1:5]

with open(metrics_path, "r", encoding="utf-8-sig") as f:
    data = json.load(f)

test_acc = data.get("test_accuracy", None)
report = data.get("classification_report", {})
cm = data.get("confusion_matrix", None)

class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

run_dir = metrics_path.rsplit("/", 1)[0]
lines = []
lines.append("# Linear Probe Run Summary - " + run_dir.split("/")[-1])
lines.append("")
lines.append("## Artifacts")
lines.append(f"- Experiment: {run_dir}")
lines.append(f"- Metrics JSON: {metrics_path}")
lines.append(f"- Confusion matrix: {fig_out}")
lines.append("")
lines.append("## Key Metrics")
if test_acc is not None:
    lines.append(f"- Test accuracy: {test_acc*100:.2f}%")

macro = report.get("macro avg", {})
weighted = report.get("weighted avg", {})
if macro:
    lines.append(f"- Macro F1: {macro.get('f1-score', None):.4f}")
if weighted:
    lines.append(f"- Weighted F1: {weighted.get('f1-score', None):.4f}")

lines.append("")
lines.append("## Per-class Metrics (CIFAR-10)")
lines.append("| Class | Precision | Recall | F1-score | Support |")
lines.append("|---|---:|---:|---:|---:|")

for i, name in enumerate(class_names):
    row = report.get(str(i))
    if not row:
        continue
    lines.append(
        f"| {name} | {row.get('precision',0):.4f} | {row.get('recall',0):.4f} | {row.get('f1-score',0):.4f} | {int(row.get('support',0))} |"
    )

summary = "\n".join(lines) + "\n"
with open(summary_out, "w", encoding="utf-8") as f:
    f.write(summary)

manifest = {
    "run_tag": run_dir.split("/")[-1],
    "status": "success",
    "experiment_dir": run_dir,
    "metrics_json": metrics_path,
    "report_markdown": summary_out,
    "confusion_matrix_png": fig_out,
    "test_accuracy": test_acc,
}
with open(manifest_out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("Wrote summary:", summary_out)
print("Wrote manifest:", manifest_out)
PY

echo "Done."
echo "Summary: $SUMMARY_OUT"
echo "Figure:  $FIG_OUT"

