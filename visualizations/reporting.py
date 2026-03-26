"""Plot confusion matrix."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("metrics_json", type=Path, help="Path to linear_probe_metrics.json")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/figures/confusion_matrix.png"),
        help="Output PNG path",
    )
    p.add_argument("--title", type=str, default="Linear probe - confusion matrix")
    args = p.parse_args()

    # PowerShell `Out-File -Encoding UTF8` often writes a UTF-8 BOM.
    # `utf-8-sig` transparently handles that BOM.
    data = json.loads(args.metrics_json.read_text(encoding="utf-8-sig"))
    cm = np.array(data["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xlabel="Predicted",
        ylabel="True",
        title=args.title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()