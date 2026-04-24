import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"EPOCH\s+(\d+),\s*lr:\s*([^\s]+)")
TOTAL_RE = re.compile(r"[A-Za-z]*Total loss:\s*([-+eE0-9\.]+)")
ACTION_RE = re.compile(r"Action loss:\s*([-+eE0-9\.]+)")
FEATURE_RE = re.compile(r"Feature loss:\s*([-+eE0-9\.]+)")
DIFF_RE = re.compile(r"Diff action feature loss:\s*([-+eE0-9\.]+)")


def parse_log(log_path: Path) -> list[dict]:
    rows = []
    current = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = EPOCH_RE.search(line)
        if match:
            if current is not None and "epoch" in current:
                rows.append(current)
            current = {
                "epoch": int(match.group(1)),
                "lr": float(match.group(2)),
            }
            continue

        if current is None:
            continue

        for key, regex in (
            ("total_loss", TOTAL_RE),
            ("action_loss", ACTION_RE),
            ("feature_loss", FEATURE_RE),
            ("diff_action_feature_loss", DIFF_RE),
        ):
            metric_match = regex.search(line)
            if metric_match:
                current[key] = float(metric_match.group(1))
                break

    if current is not None and "epoch" in current:
        rows.append(current)

    rows = [
        row for row in rows
        if all(
            key in row
            for key in (
                "total_loss",
                "action_loss",
                "feature_loss",
                "diff_action_feature_loss",
            )
        )
    ]
    return rows


def save_summary(rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "afe_training_summary.json"
    payload = {
        "num_epochs": len(rows),
        "epochs": rows,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def plot_losses(rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, [row["total_loss"] for row in rows], label="Total loss", linewidth=2)
    axes[0].plot(epochs, [row["action_loss"] for row in rows], label="Action loss", linewidth=1.8)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("AFE Training Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["feature_loss"] for row in rows],
        label="Feature loss",
        linewidth=1.8,
    )
    axes[1].plot(
        epochs,
        [row["diff_action_feature_loss"] for row in rows],
        label="Diff action feature loss",
        linewidth=1.8,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    figure_path = output_dir / "afe_training_losses.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AFE training log losses.")
    parser.add_argument("log_path", help="Path to the training log text file.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/afe_training_plots",
        help="Directory for output figure and summary json.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = parse_log(log_path)
    if not rows:
        raise ValueError(
            f"No complete epoch/loss records were parsed from log: {log_path}"
        )

    summary_path = save_summary(rows, output_dir)
    figure_path = plot_losses(rows, output_dir)

    print("AFE_LOG_VIS_OK")
    print(f"log: {log_path}")
    print(f"epochs: {len(rows)}")
    print(f"summary: {summary_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
