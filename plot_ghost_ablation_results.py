from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "revision" / "experiments" / "results" / "ghost_ablation" / "ghost_ablation_summary.csv"
OUTPUT_DIR = ROOT / "Upload" / "images"

MODEL_ORDER = [
    "ghostnet_original",
    "ghostnet_no_se",
    "ghostnet_no_shortcut",
    "ghostnet_no_se_no_shortcut",
    "ghostnet_change",
    "ghostnet_simple",
]

MODEL_LABELS = {
    "ghostnet_original": "Original",
    "ghostnet_no_se": "No SE",
    "ghostnet_no_shortcut": "No shortcut",
    "ghostnet_no_se_no_shortcut": "No SE/shortcut",
    "ghostnet_change": "Modified",
    "ghostnet_simple": "Direct stack",
}

MODEL_COLORS = {
    "ghostnet_original": "#8a8f98",
    "ghostnet_no_se": "#3f7cac",
    "ghostnet_no_shortcut": "#4f8f62",
    "ghostnet_no_se_no_shortcut": "#7b5ea7",
    "ghostnet_change": "#c17c39",
    "ghostnet_simple": "#b75d4a",
}


def load_results() -> pd.DataFrame:
    data = pd.read_csv(SUMMARY_CSV)
    required = {
        "model",
        "se",
        "shortcut",
        "bottleneck",
        "direct_stack",
        "mean_target_accuracy",
        "std_target_accuracy",
        "mean_train_accuracy",
        "n_seeds",
        "parameters",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {SUMMARY_CSV}: {sorted(missing)}")

    data = data.copy()
    data["model"] = pd.Categorical(data["model"], MODEL_ORDER, ordered=True)
    return data.sort_values("model")


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / filename
    fig.savefig(output, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Wrote {output}")


def plot_accuracy_errorbars(data: pd.DataFrame) -> None:
    labels = [MODEL_LABELS[str(model)] for model in data["model"]]
    colors = [MODEL_COLORS[str(model)] for model in data["model"]]

    fig, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.bar(
        labels,
        data["mean_target_accuracy"],
        yerr=data["std_target_accuracy"],
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
        capsize=4,
    )
    axis.set_ylim(0.75, 0.94)
    axis.set_ylabel("Mean target accuracy")
    axis.set_title("GhostNet ablation on Sudoku-Digits", fontsize=13, pad=10)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.85)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelrotation=24)

    plot_data = data.reset_index(drop=True)
    best_index = int(plot_data["mean_target_accuracy"].idxmax())
    for index, row in plot_data.iterrows():
        value = row["mean_target_accuracy"]
        std = row["std_target_accuracy"]
        axis.text(
            index,
            min(value + std + 0.006, 0.935),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold" if index == best_index else "normal",
        )

    fig.tight_layout()
    save_figure(fig, "ghostnet_ablation_accuracy.jpeg")


def plot_parameter_tradeoff(data: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8.8, 5.4))
    label_offsets = {
        "ghostnet_original": (-56, -15),
        "ghostnet_no_se": (6, -15),
        "ghostnet_no_shortcut": (-62, 8),
        "ghostnet_no_se_no_shortcut": (6, 7),
        "ghostnet_change": (6, 4),
        "ghostnet_simple": (6, 4),
    }

    for _, row in data.iterrows():
        model = str(row["model"])
        axis.scatter(
            row["parameters"] / 1_000_000,
            row["mean_target_accuracy"],
            s=145,
            color=MODEL_COLORS[model],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.92,
        )
        axis.annotate(
            MODEL_LABELS[model],
            (row["parameters"] / 1_000_000, row["mean_target_accuracy"]),
            xytext=label_offsets[model],
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xlabel("Parameters (millions)")
    axis.set_ylabel("Mean target accuracy")
    axis.set_ylim(0.80, 0.93)
    axis.set_xlim(1.75, 4.10)
    axis.set_title("Accuracy-parameter relation in GhostNet ablation", fontsize=13, pad=10)
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.85)
    fig.tight_layout()
    save_figure(fig, "ghostnet_ablation_parameter_tradeoff.jpeg")


def main() -> None:
    data = load_results()
    plot_accuracy_errorbars(data)
    plot_parameter_tradeoff(data)


if __name__ == "__main__":
    main()
