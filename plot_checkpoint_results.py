from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "revision" / "experiments" / "results" / "checkpoint_eval_summary.csv"
OUTPUT_DIR = ROOT / "Upload" / "images"

TARGET_ORDER = ["sudoku-digits", "generated-num", "usps-test", "svhn-train"]
MODEL_ORDER = ["mobilenetv2", "shufflenetv2", "resnet18", "mobilenetv3", "ghostnet"]
TARGET_LABELS = {
    "sudoku-digits": "Sudoku-Digits",
    "generated-num": "Generated-Num",
    "usps-test": "USPS test",
    "svhn-train": "SVHN train",
}
MODEL_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "shufflenetv2": "ShuffleNetV2",
    "resnet18": "ResNet18",
    "mobilenetv3": "MobileNetV3",
    "ghostnet": "GhostNet",
}
MODEL_COLORS = {
    "mobilenetv2": "#2f6f9f",
    "shufflenetv2": "#4f8f62",
    "resnet18": "#8b6f47",
    "mobilenetv3": "#7b5ea7",
    "ghostnet": "#b75d4a",
}
TARGET_MARKERS = {
    "sudoku-digits": "o",
    "generated-num": "s",
    "usps-test": "^",
    "svhn-train": "D",
}


def load_results() -> pd.DataFrame:
    data = pd.read_csv(SUMMARY_CSV)
    required = {
        "target_name",
        "model",
        "mean_accuracy",
        "std_accuracy",
        "samples",
        "parameters",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {SUMMARY_CSV}: {sorted(missing)}")

    data = data.copy()
    data["target_name"] = pd.Categorical(data["target_name"], TARGET_ORDER, ordered=True)
    data["model"] = pd.Categorical(data["model"], MODEL_ORDER, ordered=True)
    data = data.sort_values(["target_name", "model"])
    return data


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / filename
    fig.savefig(output, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Wrote {output}")


def plot_accuracy_errorbars(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for axis, target in zip(axes, TARGET_ORDER):
        subset = data[data["target_name"] == target]
        x = range(len(subset))
        colors = [MODEL_COLORS[str(model)] for model in subset["model"]]
        axis.bar(
            x,
            subset["mean_accuracy"],
            yerr=subset["std_accuracy"],
            color=colors,
            edgecolor="#222222",
            linewidth=0.8,
            capsize=4,
        )
        axis.set_title(TARGET_LABELS[target], fontsize=12, pad=8)
        axis.set_xticks(list(x))
        axis.set_xticklabels(
            [MODEL_LABELS[str(model)] for model in subset["model"]],
            rotation=25,
            ha="right",
            fontsize=9,
        )
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        for index, (_, row) in enumerate(subset.iterrows()):
            label_y = min(row["mean_accuracy"] + row["std_accuracy"] + 0.018, 0.985)
            axis.text(index, label_y, f"{row['mean_accuracy']:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Mean accuracy")
    axes[2].set_ylabel("Mean accuracy")
    fig.suptitle("Target-domain accuracy of MNIST-trained checkpoints", fontsize=14, y=0.995)
    fig.tight_layout()
    save_figure(fig, "multi_target_accuracy_errorbars.jpeg")


def plot_accuracy_heatmap(data: pd.DataFrame) -> None:
    matrix = data.pivot(index="target_name", columns="model", values="mean_accuracy")
    matrix = matrix.loc[TARGET_ORDER, MODEL_ORDER]

    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    image = axis.imshow(matrix.values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(MODEL_ORDER)))
    axis.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=25, ha="right")
    axis.set_yticks(range(len(TARGET_ORDER)))
    axis.set_yticklabels([TARGET_LABELS[target] for target in TARGET_ORDER])
    axis.set_title("Mean accuracy matrix across target domains", fontsize=13, pad=10)

    for row_index, target in enumerate(TARGET_ORDER):
        for col_index, model in enumerate(MODEL_ORDER):
            value = matrix.loc[target, model]
            text_color = "white" if value > 0.58 else "#1f1f1f"
            axis.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)

    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean accuracy", rotation=90)
    fig.tight_layout()
    save_figure(fig, "target_model_accuracy_heatmap.jpeg")


def plot_parameter_tradeoff(data: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.6))

    for target in TARGET_ORDER:
        subset = data[data["target_name"] == target]
        for _, row in subset.iterrows():
            model = str(row["model"])
            axis.scatter(
                row["parameters"] / 1_000_000,
                row["mean_accuracy"],
                s=105,
                marker=TARGET_MARKERS[target],
                color=MODEL_COLORS[model],
                edgecolor="#222222",
                linewidth=0.8,
                alpha=0.9,
            )

    axis.set_xscale("log")
    axis.set_xlabel("Parameters (millions, log scale)")
    axis.set_ylabel("Mean accuracy")
    axis.set_ylim(0, 1.02)
    axis.set_title("Parameter-accuracy trade-off under target-domain shifts", fontsize=13, pad=10)
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    target_handles = [
        Line2D(
            [0],
            [0],
            marker=TARGET_MARKERS[target],
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="#222222",
            markersize=8,
            label=TARGET_LABELS[target],
        )
        for target in TARGET_ORDER
    ]
    model_handles = [Patch(facecolor=MODEL_COLORS[model], edgecolor="#222222", label=MODEL_LABELS[model]) for model in MODEL_ORDER]
    target_legend = axis.legend(handles=target_handles, title="Target", loc="center left", frameon=True)
    axis.add_artist(target_legend)
    axis.legend(handles=model_handles, title="Model", loc="lower left", frameon=True, ncols=2)
    fig.tight_layout()
    save_figure(fig, "parameter_accuracy_tradeoff.jpeg")


def main() -> None:
    data = load_results()
    plot_accuracy_errorbars(data)
    plot_accuracy_heatmap(data)
    plot_parameter_tradeoff(data)


if __name__ == "__main__":
    main()
