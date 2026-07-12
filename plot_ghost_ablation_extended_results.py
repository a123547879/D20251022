from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUDOKU_CSV = ROOT / "revision" / "experiments" / "results" / "ghost_ablation" / "ghost_ablation_summary.csv"
USPS_CSV = ROOT / "revision" / "experiments" / "results" / "ghost_ablation_usps" / "ghost_ablation_usps_summary.csv"
COMPLEXITY_CSV = ROOT / "revision" / "experiments" / "results" / "model_complexity.csv"
OUTPUT_DIR = ROOT / "revision" / "experiments" / "results" / "ghost_ablation_combined"
IMAGE_DIR = ROOT / "Upload" / "images"

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


def load_combined() -> pd.DataFrame:
    sudoku = pd.read_csv(SUDOKU_CSV)
    usps = pd.read_csv(USPS_CSV)
    complexity = pd.read_csv(COMPLEXITY_CSV)

    sudoku = sudoku.rename(
        columns={
            "mean_target_accuracy": "sudoku_mean_accuracy",
            "std_target_accuracy": "sudoku_std_accuracy",
        }
    )
    usps = usps.rename(
        columns={
            "mean_accuracy": "usps_mean_accuracy",
            "std_accuracy": "usps_std_accuracy",
        }
    )
    complexity = complexity.rename(columns={"parameters": "complexity_parameters"})

    combined = sudoku[
        [
            "model",
            "se",
            "shortcut",
            "bottleneck",
            "direct_stack",
            "sudoku_mean_accuracy",
            "sudoku_std_accuracy",
            "n_seeds",
            "parameters",
        ]
    ].merge(
        usps[["model", "usps_mean_accuracy", "usps_std_accuracy", "samples"]],
        on="model",
        how="left",
    )
    combined = combined.merge(
        complexity[["model", "macs_m", "flops_m"]],
        on="model",
        how="left",
    )
    combined["model"] = pd.Categorical(combined["model"], MODEL_ORDER, ordered=True)
    combined = combined.sort_values("model")
    combined["label"] = combined["model"].astype(str).map(MODEL_LABELS)
    return combined


def save_tables(data: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / "ghost_ablation_target_complexity_summary.csv"
    output_md = OUTPUT_DIR / "ghost_ablation_target_complexity_summary.md"
    data.to_csv(output_csv, index=False)

    lines = [
        "# GhostNet Ablation Target and Complexity Summary",
        "",
        "| Variant | SE | Shortcut | Bottleneck | Sudoku Acc. | USPS Acc. | Params | MACs (M) | FLOPs (M) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in data.iterrows():
        lines.append(
            f"| {row['label']} | {row['se']} | {row['shortcut']} | {row['bottleneck']} | "
            f"{float(row['sudoku_mean_accuracy']):.4f} +/- {float(row['sudoku_std_accuracy']):.4f} | "
            f"{float(row['usps_mean_accuracy']):.4f} +/- {float(row['usps_std_accuracy']):.4f} | "
            f"{int(row['parameters'])} | {float(row['macs_m']):.2f} | {float(row['flops_m']):.2f} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")


def save_figure(fig: plt.Figure, filename: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output = IMAGE_DIR / filename
    fig.savefig(output, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"Wrote {output}")


def plot_target_comparison(data: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.6))
    x = list(range(len(data)))
    width = 0.36
    colors = [MODEL_COLORS[str(model)] for model in data["model"]]
    light_colors = [f"{color}99" for color in colors]

    axis.bar(
        [value - width / 2 for value in x],
        data["sudoku_mean_accuracy"],
        yerr=data["sudoku_std_accuracy"],
        width=width,
        color=colors,
        edgecolor="#222222",
        linewidth=0.7,
        capsize=3,
        label="Sudoku-Digits",
    )
    axis.bar(
        [value + width / 2 for value in x],
        data["usps_mean_accuracy"],
        yerr=data["usps_std_accuracy"],
        width=width,
        color=light_colors,
        edgecolor="#222222",
        linewidth=0.7,
        capsize=3,
        label="USPS test",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(data["label"], rotation=24, ha="right")
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Mean target accuracy")
    axis.set_title("GhostNet ablation across two target domains", fontsize=13, pad=10)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.85)
    axis.set_axisbelow(True)
    axis.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, "ghostnet_ablation_sudoku_usps_comparison.jpeg")


def plot_accuracy_macs_tradeoff(data: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8.8, 5.6))
    for _, row in data.iterrows():
        model = str(row["model"])
        axis.scatter(
            row["macs_m"],
            row["sudoku_mean_accuracy"],
            s=125,
            marker="o",
            color=MODEL_COLORS[model],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.92,
        )
        axis.scatter(
            row["macs_m"],
            row["usps_mean_accuracy"],
            s=125,
            marker="^",
            color=MODEL_COLORS[model],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.72,
        )
        axis.annotate(row["label"], (row["macs_m"], row["sudoku_mean_accuracy"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    axis.set_xlabel("MACs (millions)")
    axis.set_ylabel("Mean target accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("GhostNet ablation accuracy-complexity relation", fontsize=13, pad=10)
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.85)
    axis.scatter([], [], marker="o", color="#777777", edgecolor="#222222", label="Sudoku-Digits")
    axis.scatter([], [], marker="^", color="#777777", edgecolor="#222222", label="USPS test")
    axis.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "ghostnet_ablation_accuracy_macs_tradeoff.jpeg")


def main() -> None:
    data = load_combined()
    save_tables(data)
    plot_target_comparison(data)
    plot_accuracy_macs_tradeoff(data)


if __name__ == "__main__":
    main()
