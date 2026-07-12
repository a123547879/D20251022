from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "revision" / "experiments" / "results" / "ghost_ablation_usps"
INPUT_CSV = RESULT_DIR / "ghost_ablation_usps_test.csv"
SUMMARY_CSV = RESULT_DIR / "ghost_ablation_usps_summary.csv"
SUMMARY_MD = RESULT_DIR / "ghost_ablation_usps_summary.md"

VARIANT_FACTORS = {
    "ghostnet_original": ("yes", "yes", "yes", "no"),
    "ghostnet_no_se": ("no", "yes", "yes", "no"),
    "ghostnet_no_shortcut": ("yes", "no", "yes", "no"),
    "ghostnet_no_se_no_shortcut": ("no", "no", "yes", "no"),
    "ghostnet_change": ("no", "no", "partial", "no"),
    "ghostnet_simple": ("no", "no", "no", "yes"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def collect() -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_rows(INPUT_CSV):
        grouped.setdefault(row["model"], []).append(row)

    rows = []
    for model, model_rows in sorted(grouped.items()):
        accuracies = [float(row["accuracy"]) for row in model_rows]
        se, shortcut, bottleneck, direct_stack = VARIANT_FACTORS.get(
            model,
            ("unknown", "unknown", "unknown", "unknown"),
        )
        rows.append(
            {
                "target_name": model_rows[0]["target_name"],
                "model": model,
                "se": se,
                "shortcut": shortcut,
                "bottleneck": bottleneck,
                "direct_stack": direct_stack,
                "mean_accuracy": fmt(mean(accuracies)),
                "std_accuracy": fmt(std(accuracies)),
                "n_seeds": str(len(model_rows)),
                "samples": model_rows[0]["samples"],
                "parameters": model_rows[0]["parameters"],
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "target_name",
        "model",
        "se",
        "shortcut",
        "bottleneck",
        "direct_stack",
        "mean_accuracy",
        "std_accuracy",
        "n_seeds",
        "samples",
        "parameters",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]]) -> None:
    lines = [
        "# GhostNet USPS Ablation Summary",
        "",
        "| Target | Model | SE | Shortcut | Bottleneck | Direct Stack | Mean Acc. | Std | Seeds | Samples | Params |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_name']} | {row['model']} | {row['se']} | {row['shortcut']} | "
            f"{row['bottleneck']} | {row['direct_stack']} | {row['mean_accuracy']} | "
            f"{row['std_accuracy']} | {row['n_seeds']} | {row['samples']} | {row['parameters']} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect()
    write_csv(rows)
    write_markdown(rows)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
