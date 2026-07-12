from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "revision" / "experiments" / "results" / "full"
SUMMARY_CSV = ROOT / "revision" / "experiments" / "results" / "checkpoint_eval_summary.csv"
SUMMARY_MD = ROOT / "revision" / "experiments" / "results" / "checkpoint_eval_summary.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def collect() -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for path in sorted(RESULT_DIR.glob("checkpoint_eval_*.csv")):
        if path.name == "checkpoint_eval_summary.csv":
            continue
        for row in read_rows(path):
            key = (row["target_name"], row["model"])
            grouped.setdefault(key, []).append(row)

    output = []
    for (target_name, model), rows in sorted(grouped.items()):
        accuracies = [float(row["accuracy"]) for row in rows]
        parameters = int(rows[0]["parameters"])
        samples = int(rows[0]["samples"])
        output.append(
            {
                "target_name": target_name,
                "model": model,
                "mean_accuracy": fmt(mean(accuracies)),
                "std_accuracy": fmt(std(accuracies)),
                "n_seeds": str(len(accuracies)),
                "samples": str(samples),
                "parameters": str(parameters),
            }
        )
    return output


def write_csv(rows: list[dict[str, str]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_name",
        "model",
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
        "# Checkpoint Evaluation Summary",
        "",
        "These results evaluate existing MNIST-trained checkpoints on ImageFolder and torchvision target domains.",
        "",
        "| Target | Model | Mean Accuracy | Std | Seeds | Samples | Parameters |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_name']} | {row['model']} | {row['mean_accuracy']} | "
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
