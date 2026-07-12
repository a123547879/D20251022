from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from digit_shift_pipeline import build_model, build_transform, count_parameters


ROOT = Path(__file__).resolve().parents[2]

MODEL_ORDER = [
    "ghostnet_original",
    "ghostnet_no_se",
    "ghostnet_no_shortcut",
    "ghostnet_no_se_no_shortcut",
    "ghostnet_change",
    "ghostnet_simple",
]


@dataclass(frozen=True)
class EvaluationResult:
    target_name: str
    model: str
    seed: int
    in_channels: int
    parameters: int
    samples: int
    accuracy: float
    checkpoint: str


def checkpoint_path(checkpoint_dir: Path, model: str, seed: int) -> Path:
    return checkpoint_dir / f"{model}_mnist_to_imagefolder_c1_seed{seed}.pth"


def build_usps_dataset(args: argparse.Namespace):
    transform = build_transform(args.in_channels)
    return datasets.USPS(
        root=str(args.data_root),
        train=args.target_split == "train",
        transform=transform,
        download=args.download,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total if total else 0.0


def write_results(path: Path, rows: list[EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvaluationResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GhostNet ablation checkpoints on USPS.")
    parser.add_argument("--target-name", default="usps-test")
    parser.add_argument("--target-split", choices=["train", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "revision" / "experiments" / "torchvision_data")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "revision" / "experiments" / "checkpoints" / "ghost_ablation")
    parser.add_argument("--output", type=Path, default=ROOT / "revision" / "experiments" / "results" / "ghost_ablation_usps" / "ghost_ablation_usps_test.csv")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2025, 1024])
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_usps_dataset(args)
    if args.limit_eval is not None:
        dataset = Subset(dataset, list(range(min(args.limit_eval, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows: list[EvaluationResult] = []
    for model_name in args.models:
        for seed in args.seeds:
            checkpoint = checkpoint_path(args.checkpoint_dir, model_name, seed)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = build_model(model_name, args.in_channels)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model = model.to(device)
            accuracy = evaluate(model, loader, device)
            row = EvaluationResult(
                target_name=args.target_name,
                model=model_name,
                seed=seed,
                in_channels=args.in_channels,
                parameters=count_parameters(model),
                samples=len(dataset),
                accuracy=accuracy,
                checkpoint=str(checkpoint),
            )
            rows.append(row)
            print(
                f"target={args.target_name} model={model_name} seed={seed} "
                f"accuracy={accuracy:.4f} samples={len(dataset)}"
            )

    write_results(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
