from __future__ import annotations

import argparse
import csv
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "D20251116"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

if "torchsummary" not in sys.modules:
    torchsummary_stub = types.ModuleType("torchsummary")
    torchsummary_stub.summary = lambda *args, **kwargs: None
    sys.modules["torchsummary"] = torchsummary_stub

from cnn_model2 import PrecisionBalancedCNN  # noqa: E402
from GhostNet import GhostNet as OriginalGhostNet  # noqa: E402
from GhostNet_change import GhostNet as ChangedGhostNet  # noqa: E402
from ghost_simple import SimpleGhost  # noqa: E402
from ghostnet_ablation_models import (  # noqa: E402
    ghostnet_no_se,
    ghostnet_no_se_no_shortcut,
    ghostnet_no_shortcut,
)
from MobileNetV2 import MobileNetV2  # noqa: E402
from MobileNetV3 import MobileNetV3  # noqa: E402
from ResNet18 import resnet18  # noqa: E402
from ShuffleNetV2 import ShuffleNetV2  # noqa: E402


BALANCED_MODEL_TYPES = {
    "basic",
    "ghost",
    "inverted_residual",
    "residual",
    "shuffle",
    "selayer",
}


@dataclass(frozen=True)
class RunResult:
    seed: int
    model: str
    source: str
    target: str
    target_split: str
    epochs: int
    batch_size: int
    learning_rate: float
    in_channels: int
    parameters: int
    train_accuracy: float
    target_accuracy: float
    checkpoint: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_transform(in_channels: int) -> transforms.Compose:
    if in_channels not in {1, 3}:
        raise ValueError("--in-channels must be 1 or 3")

    mean = [0.1307] * in_channels
    std = [0.3081] * in_channels
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=in_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_dataset(
    name: str,
    root: Path,
    split: str,
    transform: transforms.Compose,
    download: bool,
    imagefolder_path: Path | None = None,
):
    name = name.lower()
    if name == "mnist":
        return datasets.MNIST(
            root=str(root),
            train=split == "train",
            transform=transform,
            download=download,
        )
    if name == "usps":
        return datasets.USPS(
            root=str(root),
            train=split == "train",
            transform=transform,
            download=download,
        )
    if name == "svhn":
        return datasets.SVHN(
            root=str(root),
            split=split,
            transform=transform,
            download=download,
        )
    if name in {"sudoku", "imagefolder"}:
        if imagefolder_path is None:
            raise ValueError("--target-path is required for imagefolder targets")
        return datasets.ImageFolder(root=str(imagefolder_path), transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def maybe_limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("Dataset limits must be positive")
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_model(model_name: str, in_channels: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name in BALANCED_MODEL_TYPES:
        return PrecisionBalancedCNN(model_name, in_channels=in_channels)
    if model_name == "mobilenetv2":
        return MobileNetV2(in_ch=in_channels, n_classes=10)
    if model_name == "mobilenetv3":
        return MobileNetV3(in_ch=in_channels, n_classes=10)
    if model_name == "resnet18":
        return resnet18(in_ch=in_channels, n_classes=10)
    if model_name == "shufflenetv2":
        return ShuffleNetV2(in_ch=in_channels, n_classes=10)
    if model_name == "ghostnet_original":
        return OriginalGhostNet(in_ch=in_channels, num_classes=10)
    if model_name == "ghostnet_no_se":
        return ghostnet_no_se(in_ch=in_channels, num_classes=10)
    if model_name == "ghostnet_no_shortcut":
        return ghostnet_no_shortcut(in_ch=in_channels, num_classes=10)
    if model_name == "ghostnet_no_se_no_shortcut":
        return ghostnet_no_se_no_shortcut(in_ch=in_channels, num_classes=10)
    if model_name == "ghostnet_change":
        return ChangedGhostNet(in_ch=in_channels, num_classes=10)
    if model_name == "ghostnet_simple":
        return SimpleGhost(in_channels=in_channels, num_classes=10)
    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total if total else 0.0


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


def append_result(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunResult.__dataclass_fields__.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(result.__dict__)


def run_for_seed(args: argparse.Namespace, seed: int) -> RunResult:
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.in_channels).to(device)
    parameters = count_parameters(model)

    if args.dry_run:
        return RunResult(
            seed=seed,
            model=args.model,
            source=args.source,
            target=args.target,
            target_split=args.target_split,
            epochs=0,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            in_channels=args.in_channels,
            parameters=parameters,
            train_accuracy=0.0,
            target_accuracy=0.0,
            checkpoint="",
        )

    transform = build_transform(args.in_channels)
    train_dataset = build_dataset(
        name=args.source,
        root=args.data_root,
        split="train",
        transform=transform,
        download=args.download,
    )
    target_dataset = build_dataset(
        name=args.target,
        root=args.data_root,
        split=args.target_split,
        transform=transform,
        download=args.download,
        imagefolder_path=args.target_path,
    )
    train_dataset = maybe_limit_dataset(train_dataset, args.limit_train)
    target_dataset = maybe_limit_dataset(target_dataset, args.limit_eval)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_accuracy = 0.0
    for _ in range(args.epochs):
        train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)

    target_accuracy = evaluate(model, target_loader, device)

    checkpoint_path = ""
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = (
            args.checkpoint_dir
            / f"{args.model}_{args.source}_to_{args.target}_c{args.in_channels}_seed{seed}.pth"
        )
        torch.save(model.state_dict(), checkpoint)
        checkpoint_path = str(checkpoint)

    return RunResult(
        seed=seed,
        model=args.model,
        source=args.source,
        target=args.target,
        target_split=args.target_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        in_channels=args.in_channels,
        parameters=parameters,
        train_accuracy=train_accuracy,
        target_accuracy=target_accuracy,
        checkpoint=checkpoint_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on a source digit dataset and evaluate on a shifted target domain."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="mnist", choices=["mnist"])
    parser.add_argument("--target", required=True, choices=["mnist", "usps", "svhn", "sudoku", "imagefolder"])
    parser.add_argument("--target-split", default="test", choices=["train", "test"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "D20251116" / "dataFolder")
    parser.add_argument("--target-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "revision" / "experiments" / "digit_shift_results.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2025, 1024, 512, 100])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        result = run_for_seed(args, seed)
        if not args.dry_run:
            append_result(args.output, result)
        print(
            f"seed={result.seed} model={result.model} target={result.target} "
            f"train_acc={result.train_accuracy:.4f} target_acc={result.target_accuracy:.4f} "
            f"params={result.parameters}"
        )


if __name__ == "__main__":
    main()
