from __future__ import annotations

import argparse
import csv
import ssl
import sys
import types
from dataclasses import dataclass
from pathlib import Path

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

from GhostNet import GhostNet  # noqa: E402
from MobileNetV2 import MobileNetV2  # noqa: E402
from MobileNetV3 import MobileNetV3  # noqa: E402
from ResNet18 import resnet18  # noqa: E402
from ShuffleNetV2 import ShuffleNetV2  # noqa: E402


MODEL_BUILDERS = {
    "mobilenetv2": lambda in_channels: MobileNetV2(in_ch=in_channels, n_classes=10),
    "mobilenetv3": lambda in_channels: MobileNetV3(version="small", in_ch=in_channels, n_classes=10),
    "resnet18": lambda in_channels: resnet18(in_ch=in_channels, n_classes=10),
    "shufflenetv2": lambda in_channels: ShuffleNetV2(in_ch=in_channels, n_classes=10),
    "ghostnet": lambda in_channels: GhostNet(in_ch=in_channels, num_classes=10),
}


@dataclass(frozen=True)
class EvaluationResult:
    target_name: str
    target_path: str
    model: str
    seed: int
    in_channels: int
    parameters: int
    samples: int
    accuracy: float
    checkpoint: str


def build_transform(in_channels: int) -> transforms.Compose:
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


def build_dataset(args: argparse.Namespace):
    transform = build_transform(args.in_channels)
    target = args.target.lower()
    if target == "imagefolder":
        if args.target_path is None:
            raise ValueError("--target-path is required when --target imagefolder")
        return datasets.ImageFolder(str(args.target_path), transform=transform), str(args.target_path)
    if target == "usps":
        return (
            datasets.USPS(
                root=str(args.data_root),
                train=args.target_split == "train",
                transform=transform,
                download=args.download,
            ),
            str(args.data_root / "USPS"),
        )
    if target == "svhn":
        return (
            datasets.SVHN(
                root=str(args.data_root),
                split=args.target_split,
                transform=transform,
                download=args.download,
            ),
            str(args.data_root / "SVHN"),
        )
    raise ValueError(f"Unsupported target: {args.target}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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


def append_results(path: Path, results: list[EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvaluationResult.__dataclass_fields__.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def append_result(path: Path, result: EvaluationResult) -> None:
    append_results(path, [result])


def checkpoint_path(checkpoint_dir: Path, model: str, seed: int) -> Path:
    return checkpoint_dir / f"{model}_md_{seed}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing MNIST-trained checkpoints on target datasets.")
    parser.add_argument("--target", choices=["imagefolder", "usps", "svhn"], default="imagefolder")
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-path", type=Path)
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "revision" / "experiments" / "data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--insecure-download", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "D20251116" / "models" / "complete")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenetv2", "resnet18", "mobilenetv3", "shufflenetv2", "ghostnet"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2025, 1024, 512, 100])
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.insecure_download and args.download:
        ssl._create_default_https_context = ssl._create_unverified_context
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.overwrite and args.output.exists():
        args.output.unlink()
    dataset, target_path = build_dataset(args)
    if args.limit_eval is not None:
        dataset = Subset(dataset, list(range(min(args.limit_eval, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results: list[EvaluationResult] = []
    for model_name in args.models:
        if model_name not in MODEL_BUILDERS:
            raise ValueError(f"Unsupported model: {model_name}")
        for seed in args.seeds:
            checkpoint = checkpoint_path(args.checkpoint_dir, model_name, seed)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = MODEL_BUILDERS[model_name](args.in_channels)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model = model.to(device)
            parameters = count_parameters(model)
            accuracy = evaluate(model, loader, device)
            result = EvaluationResult(
                target_name=args.target_name,
                target_path=target_path,
                model=model_name,
                seed=seed,
                in_channels=args.in_channels,
                parameters=parameters,
                samples=len(dataset),
                accuracy=accuracy,
                checkpoint=str(checkpoint),
            )
            results.append(result)
            append_result(args.output, result)
            print(
                f"target={args.target_name} model={model_name} seed={seed} "
                f"accuracy={accuracy:.4f} samples={len(dataset)}"
            )


if __name__ == "__main__":
    main()
