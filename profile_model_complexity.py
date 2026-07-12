from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from digit_shift_pipeline import build_model, count_parameters
from MobileNetV3 import MobileNetV3


ROOT = Path(__file__).resolve().parents[2]

MODEL_ORDER = [
    "mobilenetv2",
    "shufflenetv2",
    "resnet18",
    "mobilenetv3",
    "ghostnet_original",
    "ghostnet_no_se",
    "ghostnet_no_shortcut",
    "ghostnet_no_se_no_shortcut",
    "ghostnet_change",
    "ghostnet_simple",
]

MODEL_LABELS = {
    "mobilenetv2": "MobileNetV2",
    "shufflenetv2": "ShuffleNetV2",
    "resnet18": "ResNet18",
    "mobilenetv3": "MobileNetV3",
    "ghostnet_original": "GhostNet original",
    "ghostnet_no_se": "GhostNet no SE",
    "ghostnet_no_shortcut": "GhostNet no shortcut",
    "ghostnet_no_se_no_shortcut": "GhostNet no SE/shortcut",
    "ghostnet_change": "GhostNet modified",
    "ghostnet_simple": "GhostNet direct stack",
}


@dataclass(frozen=True)
class ComplexityResult:
    model: str
    label: str
    input_channels: int
    input_size: int
    parameters: int
    macs: int
    flops: int
    macs_m: float
    flops_m: float


def conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    output_elements = output.numel()
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
    return int(output_elements * kernel_ops)


def linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features)


def batchnorm_ops(output: torch.Tensor) -> int:
    return int(output.numel() * 2)


def elementwise_ops(output: torch.Tensor) -> int:
    return int(output.numel())


def pool_ops(module: nn.Module, output: torch.Tensor) -> int:
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return int(output.numel())
    kernel_size = getattr(module, "kernel_size", 1)
    if isinstance(kernel_size, tuple):
        kernel_ops = kernel_size[0] * kernel_size[1]
    else:
        kernel_ops = int(kernel_size) * int(kernel_size)
    return int(output.numel() * kernel_ops)


def profile_model(model: nn.Module, input_shape: tuple[int, int, int]) -> tuple[int, int]:
    operation_counts: list[int] = []
    hooks = []

    def register(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(lambda mod, _inp, out: operation_counts.append(conv2d_macs(mod, out))))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(lambda mod, _inp, out: operation_counts.append(linear_macs(mod, out))))
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(lambda _mod, _inp, out: operation_counts.append(batchnorm_ops(out))))
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.Hardsigmoid, nn.Sigmoid)):
            hooks.append(module.register_forward_hook(lambda _mod, _inp, out: operation_counts.append(elementwise_ops(out))))
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            hooks.append(module.register_forward_hook(lambda mod, _inp, out: operation_counts.append(pool_ops(mod, out))))

    model.apply(register)
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.zeros((1, *input_shape), device=device)
    with torch.no_grad():
        model(dummy_input)
    for hook in hooks:
        hook.remove()

    macs = sum(operation_counts)
    flops = macs * 2
    return macs, flops


def write_csv(path: Path, rows: list[ComplexityResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ComplexityResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(path: Path, rows: list[ComplexityResult]) -> None:
    lines = [
        "# Model Complexity Summary",
        "",
        "MACs are counted for one forward pass with input shape `1 x 224 x 224`. FLOPs are reported as `2 x MACs`.",
        "",
        "| Model | Params | MACs (M) | FLOPs (M) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row.label} | {row.parameters} | {row.macs_m:.2f} | {row.flops_m:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile model parameters, MACs, and FLOPs.")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=ROOT / "revision" / "experiments" / "results" / "model_complexity.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "revision" / "experiments" / "results" / "model_complexity.md")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows: list[ComplexityResult] = []
    for model_name in args.models:
        if model_name == "mobilenetv3":
            model = MobileNetV3(version="small", in_ch=args.in_channels, n_classes=10).to(device)
        else:
            model = build_model(model_name, args.in_channels).to(device)
        macs, flops = profile_model(model, (args.in_channels, args.input_size, args.input_size))
        row = ComplexityResult(
            model=model_name,
            label=MODEL_LABELS.get(model_name, model_name),
            input_channels=args.in_channels,
            input_size=args.input_size,
            parameters=count_parameters(model),
            macs=macs,
            flops=flops,
            macs_m=macs / 1_000_000,
            flops_m=flops / 1_000_000,
        )
        rows.append(row)
        print(f"model={row.model} params={row.parameters} macs_m={row.macs_m:.2f} flops_m={row.flops_m:.2f}")

    write_csv(args.output, rows)
    write_markdown(args.markdown, rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
