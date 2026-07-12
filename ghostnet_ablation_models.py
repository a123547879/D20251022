from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25, divisor: int = 4):
        super().__init__()
        reduced_channels = make_divisible(channels * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(channels, reduced_channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.conv_reduce(scale)
        scale = self.act(scale)
        scale = self.conv_expand(scale)
        return x * hard_sigmoid(scale)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        relu: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        output = torch.cat([primary, cheap], dim=1)
        return output[:, : self.out_channels, :, :]


class GhostBottleneckAblation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
        enable_se: bool = True,
        enable_shortcut: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.enable_shortcut = enable_shortcut
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)
        if stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_channels,
                mid_channels,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_channels,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_channels)
        else:
            self.conv_dw = None
            self.bn_dw = None
        if enable_se and se_ratio is not None and se_ratio > 0.0:
            self.se = SqueezeExcite(mid_channels, se_ratio=se_ratio)
        else:
            self.se = nn.Identity()
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ghost1(x)
        if self.conv_dw is not None and self.bn_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        x = self.se(x)
        x = self.ghost2(x)
        if self.enable_shortcut:
            x = x + self.shortcut(residual)
        return x


class GhostNetAblation(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 10,
        width: float = 1.0,
        dropout: float = 0.2,
        enable_se: bool = True,
        enable_shortcut: bool = True,
    ):
        super().__init__()
        cfgs = [
            [[3, 16, 16, 0, 1]],
            [[3, 48, 24, 0, 2]],
            [[3, 72, 24, 0, 1]],
            [[5, 72, 40, 0.25, 2]],
            [[5, 120, 40, 0.25, 1]],
            [
                [3, 240, 80, 0, 2],
                [3, 200, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1],
            ],
            [[5, 672, 160, 0.25, 2]],
            [
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
            ],
        ]
        output_channel = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_ch, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel
        stages = []
        for cfg in cfgs:
            layers = []
            for kernel_size, exp_size, channels, se_ratio, stride in cfg:
                output_channel = make_divisible(channels * width, 4)
                hidden_channel = make_divisible(exp_size * width, 4)
                layers.append(
                    GhostBottleneckAblation(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        kernel_size,
                        stride,
                        se_ratio=se_ratio,
                        enable_se=enable_se,
                        enable_shortcut=enable_shortcut,
                    )
                )
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        output_channel = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        self.blocks = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, 1280, 1, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(1280, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def ghostnet_no_se(in_ch: int = 1, num_classes: int = 10) -> GhostNetAblation:
    return GhostNetAblation(in_ch=in_ch, num_classes=num_classes, enable_se=False, enable_shortcut=True)


def ghostnet_no_shortcut(in_ch: int = 1, num_classes: int = 10) -> GhostNetAblation:
    return GhostNetAblation(in_ch=in_ch, num_classes=num_classes, enable_se=True, enable_shortcut=False)


def ghostnet_no_se_no_shortcut(in_ch: int = 1, num_classes: int = 10) -> GhostNetAblation:
    return GhostNetAblation(in_ch=in_ch, num_classes=num_classes, enable_se=False, enable_shortcut=False)
