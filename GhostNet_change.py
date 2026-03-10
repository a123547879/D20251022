import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class GhostNet(nn.Module):
    def __init__(self, in_ch= 3, num_classes=10, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        cfgs = [
            # k, t, c, SE, s 
            # stage1
            [[48, 0, 1],
             [48, 0, 1]],
            # stage2
            [[48*2, 0, 2],
             [48*2, 0, 1]],
            # stage3
            [[48*4, 0.25, 2],
             [48*4, 0.25, 1]],
            # stage4
            [[48*8, 0, 2],
             [48*8, 0, 1],
             [48*8, 0, 1]],
            # stage5
            [[48*16, 0, 2],
             [48*16, 0.25, 1],
             [48*16, 0, 1]]
        ]
        # cfgs = [
        #     # k, t, c, SE, s 
        #     # stage1
        #     [[16, 0, 1]],
        #     # stage2
        #     [[48, 0, 2]],
        #     [[72, 0, 1]],
        #     # stage3
        #     [[72, 0.25, 2]],
        #     [[120, 0.25, 1]],
        #     # stage4
        #     [[240, 0, 2]],
        #     [[200, 0, 1],
        #     [184, 0, 1],
        #     [184, 0, 1],
        #     [480, 0.25, 1],
        #     [672, 0.25, 1]
        #     ],
        #     # stage5
        #     [[672, 0.25, 2]],
        #     [[960, 0, 1],
        #     [960, 0.25, 1],
        #     [960, 0, 1],
        #     [960, 0.25, 1]
        #     ]
        # ]


        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(48 * width, 4)
        self.conv_stem = nn.Conv2d(in_ch, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostModule
        for cfg in self.cfgs:
            layers = []
            for c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                # hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, output_channel, stride= s))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        # output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        
        
        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化（Conv2d用Kaiming正态分布，BN用默认初始化）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    deivce = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GhostNet().to(deivce)
    x = torch.randn(1, 3, 224, 224).to(deivce)
    summary(net, x.shape[1:])

