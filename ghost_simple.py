import torch
from torch import nn
from torchsummary import summary

class SimpleGhost(nn.Module):
    def __init__(self, in_channels= 1, num_classes= 10):
        super(SimpleGhost, self).__init__()

        base_channels = 48

        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 2, 1, bias= False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace= True)
        )

        self.stages = nn.ModuleList()
        current_channels = base_channels

        stages_config = [
            (current_channels * 1, 1, 2),
            (current_channels * 2, 2, 2),
            (current_channels * 4, 2, 2),
            (current_channels * 8, 2, 3),
            (current_channels * 16, 2, 3)
            # (1280, 1, 1)
        ]

        for i, (out_channels, stride, n_blocks) in enumerate(stages_config):
            # current_channels = current_channels if current_channels % 2 == 0 else current_channels + 1
            out_channels = out_channels if  out_channels % 2 == 0 else out_channels + 1

            stage = []
            stage.append(GhostBlock(current_channels, out_channels, stride= stride))

            for i in range(1, n_blocks):
                stage.append(GhostBlock(out_channels, out_channels, stride= 1))

            self.stages.append(nn.Sequential(*stage))
            current_channels = out_channels

        self.last_conv = nn.Sequential(
            nn.Conv2d(current_channels, 1280, 1, 1, 0, bias= False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace= True)
        )

        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(1280, 10)
        # )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(1280, 10)

    def forward(self, x):
        out = self.stem_conv(x)
        for stage in self.stages:
            out = stage(out)
        out = self.global_pool(out)
        out = self.last_conv(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out
    
# class GhostBottleneck(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, stride= 1):
#         super(GhostBottleneck, self).__init__()
#         self.ghost1 = GhostBlock(in_channels, out_channels, stride= stride)
#         # self.ghost2 = GhostBlock(mid_channels, out_channels, stride= stride)
#     def forward(self, x):
#         x = self.ghost1(x)
#         # x = self.ghost2(x)
#         return x
    
class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride= 1):
        super(GhostBlock, self).__init__()

        in_channels = max(2, in_channels)
        out_channels = max(2 ,out_channels)
        
        if out_channels % 2 != 0:
            out_channels += 1

        primary_out = out_channels // 2

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, primary_out, 1, stride, 0, bias= False),
            nn.BatchNorm2d(primary_out),
            nn.ReLU(inplace= True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(primary_out, primary_out, 3, 1, 1, groups= primary_out, bias= False),
            nn.BatchNorm2d(primary_out),
            nn.ReLU(inplace= True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim= 1)
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ghost = SimpleGhost(1, 10).to(device)
    x = torch.randn([10, 1, 224, 224])
    summary(ghost, (1, 224, 224))