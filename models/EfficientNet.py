import torch
from torch import nn

class MBConvBlock(nn.Module):
    """EfficientNet 的 MBConv 块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expansion_ratio, se_ratio=0.25, dropout_rate=0.2):
        super().__init__()
        expanded_channels = int(in_channels * expansion_ratio)
        
        # 扩展阶段
        self.expand_conv = None
        if expansion_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        
        # 深度卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # SE 注意力
        self.se_block = SEModule(expanded_channels, se_ratio)
        
        # 投影层
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        residual = x
        
        if self.expand_conv:
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.se_block(x)
        x = self.project_conv(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        if self.use_residual:
            x += residual
        
        return x

class SEModule(nn.Module):
    """压缩和激励模块"""
    def __init__(self, channels, reduction_ratio=0.25):
        super().__init__()
        reduced_channels = max(1, int(channels * reduction_ratio))
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)