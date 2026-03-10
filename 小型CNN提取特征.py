import torch
import torch.nn as nn

class HeatmapEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            # 第一层：下采样+特征提取
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # 第二层：下采样+特征融合
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 全局池化
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # 输出低维特征
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        # x: [B, H, W] → 转为 [B, 1, H, W]（单通道输入）
        x = x.unsqueeze(1)
        return self.encoder(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用示例
encoder = HeatmapEncoder(output_dim=128).to(device)
encoder.eval()  # 无训练时用eval模式，避免BatchNorm波动

# # 提取低维特征
# heatmaps_lowdim = encoder(heatmaps)  # [B, 128]
# masks_lowdim = encoder(masks)        # [B, 128]（掩码图也通过同一 encoder 提取特征，保证特征空间一致）