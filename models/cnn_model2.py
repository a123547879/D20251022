import torch
import torch.nn as nn
from torchsummary import summary

class PrecisionBalancedCNN(nn.Module):
    def __init__(self, block_type, in_channels=1):
        """
        精确参数平衡的轻量级架构测试
        """
        super(PrecisionBalancedCNN, self).__init__()
        self.block_type = block_type

        # 基于测试结果重新调整的基础通道数
        # 目标：所有架构都接近1.6M参数
        base_channels_config = {
            'basic': 24,           # 从24增加到36
            'residual': 24,        # 从24增加到36  
            'inverted_residual': 32,  # 保持32
            'shuffle': 44,         # 从48增加到64 (大幅增加)
            'ghost': 48           # 从52增加到72 (大幅增加)
        }
        
        base_channels = base_channels_config[block_type]
        print(f"Using base_channels={base_channels} for {block_type}")

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 阶段配置
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        # 调整阶段配置以平衡参数量
        if block_type in ['shuffle', 'ghost']:
            # 对于高效架构，使用更多阶段来增加参数量
            stages_config = [
                (current_channels * 1, 1, 2),   # 阶段1
                (current_channels * 2, 2, 2),   # 阶段2
                (current_channels * 4, 2, 2),   # 阶段3  
                (current_channels * 8, 2, 3),   # 阶段4: 增加块数
                (current_channels * 16, 2, 3)  # 阶段5: 新增阶段
            ]
        else:
            # 对于低效架构，保持原有阶段配置
            stages_config = [
                (current_channels * 1, 2, 2),   # 阶段1
                (current_channels * 2, 2, 2),   # 阶段2
                (current_channels * 4, 2, 2),   # 阶段3
                (current_channels * 8, 2, 2),   # 阶段4
            ]
        
        for i, (out_channels, stride, num_blocks_in_stage) in enumerate(stages_config):
            if block_type in ['shuffle', 'ghost']:
                in_channels = in_channels if in_channels % 2 == 0 else in_channels + 1
                out_channels = out_channels if out_channels % 2 == 0 else out_channels + 1
                
            stage = self._make_stage(
                block_type, 
                current_channels, 
                out_channels, 
                stride, 
                num_blocks_in_stage
            )
            self.stages.append(stage)
            current_channels = out_channels

        # 最后的卷积和分类头
        self.last_conv = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, 10)
        )

        self._initialize_weights()
    
    def _make_stage(self, block_type, in_channels, out_channels, stride, num_blocks):
        blocks = []
        
        if block_type in ['shuffle', 'ghost']:
            in_channels = max(2, in_channels)
            out_channels = max(2, out_channels)
            in_channels = in_channels if in_channels % 2 == 0 else in_channels + 1
            out_channels = out_channels if out_channels % 2 == 0 else out_channels + 1
        
        # 第一个块
        if block_type == 'basic':
            blocks.append(BasicBlock(in_channels, out_channels, stride))
        elif block_type == 'inverted_residual':
            blocks.append(InvertedResidual(in_channels, out_channels, stride, expansion_ratio=6))
        elif block_type == 'shuffle':
            blocks.append(ShuffleBlock(in_channels, out_channels, stride))
        elif block_type == 'ghost':
            blocks.append(GhostBlock(in_channels, out_channels, stride))
        elif block_type == 'residual':
            blocks.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 剩余块
        for i in range(1, num_blocks):
            if block_type == 'basic':
                blocks.append(BasicBlock(out_channels, out_channels, 1))
            elif block_type == 'inverted_residual':
                blocks.append(InvertedResidual(out_channels, out_channels, 1, expansion_ratio=6))
            elif block_type == 'shuffle':
                blocks.append(ShuffleBlock(out_channels, out_channels, 1))
            elif block_type == 'ghost':
                blocks.append(GhostBlock(out_channels, out_channels, 1))
            elif block_type == 'residual':
                blocks.append(ResidualBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*blocks)

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
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.last_conv(out)
        out = self.head(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio=6):  # 保持原论文的6
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expansion_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        # Expansion layer
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Projection layer
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        
        in_channels = max(2, in_channels)
        out_channels = max(2, out_channels)
        if in_channels % 2 != 0:
            in_channels += 1
        if out_channels % 2 != 0:
            out_channels += 1
            
        mid_channels = out_channels // 2

        if stride == 1:
            self.branch_main = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 
                         groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch_main = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, 
                         groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            
            self.branch_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, stride, 1, 
                         groups=in_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )

    def _channel_shuffle(self, x):
        b, c, h, w = x.size()
        x = x.reshape(b, 2, c//2, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)
        return x
    
    def forward(self, x):
        if self.stride == 1:
            split_size = x.size(1) // 2
            x1 = x[:, :split_size, :, :]
            x2 = x[:, split_size:, :, :]
            out = torch.cat([x1, self.branch_main(x2)], dim=1)
        else:
            x1 = self.branch_shortcut(x)
            x2 = self.branch_main(x)
            out = torch.cat([x1, x2], dim=1)
            
        out = self._channel_shuffle(out)
        return out

class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(GhostBlock, self).__init__()
        in_channels = max(2, in_channels)
        out_channels = max(2, out_channels)
        
        if out_channels % 2 != 0:
            out_channels += 1
        
        primary_out = out_channels // 2
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, primary_out, 1, stride, 0, bias=False),
            nn.BatchNorm2d(primary_out),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(primary_out, primary_out, 3, 1, 1, 
                      groups=primary_out, bias=False),
            nn.BatchNorm2d(primary_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def test_precision_balanced_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    x = torch.randn(4, 1, 224, 224).to(device)
    block_types = ['basic', 'residual', 'inverted_residual', 'shuffle', 'ghost']
    
    print("=== Precision Balanced Parameter Comparison ===")
    
    results = {}
    for block_type in block_types:
        try:
            model = PrecisionBalancedCNN(
                block_type=block_type,
                in_channels=1
            ).to(device)
            
            params = sum(p.numel() for p in model.parameters())
            output = model(x)
            
            results[block_type] = {
                'params': params,
                'output_shape': output.shape
            }
            
            print(f"\n--- {block_type.upper()} ---")
            print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
            print(f"Output shape: {output.shape}")
            summary(model, (1, 224, 224))
            
        except Exception as e:
            print(f"\n--- {block_type.upper()} ---")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 汇总比较
    print("\n" + "="*50)
    print("SUMMARY COMPARISON:")
    print("="*50)
    for block_type, result in results.items():
        print(f"{block_type:15} | {result['params']:>8,} params | {result['params']/1e6:>5.2f}M")

if __name__ == "__main__":
    test_precision_balanced_models()