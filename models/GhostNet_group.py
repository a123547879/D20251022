import torch
from torch import nn
import torch.nn.functional as F

class GhostBottleneck(nn.Module):
    '''
    使用深度可分离卷积替换GhostModule的GhostBottleneck
    in_ch：输入特征图的通道数；
    mid_ch：中间特征图的通道数；
    out_ch：输出特征图的通道数；
    stride：步长
    '''
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # 第一个深度可分离卷积模块 - 用于特征扩展
        self.ghost1 = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride, 
                     groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            # 逐点卷积 - 扩展通道数
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True)
        )

        # 如果步长大于1，添加额外的深度卷积进行下采样
        if stride > 1:
            self.dwconv = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, 
                         padding=1, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch)
            )
        else:
            self.dwconv = nn.Identity()

        # 第二个深度可分离卷积模块 - 用于特征压缩
        self.ghost2 = nn.Sequential(
            # 深度卷积
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1,
                     groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            # 逐点卷积 - 压缩通道数
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride == 1 and in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = x

        # 第一个ghost模块
        x = self.ghost1(x)
        
        # 下采样（如果需要）
        x = self.dwconv(x)
        
        # 第二个ghost模块
        x = self.ghost2(x)

        # 残差连接
        if self.stride == 1:
            x += self.shortcut(residual)
        
        return x


class GhostNet(nn.Module):
    def __init__(self, n_classes=10, in_ch=3, width_mult=1.0):
        super(GhostNet, self).__init__()

        cfgs = [
            # k: 重复次数, c: 输出通道, exp_size: 扩展通道, s: 步长
            [1,  16, 16, 1],
            [2,  24, 48, 2],
            [2,  32, 72, 2],
            [3,  64, 120, 2],
            [2, 160, 240, 1],
            [3, 160, 272, 2],
            [4, 160, 384, 1],
            [2, 160, 576, 2],
            [3, 160, 960, 1],
        ]

        def _make_divisible(v, divisor=4, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # 初始卷积层
        out_ch = _make_divisible(16 * width_mult, 4)
        layers = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            )
        ]
        input_ch = out_ch

        # 构建GhostBottleneck层
        for k, exp_size, c, s in cfgs:
            out_ch = _make_divisible(c * width_mult, 4)
            exp_size = _make_divisible(exp_size * width_mult, 4)
            for i in range(k):
                stride = s if i == 0 else 1
                layers.append(GhostBottleneck(input_ch, exp_size, out_ch, stride=stride))
                input_ch = out_ch
        
        # 分类头
        out_ch = _make_divisible(1280 * width_mult, 4)
        layers.append(nn.Sequential(
            nn.Conv2d(input_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_ch, n_classes)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = GhostNet(n_classes=10, width_mult=1.0)
    
    # 测试输入
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")