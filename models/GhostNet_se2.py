import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

def _make_divisible(v, divisor=4, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

'''
参数说明：in_channels 是输入特征图的通道数；reduction 是降维比例（默认 4），用于减少全连接层的参数量。
Squeeze 操作：self.avg_pool = nn.AdaptiveAvgPool2d(1) 是自适应平均池化，将任意空间大小（如 H×W）的特征图压缩为 1×1，
最终每个通道得到一个 “全局统计值”（形状：(b, c, 1, 1)，b 为批次大小，c 为通道数）。
Excitation 操作：通过两层全连接层（self.fc）学习通道权重：
第一层全连接：将通道数从 in_channels 降为 in_channels//reduction（降维，减少计算量），无偏置。
ReLU 激活：引入非线性，增强表达能力。
第二层全连接：将通道数从 in_channels//reduction 恢复为 in_channels（升维），无偏置。
Hardsigmoid 激活：输出值范围为 0-1，作为每个通道的 “重要性权重”（替代原始 SE 论文中的 Sigmoid，计算更快）。
'''
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction= 4):
        # 调用父类的构造函数
        super(SELayer, self).__init__()
        # 初始化自适应平均池化层，输出特征图大小为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 初始化全连接层序列
        self.fc = nn.Sequential(
            # 第一个全连接层，输入通道数为in_channels，输出通道数为in_channels // reduction，偏置项为False
            nn.Linear(in_channels, in_channels // reduction, bias= False), # 降维
            # ReLU激活函数，inplace=True表示直接在原变量上进行修改
            nn.ReLU(inplace= True),
            # 第二个全连接层，输入通道数为in_channels // reduction，输出通道数为in_channels，偏置项为False
            nn.Linear(in_channels // reduction, in_channels, bias=False), # 升维
            # Hardsigmoid激活函数
            nn.Hardsigmoid() # Excitation: 输出0-1权重
        )
    
    '''
    步骤解析：
        输入 x 的形状为 (b, c, h, w)（批次、通道、高度、宽度）。
        Squeeze：self.avg_pool(x) 将 x 压缩为 (b, c, 1, 1)，再通过 view(b, c) 展平为 (b, c)，适配全连接层输入。
        Excitation：self.fc(y) 对展平后的向量计算通道权重（形状：(b, c)），
        再通过 view(b, c, 1, 1) 调整为与原特征图空间维度兼容的形状（便于广播相乘）。
        应用权重：x * y 中，y 的形状 (b, c, 1, 1) 会通过广播与 x (b, c, h, w) 逐元素相乘，实现 “重要通道增强、次要通道抑制”。
    '''
    def forward(self, x):
        # 获取输入x的维度，分别为批次大小b、通道数c、高度和宽度
        b, c, _, _ = x.size()
        # 对输入x进行平均池化，并调整形状为(b, c)
        y = self.avg_pool(x).view(b, c)
        # 对池化后的结果y进行全连接操作，并调整形状为(b, c, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # 返回输入x与经过全连接操作后的y的逐元素乘积
        return x * y

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)
        init_channels = init_channels if init_channels > 0 else 1

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, init_channels * (ratio - 1), kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(init_channels * (ratio - 1)),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, use_se=False, se_ratio=0.25):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se

        self.ghost1 = GhostModule(in_ch, mid_ch, relu=True)

        if stride > 1:
            self.dwconv = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch)
            )
        
        self.ghost2 = GhostModule(mid_ch, out_ch, relu=False)

        # 添加SE注意力机制
        if use_se:
            self.se = SELayer(out_ch)
        else:
            self.se = None

        self.shortcut = nn.Sequential()
        if stride == 1 and in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.stride > 1:
            x = self.dwconv(x)
    
        x = self.ghost2(x)

        # 应用SE注意力机制
        if self.se is not None:
            x = self.se(x)

        if self.stride == 1:
            x += self.shortcut(residual)
        
        return x

class GhostNet(nn.Module):
    def __init__(self, n_classes=10, in_ch=3, width_mult=1.0, use_se=True, se_ratio=0.25):
        super(GhostNet, self).__init__()

        # 修改配置，添加use_se标志位
        cfgs = [
            # k, c, exp_size, s, use_se
            [1,  16,  16, 1, False],  # 第一个阶段通常不使用SE
            [2,  24,  48, 2, use_se],
            [2,  32,  72, 2, use_se],
            [3,  64, 120, 2, use_se],
            [2, 160, 240, 1, use_se],
            [3, 160, 272, 2, use_se],
            [4, 160, 384, 1, use_se],
            [2, 160, 576, 2, use_se],
            [3, 160, 960, 1, use_se],
        ]

        out_ch = _make_divisible(16 * width_mult, 4)
        layers = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        ]
        input_ch = out_ch

        # 根据配置构建网络，传递use_se参数
        for k, exp_size, c, s, use_se_block in cfgs:
            out_ch = _make_divisible(c * width_mult, 4)
            exp_size = _make_divisible(exp_size * width_mult, 4)
            for i in range(k):
                stride = s if i == 0 else 1
                layers.append(GhostBottleneck(input_ch, exp_size, out_ch, 
                                            stride=stride, 
                                            use_se=use_se_block,
                                            se_ratio=se_ratio))
                input_ch = out_ch
        
        out_ch = _make_divisible(1280 * width_mult, 4)
        layers.append(nn.Sequential(
            nn.Conv2d(input_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建不带SE的GhostNet
    # model_without_se = GhostNet(n_classes=10, use_se=False).to(device)
    
    # 创建带SE的GhostNet
    model_with_se = GhostNet(n_classes=10, use_se=True, se_ratio=0.25).to(device)
    
    # 测试输入
    x = torch.randn(100, 3, 224, 224)
    summary(model_with_se, x.shape[1:])

    
