import torch
from torch import nn
import torch.nn.functional as F

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
'''
这段代码实现了一个轻量级瓶颈（Bottleneck）模块，结合了深度可分离卷积（Depthwise Separable Convolution）、
可选的 SE 注意力机制和残差连接，常用于 MobileNetV2/V3 等高效神经网络中。
其核心设计是通过 “升维 - 深度卷积 - 降维” 的流程减少计算量，同时保留特征表达能力。
in_channels：输入特征图的通道数
exp_channels：“扩张” 后的通道数（用于中间特征增强）
out_channels：输出特征图的通道数
kernel_size：深度卷积的卷积核大小（如 3 或 5）
stride：深度卷积的步长（控制特征图尺寸是否缩小）
use_se：是否使用 SE 注意力模块（True/False）
act：激活函数类型（支持'h-switch'（Hardswish）或'relu'）
'''    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, act):
        super(Bottleneck, self).__init__()
        self.stride = stride
        # 只有当步长为 1（特征图尺寸不变）且输入输出通道数相同时，
        # 才启用残差连接（避免维度不匹配），与 ResNet 的残差逻辑一致。
        self.use_residual = (stride == 1) and (in_channels == out_channels)  # 判断是否使用残差连接

        # 1. 升维（1x1卷积）
        '''
        作用：通过 1x1 卷积将输入通道数从in_channels扩展到exp_channels（升维），增强中间特征的表达能力。
        细节：无偏置（批归一化已包含偏置功能），后接批归一化和指定激活函数。
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size= 1, padding= 0, bias= False),
            nn.BatchNorm2d(exp_channels),
            self._get_activation(act)  # 获取激活函数 (Relu or Swish)
        )

        # 2.深度可分离卷积（3x3or5x5）
        '''
        这是深度卷积（Depthwise Convolution），属于深度可分离卷积的第一部分：
        groups=exp_channels：将输入的exp_channels个通道分为exp_channels组，
        每组单独用一个卷积核卷积（即 “逐通道卷积”），大幅减少计算量（相比标准卷积，计算量降至 1/exp_channels）。
        padding=(kernel_size-1)//2：确保步长为 1 时，
        卷积后特征图尺寸不变（例如 kernel_size=3 时 padding=1，3x3 卷积后尺寸不变）。
        后接批归一化和激活函数，进一步处理特征。
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            exp_channels, exp_channels,
            kernel_size= kernel_size,
            stride= stride,
                padding= (kernel_size - 1) // 2,  # 保持尺寸（步长1时）
                groups= exp_channels,  # 分组卷积，通道数等于exp_channels
            bias= False
            ),
            nn.BatchNorm2d(exp_channels),
            self._get_activation(act)
        )

        # 3.SE注意力模块(可选)
        # 若use_se=True，则插入 SE 模块（通道注意力），增强重要通道的特征；否则用nn.Identity()占位（不做处理），保持网络结构一致性。
        # 在 PyTorch 中，nn.Identity() 是一个特殊的模块，它的功能非常简单：对输入不做任何处理，直接返回输入本身（恒等映射）。
        self.se = SELayer(exp_channels) if use_se else nn.Identity()

        # 4.降维（1x1卷积, 线性激活）
        '''
        作用：通过 1x1 卷积将通道数从exp_channels压缩到out_channels（降维），减少输出特征的通道数，降低后续计算量。
        细节：无激活函数（线性输出），仅保留批归一化，避免破坏残差连接的数值稳定性。
        '''
        self.conv3 = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, kernel_size= 1, padding= 0, bias= False),
            nn.BatchNorm2d(out_channels)
        )

    def _get_activation(self, act):
        # 如果激活函数为 'h-switch'
            if act == 'h-swish':
                return nn.Hardswish(inplace= True) # h-switch(x) = x * max(0, min(x + 3, 6)) / 6 = x * ReLU6(x+3)/6
        # 如果激活函数为 'relu'
            elif act == 'relu':
                return nn.ReLU(inplace= True)
        # 如果激活函数不是上述两种之一
            else:
                raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x):
        residual = x

        # 前向传播
        x = self.conv1(x)

        # DW: Depthwise Separable Convolution
        x = self.conv2(x)

        # SE
        x = self.se(x)
        
        # 最后的1x1卷积
        x = self.conv3(x)

        if self.use_residual:
            x += residual
        return x
    
'''
设备和嵌入式系统设计，结合了深度可分离卷积、SE 注意力机制、动态激活函数等优化策略，在精度和效率之间取得了良好平衡。
核心设计特点
MobileNetV3 的核心改进包括：
沿用 MobileNetV2 的 “倒残差结构”（1x1升维→深度卷积→1x1降维）；
引入 SE（Squeeze-and-Excitation）注意力模块，增强关键通道特征；
使用 Hardswish（h-swish）替代部分 ReLU，提升精度同时保持高效性；
针对不同场景设计large（高精度）和small（高速度）两个版本。
version：指定 MobileNetV3 版本（'large'或'small'），两者结构和参数不同；
n_classes：分类任务的类别数（默认 10，适合 CIFAR 等数据集）；
ch_in：输入图像的通道数（默认 3，对应 RGB 图像）。
'''
class MobileNetV3(nn.Module):
    def __init__(self, version= 'large', n_classes= 10, in_ch= 3, use_se= True):
        super(MobileNetV3, self).__init__()
        assert version in ['large', 'small'], 'version must be large or small'

        # 配置参数：(in_channels, exp_channels, out_channels, kernel_size, stride, use_se, activation)
        '''
        根据版本定义瓶颈层（Bottleneck）的参数，每个配置项为一个元组：
        (in_channels, exp_channels, out_channels, kernel_size, stride, use_se, activation)
        对应 Bottleneck 的输入通道、扩张通道、输出通道、卷积核大小、步长、是否用 SE 模块、激活函数。
        large版本：配置更复杂，通道数和层数更多，适合对精度要求高的场景；
        small版本：结构更精简，计算量更小，适合对速度要求高的场景。
        '''
        if version == 'large':
            self.config = [
                (16, 16, 16, 3, 1, False, 'relu'),
                (16, 64, 24, 3, 2, False, 'relu'),
                (24, 72, 24, 3, 1, False, 'relu'),
                (24, 72, 40, 5, 2, True, 'relu'),
                (40, 120, 40, 5, 1, True, 'relu'),
                (40, 120, 40, 5, 1, True, 'relu'),
                (40, 240, 80, 3, 2, False, 'h-swish'),
                (80, 200, 80, 3, 1, False, 'h-swish'),
                (80, 184, 80, 3, 1, False, 'h-swish'),
                (80, 184, 80, 3, 1, False, 'h-swish'),
                (80, 480, 112, 3, 1, True, 'h-swish'),
                (112, 672, 112, 3, 1, True, 'h-swish'),
                (112, 672, 160, 5, 2, True, 'h-swish'),
                (160, 960, 160, 5, 1, True, 'h-swish'),
                (160, 960, 160, 5, 1, True, 'h-swish'),
            ]
            final_exp_channels = 1280 # 最终的扩展通道数
        else:
            self.config = [
                 (16, 16, 16, 3, 2, use_se, 'relu'),
                (16, 72, 24, 3, 2, False, 'relu'),
                (24, 88, 24, 3, 1, False, 'relu'),
                (24, 96, 40, 5, 2, use_se, 'h-swish'),
                (40, 240, 40, 5, 1, use_se, 'h-swish'),
                (40, 240, 40, 5, 1, use_se, 'h-swish'),
                (40, 120, 48, 5, 1, use_se, 'h-swish'),
                (48, 144, 48, 5, 1, use_se, 'h-swish'),
                (48, 288, 96, 5, 2, use_se, 'h-swish'),
                (96, 576, 96, 5, 1, use_se, 'h-swish'),
                (96, 576, 96, 5, 1, use_se, 'h-swish'),
            ]
            final_exp_channels = 1024 # 最终的扩展通道数

        # 特征提取层
        self.features = nn.Sequential()
        # 初始卷积层 (3x3, stride:2)
        '''
        作用：对输入图像进行初步特征提取，同时通过步长 2 缩小空间尺寸（如 224x224→112x112）。
        细节：3x3 卷积（感受野适中），输出 16 通道，后接批归一化和 Hardswish 激活（MobileNetV3 起始层常用）。
        '''
        self.features.add_module(
            'init_conv',
            nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size= 3, stride= 2, padding= 1, bias= False),
                nn.BatchNorm2d(16),
                nn.Hardswish(inplace= True)
            )
        )

        # 添加Bottleneck模块
        '''
        作用：通过多个瓶颈层逐步提取高级特征，是特征提取的核心。
        每个 Bottleneck 的功能：
        用 1x1 卷积升维（增强特征表达）；
        深度可分离卷积（减少计算量）；
        可选 SE 模块（增强关键通道）；
        1x1 卷积降维（压缩特征）；
        满足条件时添加残差连接（缓解梯度消失）。
        '''
        for i, (in_c, exp_c, out_c, k, s, use_se, act) in enumerate(self.config):
            self.features.add_module(
                f'bottleneck_{i}',
                Bottleneck(in_c, exp_c, out_c, k, s, use_se, act)
            )

        # 最后的卷积层 (升维)
        '''
        作用：将最后一个瓶颈层的输出通道数扩展到final_exp_channels（large为 1280，small为 1024），进一步增强特征表达。
        细节：1x1 卷积（仅改变通道数，不改变空间尺寸），后接批归一化和 Hardswish。 
        '''
        last_out_channels = self.config[-1][2]
        self.features.add_module(
            'final_conv',
            nn.Sequential(
                nn.Conv2d(last_out_channels, final_exp_channels, kernel_size= 1, stride= 1, padding= 0, bias= False),
                nn.BatchNorm2d(final_exp_channels),
                nn.Hardswish(inplace= True)
            )
        )
        
        # 分类器
        '''
        全局平均池化：将任意空间尺寸的特征图压缩为 1x1，保留通道维度的全局信息；
        1x1 卷积降维：将final_exp_channels降至 512（large）或 256（small），减少后续计算量；
        Dropout：随机丢弃 20% 的神经元，缓解过拟合；
        输出层：1x1 卷积将通道数映射到n_classes，等价于全连接层（因输入为 1x1，卷积操作退化为矩阵乘法）。
        '''
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局平均池化：(b, c, 1, 1)
            nn.Conv2d(final_exp_channels, 512 if version == 'large' else 256, kernel_size= 1, stride= 1, bias= False),
            nn.Hardswish(inplace= True),
            nn.Dropout(0.2),
            nn.Conv2d(512 if version == 'large' else 256, n_classes, kernel_size= 1) # 最后的全连接层
        )

    def forward(self, x):
        x = self.features(x) # 特征提取：(b, c, h, w)
        x = self.classifier(x) # 分类头：(b, num_classes, 1, 1)
        return x.view(x.size(0), -1) # 展平为(b, num_classes)
