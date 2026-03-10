import torch
from torch import nn
import torch.nn.functional as F



'''
ratio：控制 “固有特征” 与 “幻影特征” 的比例（默认 2）。例如ratio=2时，固有特征占 1/2，幻影特征占 1/2；
init_channels：固有特征图的通道数（out_channels / ratio），是整个模块的 “基础特征”；
primary_conv：用 1x1 卷积生成固有特征（1x1 卷积计算量远低于 3x3，适合降维）；
cheap_operation：用深度卷积（group convolution with groups = 输入通道）生成幻影特征。
深度卷积的计算量是普通卷积的 1 / 输入通道数，属于 “廉价操作”。
'''
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size= 1, ratio= 2, stride= 1, relu= True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)
        init_channels = init_channels if init_channels > 0 else 1

        # 第一步：少量卷积生成固有特征图（通常用1x1卷积降维/升维）
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias= False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace= True) if  relu else nn.Sequential()
        )

        # 第二步：用廉价操作（深度卷积）生成幻影特征图
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, init_channels * (ratio - 1), kernel_size= 3, stride= 1, padding= 1, groups= init_channels, bias= False),
            nn.BatchNorm2d(init_channels * (ratio - 1)),
            nn.ReLU(inplace= True) if relu else nn.Sequential()
        )
    
    '''
    输入x先通过primary_conv生成固有特征x1（通道数init_channels）；
    用cheap_operation对x1处理，生成幻影特征x2（通道数init_channels*(ratio-1)）；
    沿通道维度（dim=1）拼接x1和x2，总通道数为init_channels*ratio；
    最后截断到out_channels（因为init_channels可能因除法取整与目标通道数有微小偏差）。
    '''
    def forward(self, x):
        # 执行主卷积操作
        x1 = self.primary_conv(x)

        # 执行廉价操作
        x2 = self.cheap_operation(x1)

        # 将x1和x2沿着第1维（列）拼接
        out = torch.cat([x1, x2], dim= 1)

        # 返回拼接后的结果，只保留前self.out_channels个通道
        return  out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    '''
    in_ch：输入特征图的通道数；
    mid_ch：中间特征图的通道数（通常是in_ch的几倍，如 4 倍，用于扩展特征维度，增强表达能力）；
    out_ch：输出特征图的通道数；
    stride：步长（stride=1时不改变特征图尺寸；stride>1时进行下采样，缩小特征图尺寸）。
    '''
    def __init__(self, in_ch, mid_ch, out_ch, stride= 1):
        # 调用父类的构造函数
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # 创建第一个Ghost模块
        # ghost1：作为 “扩展层”，将输入通道in_ch映射到中间通道mid_ch（通常是升维，如从 64→256）。
        # 通过GhostModule的 “固有特征 + 幻影特征” 生成更多中间特征，同时控制计算量。
        self.ghost1  = GhostModule(in_ch, mid_ch, relu= True)

        # 如果步长大于1，则添加深度可分离卷积和批量归一化
        # dwconv（可选）：当stride>1时（需要下采样），用深度可分离卷积（groups=mid_ch，每个通道单独卷积）实现下采样。
        # 相比传统 3x3 卷积，深度卷积的计算量仅为其 1/mid_ch，极大降低成本。
        if stride > 1:
            self.dwconv = nn.Sequential(
                # 深度可分离卷积
                nn.Conv2d(mid_ch, mid_ch, kernel_size= 3, stride= stride, padding= 1, groups= mid_ch, bias= False),
                # 批量归一化
                nn.BatchNorm2d(mid_ch)
            )
        
        # 创建第二个Ghost模块
        # ghost2：作为 “压缩层”，将中间通道mid_ch映射到输出通道out_ch（通常是降维，如从 256→64），同样通过GhostModule高效生成特征。
        self.ghost2 = GhostModule(mid_ch, out_ch, relu= True)

        # 初始化捷径（shortcut）
        self.shortcut = nn.Sequential()
        # 如果步长为1且输入通道数不等于输出通道数，则添加捷径
        # shortcut：残差连接通路。
        # 当stride=1（特征图尺寸不变）且in_ch != out_ch时，用 1x1 卷积调整输入通道至out_ch，保证与ghost2的输出维度匹配，实现 “残差相加”；
        # 若通道相同，则直接传递输入（nn.Sequential()为空操作）。
        if stride == 1 and in_ch != out_ch:
            self.shortcut = nn.Sequential(
                # 1x1卷积
                nn.Conv2d(in_ch, out_ch, kernel_size= 1, stride= 1, padding= 0, bias= False),
                # 批量归一化
                nn.BatchNorm2d(out_ch)
            )

    '''
    输入x → 
    ghost1（升维生成中间特征） → 
    （若下采样则经dwconv） → 
    ghost2（降维生成输出特征） → 
    （若不下采样则加shortcut的残差） → 
    最终输出。
    '''
    def forward(self, x):
        # 保存原始输入
        residual = x

        # 应用第一个Ghost模块
        x = self.ghost1(x)

        # 如果步长大于1，则应用深度可分离卷积
        if self.stride > 1:
            x = self.dwconv(x)
    
        # 应用第二个Ghost模块
        x = self.ghost2(x)

        # 如果步长为1，则将原始输入通过shortcut后与当前输出相加
        if self.stride == 1:
            x += self.shortcut(residual)
        
        return x

'''
用GhostBottleneck替代传统卷积块，通过 “固有特征 + 幻影特征” 的低成本生成方式减少冗余计算；
引入 “宽度乘数（width_mult）” 动态调整网络通道数，灵活适配不同算力需求（如width_mult=0.5为轻量版，1.0为标准版）；
遵循 “特征图逐步缩小、通道数逐步增加” 的经典卷积网络设计，保证特征提取能力。
二、代码核心组件解析
''' 
class GhostNet(nn.Module):
    def __init__(self, n_classes = 10, in_ch= 3, width_mult= 1.0):
        super(GhostNet, self).__init__()

        '''
        每个子列表对应网络的一个 “阶段”，参数含义：
        k：该阶段中GhostBottleneck的重复次数（控制网络深度）；
        c：该阶段输出特征图的通道数（控制网络宽度）；
        exp_size：GhostBottleneck中中间扩展层的通道数（用于特征扩展，增强表达能力）；
        s：该阶段第一个GhostBottleneck的步长（s=2时进行下采样，缩小特征图尺寸；s=1时保持尺寸）。
        '''
        cfgs = [
            # k: 重复次数, c: 输出通道, t: 扩展因子, s: 步长
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

        '''
        作用：确保网络通道数是divisor（默认 4）的整数倍。
        原因：移动端硬件（如 GPU）对 4 的整数倍通道数的计算效率更高，可加速推理。
        逻辑：将输入v（原始通道数）调整为最接近且不小于v的divisor倍数（如v=17、divisor=4时，输出20）。
        '''
        def _make_divisible(v, divisor= 4, min_value= None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return  new_v

        '''
        作用：对原始输入图像（如 3 通道 RGB 图）进行第一次特征提取和下采样。
        细节：3x3 卷积（步长 2）将输入尺寸缩小一半（如 224x224→112x112），通道数调整为16*width_mult的整数倍（由_make_divisible处理）。
        '''
        out_ch = _make_divisible(16 * width_mult, 4)
        layers = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size= 3, stride= 2, padding= 1, bias= False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace= True)
            )
        ]
        input_ch = out_ch

        '''
        按cfgs配置逐步构建网络的每个阶段：
        每个阶段包含k个GhostBottleneck，第一个负责下采样（s=2）或保持尺寸（s=1），后续k-1个仅做特征提取（步长 1）；
        所有通道数通过width_mult缩放（如width_mult=0.5时，通道数减半），并经_make_divisible调整为 4 的倍数。
        '''
        for k, exp_size, c, s in cfgs:
            out_ch = _make_divisible(c * width_mult, 4)
            exp_size = _make_divisible(exp_size * width_mult, 4)
            for i in range(k):
                if i == 0:
                    layers.append(GhostBottleneck(input_ch, exp_size, out_ch, stride= s))
                else:
                    layers.append(GhostBottleneck(input_ch, exp_size, out_ch, stride= 1))
                input_ch = out_ch
        
        '''
        特征整合：1x1 卷积将最后一个GhostBottleneck的输出通道映射到1280*width_mult（经调整），融合全局特征；
        分类头：通过自适应平均池化将特征图压缩为 1x1 向量，展平后经全连接层输出n_classes个类别概率。
        '''
        out_ch = _make_divisible(1280 * width_mult, 4)
        layers.append(nn.Sequential(
            nn.Conv2d(input_ch, out_ch, kernel_size= 1, stride= 1, padding= 0, bias= False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace= True)
        ))
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_ch, n_classes)

        # _initialize_weights：卷积层用 Kaiming 正态初始化（适合 ReLU 激活），
        # 批归一化层权重设 1、偏置设 0，保证训练稳定性；
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode= 'fan_out', nonlinearity= 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x