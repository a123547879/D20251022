import torch
from torch import nn

class ShuffleV2Block(nn.Module):
    '''
    inp：输入特征图通道数；
    oup：输出特征图总通道数（= 主分支输出通道 + 投影分支输出通道）；
    mid_ch：主分支中 1x1 卷积的中间通道数（用于特征扩展）；
    ksize：深度卷积的核大小（通常 3，即 3x3）；
    stride：步长（1：同尺寸特征提取；2：下采样，缩小特征图尺寸）。
    '''
    def __init__(self, inp, oup, mid_ch, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride= stride
        assert stride in [1, 2], "stride should be 1 or 2."

        self.mid_ch = mid_ch
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        '''
        主分支（branch_main）：
            负责特征变换，通过 “1x1 卷积（升维）→ 深度卷积（空间特征）→ 1x1 卷积（降维）” 的流程，输出通道数为outputs = oup - inp；
        投影分支（branch_proj）：
            仅在stride=2（下采样）时启用，对原始输入进行下采样和特征调整，输出通道数为inp（与输入通道一致），
            最终与主分支输出拼接后总通道数为oup。
        '''
        branch_main = [
            # pw
            nn.Conv2d(inp, mid_ch, 1, 1, 0, bias= False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace= True),
            # dw
            nn.Conv2d(mid_ch, mid_ch, ksize, stride, pad, groups= mid_ch, bias= False),
            nn.BatchNorm2d(mid_ch),
            # pw-linaer
            nn.Conv2d(mid_ch, outputs, 1, 1, 0, bias= False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace= True)
        ]

        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw 3*3
                nn.Conv2d(inp, inp, ksize, stride, pad, groups= inp, bias= False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias= False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace= True)
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None
    
    '''
    核心作用：打破分组卷积的通道隔离。
    例如：若输入通道为c=4（分成 2 组，每组 2 通道），通道混洗后会将两组的通道重新分配，使每个分支都包含原两组的特征，促进跨组信息交流。
    必要性：分组卷积虽减少计算量，但组内特征独立，导致信息封闭。通道混洗让不同组的特征交叉融合，提升特征表达能力。
    '''
    def channel_shuffle(self, x):
        b, c, h, w = x.data.size()
        assert (c % 2 == 0), "the channel of input tensor must be even."
        x = x.reshape(b * c // 2, 2, h * w)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, c // 2, h, w)
        return x[0], x[1]
    
    '''
    stride=1（同尺寸）：
    输入 → 通道混洗 → 分成 x_proj（直接作为投影分支）和 x（送入主分支） → 主分支输出与 x_proj 拼接 → 输出（通道数不变）。
    stride=2（下采样）：
    输入同时送入投影分支（下采样 + 处理）和主分支（下采样 + 处理） → 两者输出拼接 → 输出（通道数为 oup，尺寸缩小为输入的 1/2）
    '''
    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            # print(x_proj.shape, x.shape)
            return torch.cat((x_proj, self.branch_main(x)), dim= 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), dim= 1)
        
class ShuffleNetV2(nn.Module):
    '''
    ShuffleNet V2 的设计严格遵循移动端高效网络的 4 条核心原则（来自论文实验结论）：
    保持输入输出通道数相等（减少内存访问成本）；
    避免使用过多分组卷积（分组越多，内存访问成本越高）；
    减少网络分支数量（分支越多，并行计算效率越低）；
    卷积核尺寸与通道数平衡（1x1 卷积计算成本应与 3x3 卷积匹配）。
    代码通过动态通道配置、ShuffleV2Block的分支设计和通道混洗机制，完美契合这些原则，实现 “高精度 + 低资源消耗”。
    '''
    def __init__(self, inp= 224, n_classes= 10, model_size= '1.0x', in_ch= 3):
        super(ShuffleNetV2, self).__init__()
        print('model size is {}'.format(model_size))

        '''
        stage_repeats：定义网络的 3 个特征提取阶段中，ShuffleV2Block的重复次数（分别为 4、8、4 次），决定网络深度；
        stage_out_channels：每个阶段的输出通道数（索引 0 为占位，1 为初始卷积输出，2-4 为 3 个特征阶段输出，
        5 为最终特征整合通道），通过model_size动态调整（如0.5x通道数减半，适合低端设备；
        2.0x通道数增加，适合算力较强设备）。
        '''
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError
        
        '''
        作用：对原始输入图像（如 3 通道 RGB 图）进行初步特征提取和下采样：
        first_conv：3x3 卷积（步长 2）将输入尺寸从224x224→112x112，通道数从 3→24；
        max_pool：3x3 最大池化（步长 2）进一步将尺寸从112x112→56x56，减少后续计算量。
        '''
        inp_ch = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_ch, inp_ch, kernel_size= 3, stride= 2, padding= 1, bias= False),
            nn.BatchNorm2d(inp_ch),
            nn.ReLU(inplace= True)
        )

        self.max_pool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)

        '''
        阶段划分与下采样逻辑：网络共 3 个特征阶段，每个阶段通过ShuffleV2Block逐步提升特征抽象度：
        阶段 1：4 个ShuffleV2Block，第一个步长 2（尺寸56x56→28x28），输出通道 116（1.0x 模型）；
        阶段 2：8 个ShuffleV2Block，第一个步长 2（尺寸28x28→14x14），输出通道 232；
        阶段 3：4 个ShuffleV2Block，第一个步长 2（尺寸14x14→7x7），输出通道 464；
        每个阶段的后续 Block 步长为 1，仅做特征变换，不改变尺寸。
        ShuffleV2Block参数：mid_ch=out_ch//2（中间通道数为输出的 1/2），保证 1x1 卷积与 3x3 深度卷积的计算成本平衡（符合 ShuffleNet V2 设计原则）。
        '''
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            out_ch = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(inp_ch, out_ch, mid_ch= out_ch // 2, ksize= 3, stride= 2))
                else:
                    self.features.append(ShuffleV2Block(inp_ch // 2, out_ch, mid_ch= out_ch // 2, ksize= 3, stride= 1))
                
                inp_ch = out_ch
        self.features = nn.Sequential(*self.features)
        
        '''
        特征整合：conv_last通过 1x1 卷积将最后一个特征阶段的输出（464 通道）
        映射到stage_out_channels[-1]（1024 或 2048 通道），融合全局特征；
        分类头：
        global_pool 7x7 特征图压缩为 1x1 向量；
        2.0x 模型通过dropout减少过拟合；
        全连接层classifier将特征向量映射到n_classes个类别概率。
        '''
        self.conv_last = nn.Sequential(
            nn.Conv2d(inp_ch, self.stage_out_channels[-1], kernel_size= 1, stride= 1, padding= 0, bias= False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace= True)
        )

        self.global_pool = nn.AvgPool2d(7)
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_classes, bias= False))
        self.__initialize_weights()


    # __initialize_weights：
    # 卷积层用 Kaiming 初始化（适合 ReLU），
    # 批归一化层权重设 1、
    # 偏置设 0，
    # 全连接层用正态分布初始化，保证训练稳定性；
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode= 'fan_out', nonlinearity= 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.max_pool(x)
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = self.conv_last(x)

        x = self.global_pool(x)
        if self.model_size == '2.0x':
            self.dropout(x)
        
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    net = ShuffleNetV2(model_size= '1.0x')
    img = torch.randn(2, 3, 224, 224)
    print(net(img).shape)