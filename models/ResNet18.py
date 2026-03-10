import torch
import torch.nn as nn
from torchsummary import summary

class BasicBlock(nn.Module): # 定义一个BasicBlock类，继承nn.Module,适用于resnet18、34的残差结构
    expansion = 1 # 指定扩张因子为1，主分支的卷积核个数不发生变化

    # 初始化函数，定义网络层和一些参数
    def __init__(self, in_ch, out_ch, stride= 1, downsample= None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size= 3, padding= 1, stride= stride, bias= False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size= 3, padding= 1, stride= 1, bias= False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x # 记录输入的x
        if self.downsample is not None: # 如果downsample不为None，则执行下采样操作
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) 
        out += identity # 将输出与残差联相加
        out = self.relu(out)
        return out
    
class Bottleneck(nn.Module): # 定义一个Bottleneck类，继承nn.Module,适用于resnet50、101、152的残差结构
    expansion = 4

    def __init__(self, in_ch, out_ch, stride= 1, downsample= None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size= 1, stride= stride, bias= False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size= 3, padding= 1, stride= 1, bias= False)
        self.bn2= nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch  * self.expansion, kernel_size= 1, stride= 1, padding= 0, bias= False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, n_classes= 10, in_ch= 3, incldue_top= True):
        super(ResNet, self).__init__()
        self.include_top = incldue_top
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_ch, self.in_channels, kernel_size= 7, stride= 2, padding= 3, bias= False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride= 2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride= 2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride= 2)
        if self.include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 初始化卷积层权重，使用kaiming_normal_初始化方法
                nn.init.kaiming_normal_(m.weight, mode= 'fan_out', nonlinearity= 'relu')
    
    def _make_layer(self, block, channel, blocks_num, stride= 1):
        
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size= 1, stride= stride, bias= False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, channel, stride, downsample))
        self.in_channels = channel * block.expansion

        for _ in range(1, blocks_num):
            layers.append(block(self.in_channels, channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avg_pool(x)
            # x = torch.flatten(x, 1)
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
        return x
    
def resnet18(n_classes= 10, in_ch= 3, incldue_top= True, pretrained= False):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, in_ch, incldue_top= incldue_top)
            
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18().to(device)
    # print(resnet18)
    
    x = torch.randn(1, 3, 224, 224).to(device)
    y = resnet18(x)
    print(y.shape)
    summary(resnet18, (3, 224, 224))