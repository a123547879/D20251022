import os
import torch
import numpy as np
import cv2 as cv
import torchvision
from torch import nn, linalg
from torchvision import datasets
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from 小型CNN提取特征 import HeatmapEncoder
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self.hook_handles = []

        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                break
    
    def __call__(self, x, class_idx=None):
        """
        生成GradCAM热力图
        x: 输入图像张量（shape: [1, C, H, W]）
        class_idx: 目标类别索引（None则用模型预测的最高概率类别）
        """
        self.model.eval()
        x.requires_grad_()

        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        # 获取每个样本对应的目标类别输出
        batch_size = output.shape[0]
        target = output[torch.arange(batch_size), class_idx]  # 形状: [batch]
        target.sum().backward()

        # 检查梯度是否存在
        if self.gradients is None:
            print("警告: 梯度为None，返回零热力图")
            return torch.zeros(x.shape[2], x.shape[3]).cpu().numpy()

        weights = F.adaptive_avg_pool2d(self.gradients, 1)  # [1, C, 1, 1]
        weights = weights.squeeze(-1).squeeze(-1)  # [1, C]

        cam = torch.sum(weights[:, :, None, None] * self.feature_maps, dim=1)  # [1, H, W]
        cam = F.relu(cam)
        
        # 确保cam有正确的维度
        if cam.dim() == 3:
            cam = cam.unsqueeze(1)  # [1, 1, H, W]
            
        # 替换掉interpolate后的squeeze()
        cam = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3]),
            mode='bilinear',
            align_corners=False
        )  # 形状: [batch, 1, H, W]
        cam = cam.squeeze(1)  # 仅删除通道维度，保留批次维度，形状: [batch, H, W]

        # 归一化
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.detach().cpu().numpy()
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def calculate_iou(heatmap, mask, model_type, name= '浅层', threshold=0.5):
    """计算批次IoU（支持Otsu自适应阈值和设备一致性）"""
    # --------------------------
    # 1. 统一格式与设备
    # --------------------------
    # 确保heatmap是tensor，并与mask在同一设备
    if isinstance(heatmap, np.ndarray):
        # 若输入是numpy，转为tensor并移至mask所在设备
        heatmap = torch.tensor(heatmap, dtype=torch.float32, device=mask.device)
    else:
        # 确保heatmap与mask设备一致
        heatmap = heatmap.to(mask.device).float()
    
    # 确保mask维度正确（[B, H, W]）
    if mask.ndim == 2:  # 若输入为[B, 1, H, W]，移除通道维度
        mask = mask.unsqueeze(0)  # 变为[B, H, W]
    mask_binary = mask.float()  # 转为float类型，便于计算

    # 检查形状匹配
    assert heatmap.shape == mask_binary.shape, \
        f"热力图与mask形状不匹配：{heatmap.shape} vs {mask_binary.shape}"
    batch_size = heatmap.shape[0]

    # --------------------------
    # 2. 热力图二值化（Otsu阈值法）
    # --------------------------
    heatmap_binary = []
    for idx in range(batch_size):
        # 取出单张热力图（转为CPU numpy进行OpenCV处理）
        img = heatmap[idx].cpu().detach().numpy()
        
        # 归一化到0~255（Otsu需要uint8类型）
        img_min, img_max = img.min(), img.max()
        if img_max != img_min:
            img_norm = (img - img_min) / (img_max - img_min) * 255
        else:
            img_norm = np.zeros_like(img)  # 避免全零图除零
        img_uint8 = img_norm.astype(np.uint8)
        
        # Otsu阈值二值化（返回阈值和二值图）
        _, binary_np = cv.threshold(img_uint8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # 4. 关键步骤：极性检测与校正
        # if np.mean(binary_np) / 255 >= 0.5 :
        #     binary_np = cv.bitwise_not(binary_np)

        # 转为0/1（除以255）并转回tensor，移至原设备
        binary_tensor = torch.tensor(binary_np / 255, dtype=torch.float32, device=mask.device)
        heatmap_binary.append(binary_tensor)
    
    # 拼接为批次 tensor [B, H, W]
    heatmap_binary = torch.stack(heatmap_binary, dim=0)

    # --------------------------
    # 3. 计算IoU
    # --------------------------
    intersection = torch.sum(heatmap_binary * mask_binary, dim=(1, 2))  # 交集 [B,]
    union = torch.sum(heatmap_binary + mask_binary, dim=(1, 2)) - intersection  # 并集 [B,]
    iou = intersection / torch.clamp(union, min=1e-8)  # 避免除零
    mean_iou = torch.mean(iou).item()  # 批次平均IoU

    # --------------------------
    # 4. 可视化（批次第一张图）
    # --------------------------
    fig = plt.figure(figsize=(12, 5))
    # 原始热力图
    plt.subplot(1, 3, 1)
    plt.imshow(heatmap[0].cpu().detach().numpy(), cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    # 二值化热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_binary[0].cpu().detach().numpy(), cmap='gray')
    plt.title('Binary Grad-CAM')
    plt.axis('off')
    # 目标mask
    plt.subplot(1, 3, 3)
    plt.imshow(mask_binary[0].cpu().detach().numpy(), cmap='gray')
    plt.title('Target Mask')
    plt.axis('off')
    plt.tight_layout()
    fig.suptitle(f"{name}", fontsize=30, fontweight='bold', y=0.07)
    plt.savefig(f'data_img/{model_type}/{name}.png', dpi=300)
    # plt.show()


    # print(f"各样本IoU: {iou.tolist()}")
    # print(f"平均IoU: {mean_iou:.4f}")
    return mean_iou

# def get_model_layers(model):
#         for name, module in model.named_modules():
#             if any(keyword in name.lower() for keyword in ['conv', 'features', 'classifier', 'fc', 'bottleneck', 'block']):
#                 print(name)

def calculate_heatmap_entropy(heatmaps, eps=1e-8):
    """
    计算Grad-CAM热力图的熵（支持批次输入）
    
    参数:
        heatmaps: 输入热力图张量，形状为 [B, H, W]（B为批次大小，H/W为热力图高/宽）
        eps: 防止log(0)和除0的小常数
    
    返回:
        entropies: 每个热力图的熵，形状为 [B,]
    """
    # 确保输入是张量且非负（Grad-CAM经过ReLU，理论上满足）
    if not isinstance(heatmaps, torch.Tensor):
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
    
    # 归一化：将每个热力图转为概率分布（总和为1）
    batch_size, h, w = heatmaps.shape # [B, H, W]
    sum_heat = heatmaps.sum(dim=(1, 2), keepdim=True)  # 每个样本的热力图总和 [B, 1, 1]
    p = heatmaps / (sum_heat + eps)  # 归一化概率 [B, H, W]，避免除0
    
    # 计算 p * log(p)，处理p=0的情况（此时p*log(p)为0）
    log_p = torch.log(p + eps)  # 加eps避免log(0)
    p_log_p = p * log_p  # [B, H, W]
    
    # 求和并取负，得到熵
    entropy = -torch.sum(p_log_p, dim=(1, 2))  # [B,]
    
    return entropy

def load_model(model_type, model_path, device, in_channels=3, num_classes=10):
    """通用的模型加载函数"""
    model = None

    if model_type == 'mobV3':
        from models.MobileNetV3 import MobileNetV3
        model = MobileNetV3(version='small', n_classes=num_classes, in_ch=in_channels)
    elif model_type == 'mobV2':
        from models.MobileNetV2 import MobileNetV2
        model = MobileNetV2(n_classes=num_classes, in_ch=in_channels)
    elif model_type == 'ghost':
        from models.GhostNet_paper import GhostNet
        model = GhostNet(num_classes=num_classes, in_ch=in_channels)
    elif model_type == 'shufV2':
        from models.ShuffleNetV2 import ShuffleNetV2
        model = ShuffleNetV2(n_classes=num_classes, in_ch=in_channels)
    elif model_type == 'resNet18':
        from models.ResNet18 import resnet18
        model = resnet18(n_classes=num_classes, in_ch=in_channels)
    # elif model_type == 'ghost':
    #     from models.GhostNet import GhostNet
    #     model = GhostNet(n_classes=num_classes, in_ch=in_channels)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(img_paths, in_channels):
    """将图像转换为模型输入格式（归一化、转张量）"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Grayscale(1)
    ])
    if in_channels == 3: img = Image.open(img_paths).convert('RGB')
    else: img = Image.open(img_paths).convert('L')
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

def main(
    data_root,
    model_type,
    model_path,
    in_channels,
    target_layer_name,
    index,
    device='cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"=== 实验配置 ===")
    print(f"设备：{device}")
    print(f"数据集：{data_root}")
    print(f"模型类型：{model_type}")
    print(f'目标层名称：{target_layer_name}')

    # 模型加载
    model = load_model(model_type, model_path, device, in_channels)

    # get_model_layers(model)
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model=model, target_layer=target_layer_name)

    # 计算IoU
    print(f"\n=== 开始计算形状一致性IoU ===")
    total_iou = 0.0

    img_paths = 'grad_cam_img_test.png'

    # 使用原始图像生成mask（二值化）
    mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 和图像Resize尺寸一致
            transforms.ToTensor()
        ])
    img, img_tensor = preprocess_image(img_paths, 1)
    original_array = np.array(img.convert('L'))  # 转为灰度图
    mask = (original_array > 0.5 * 255).astype(np.uint8)  # 二值化
    mask = Image.fromarray(mask * 255)  # 转为PIL图像
    
    # 对mask应用Resize，确保和图像尺寸一致（224x224）
    mask = mask_transform(mask).float()
    mask = (mask > 0.5).float()  # 确保是0/1二值图
    img_tensor = img_tensor.to(device)

    heatmap = grad_cam(img_tensor)

    mask = torch.tensor(mask).to(device)

    name = 'shallow' if i == 0 else 'middle' if i == 1 else 'deep'

    total_iou = calculate_iou(heatmap, mask, model_type, name= name)
    print(total_iou)

    # 清理钩子
    grad_cam.remove_hooks()

if __name__ == "__main__":
    MODEL_TYPES = ['ghost', 'mobV2', 'mobV3', 'resNet18', 'shufV2']
    DATA_ROOT = "C:/Users/Administrator/Desktop/datas/test_data_black"
    MODEL_PATH = "save_models/mobV2/MobileNetV2_1C_MD_42.pth"
    # MODEL_TYPE = MODEL_TYPES[3]
    # TARGET_LAYER_NAME = "last_conv.1"
    # MASK_THRESHOLD = 0.5
    # HEATMAP_THRESHOLD = 0.5
    # BATCH_SIZE = 10
    DEVICE = "cuda"

    MODEL_PATHS = [
        'save_models/ghost/GhostNet_paper_1C_MD_42.pth',
        'save_models/mobV2/MobileNetV2_1C_MD_42.pth',
        'save_models/mobV3/MobileNetV3_1C_MD_42.pth',
        'save_models/resNet18/resnet18_1C_MD_42.pth',
        'save_models/shufV2/ShuffleNetV2_1C_MD_42.pth'
    ]

    ghost_paper_model_layers = [
    # 'features.1.ghost1.primary_conv.2', # 浅层 低级特征
    # 'features.5.ghost2.primary_conv.2', # 中层 中级特征
    # 'features.10.ghost2.primary_conv.2' # 深层 高级特征
        'blocks.0.0.ghost1.primary_conv.0',
        'blocks.3.0.ghost1.primary_conv.0',
        'blocks.6.2.ghost1.primary_conv.0'
    ] 

    mobV3_model_layers = [
        'features.init_conv.2', # 浅层 低级特征
        'features.bottleneck_5.conv3.1', # 中层 中级特征
        'features.final_conv.2' # 深层 高级特征
    ]

    mobV2_model_layers = [
        # 'stem_conv.2', # 浅层 低级特征
        # 'last_conv', # 中层 中级特征
        # 'last_conv.1' #  深层 高级特征
        'stem_conv.1',
        'layers.4.layers.0',
        'last_conv.0'
    ]

    shufV2_model_layers = [
        'first_conv.1', # 浅层 低级特征
        'features.6.branch_main.7', # 中层 中级特征
        'conv_last.2' # 深层 高级特征
    ]

    resnet18_model_layers = [
        'layer1.1.conv2', # 浅层 低级特征
        'layer3.1.conv2', # 中层 中级特征
        'layer4.1.conv2' # 深层 高级特征
    ]

    for MODEL_PATH, MODEL_TYPE in zip(MODEL_PATHS, MODEL_TYPES):
        layers = []
        if MODEL_TYPE == 'ghost':
            layers = ghost_paper_model_layers
            # continue
        if MODEL_TYPE == 'mobV2':
            layers = mobV2_model_layers
            # continue
        if MODEL_TYPE == 'mobV3':
            layers = mobV3_model_layers
            # continue
        if MODEL_TYPE == 'resNet18':
            layers = resnet18_model_layers
            # continue
        if MODEL_TYPE == 'shufV2':
            layers = shufV2_model_layers

        for i, TARGET_LAYER_NAME in enumerate(layers):
            main(DATA_ROOT, MODEL_TYPE, MODEL_PATH, 1, TARGET_LAYER_NAME, i, DEVICE)