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
import math
from scipy import linalg

class ImageFolderWithMask(datasets.ImageFolder):
    def __init__(self, root, transform=None, mask_threshold=0.5):
        super().__init__(root, transform)
        self.mask_threshold = mask_threshold
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        original_image = sample.copy()
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        original_array = np.array(original_image.convert('L'))
        mask = (original_array > self.mask_threshold * 255).astype(np.uint8)
        mask = Image.fromarray(mask * 255)
        
        mask = self.mask_transform(mask).float()
        mask = (mask > 0.5).float()
        
        return sample, mask, target

class MultiLayerGradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    def _forward_hook(self, layer_name):
        def hook(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook
    
    def _backward_hook(self, layer_name):
        def hook(module, grad_in, grad_out):
            self.gradients[layer_name] = grad_out[0].detach()
        return hook
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(self._forward_hook(name)))
                self.hooks.append(module.register_backward_hook(self._backward_hook(name)))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def generate_heatmaps(self, x, class_idx=None):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)
        x.requires_grad_(True)
        
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1)
        output.backward(gradient=one_hot)
        
        heatmaps = {}
        batch_size = x.shape[0]
        
        for layer_name in self.target_layers:
            if layer_name not in self.activations or layer_name not in self.gradients:
                continue
                
            activations = self.activations[layer_name]  # [B, C, H, W]
            gradients = self.gradients[layer_name]      # [B, C, H, W]
            
            # 计算权重 [B, C, 1, 1]
            weights = F.adaptive_avg_pool2d(gradients, 1)
            
            # 生成CAM [B, H, W]
            cam = torch.sum(weights * activations, dim=1)
            cam = F.relu(cam)
            
            # 上采样到输入尺寸
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(x.shape[2], x.shape[3]),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # 归一化
            cam_min = cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            cam_max = cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            heatmaps[layer_name] = cam.detach().cpu()
        
        return heatmaps, output

def calculate_iou(heatmaps, masks, layer_name):
    """计算指定层级的IoU"""
    if layer_name not in heatmaps:
        return 0.0
        
    heatmap = heatmaps[layer_name]
    device = masks.device
    
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.tensor(heatmap, dtype=torch.float32, device=device)
    else:
        heatmap = heatmap.to(device).float()
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    mask_binary = masks.float()

    assert heatmap.shape == mask_binary.shape, f"形状不匹配：{heatmap.shape} vs {mask_binary.shape}"
    
    batch_size = heatmap.shape[0]
    heatmap_binary = []
    
    for idx in range(batch_size):
        img = heatmap[idx].cpu().detach().numpy()
        img_min, img_max = img.min(), img.max()
        if img_max != img_min:
            img_norm = (img - img_min) / (img_max - img_min) * 255
        else:
            img_norm = np.zeros_like(img)
        img_uint8 = img_norm.astype(np.uint8)
        
        _, binary_np = cv.threshold(img_uint8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        binary_tensor = torch.tensor(binary_np / 255, dtype=torch.float32, device=device)
        heatmap_binary.append(binary_tensor)
    
    heatmap_binary = torch.stack(heatmap_binary, dim=0)
    
    intersection = torch.sum(heatmap_binary * mask_binary, dim=(1, 2))
    union = torch.sum(heatmap_binary + mask_binary, dim=(1, 2)) - intersection
    iou = intersection / torch.clamp(union, min=1e-8)
    
    return torch.mean(iou).item()

def calculate_heatmap_entropy(heatmaps, layer_name, eps=1e-8):
    """计算指定层级的热力图熵"""
    if layer_name not in heatmaps:
        return 0.0
        
    heatmap = heatmaps[layer_name]
    
    if not isinstance(heatmap, torch.Tensor):
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
    
    batch_size, h, w = heatmap.shape
    sum_heat = heatmap.sum(dim=(1, 2), keepdim=True)
    p = heatmap / (sum_heat + eps)
    
    log_p = torch.log(p + eps)
    p_log_p = p * log_p
    entropy = -torch.sum(p_log_p, dim=(1, 2))
    
    return entropy.mean().item()

def compute_fid(heatmaps, masks, layer_name):
    """计算FID距离"""
    if layer_name not in heatmaps:
        return float('inf')
        
    heatmap = heatmaps[layer_name]
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # 提取统计特征
    def extract_features(arr):
        features = []
        for i in range(arr.shape[0]):
            img = arr[i]
            features.append([
                img.mean(), img.std(), img.max(), img.min(),
                np.percentile(img, 25), np.percentile(img, 50),
                np.percentile(img, 75), img.var()
            ])
        return np.array(features)
    
    heatmap_features = extract_features(heatmap)
    mask_features = extract_features(masks)
    
    # 计算FID
    mu1, sigma1 = np.mean(heatmap_features, axis=0), np.cov(heatmap_features, rowvar=False)
    mu2, sigma2 = np.mean(mask_features, axis=0), np.cov(mask_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_covariance_alignment(heatmaps, masks, layer_name):
    """计算协方差对齐（改进版）"""
    if layer_name not in heatmaps:
        return 0.0
        
    heatmap = heatmaps[layer_name]
    device = masks.device
    
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.tensor(heatmap, dtype=torch.float32, device=device)
    else:
        heatmap = heatmap.to(device).float()
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    batch_size, h, w = heatmap.shape
    
    # 计算批次相关系数
    correlations = []
    for i in range(batch_size):
        heat_flat = heatmap[i].flatten()
        mask_flat = masks[i].flatten()
        
        # 计算皮尔逊相关系数
        correlation = torch.corrcoef(torch.stack([heat_flat, mask_flat]))[0, 1]
        if not torch.isnan(correlation):
            correlations.append(correlation)
    
    if len(correlations) == 0:
        return 0.0
    
    alignment = torch.mean(torch.stack(correlations)).item()
    return alignment

def load_model(model_type, model_path, device, in_channels=3, num_classes=10):
    """通用的模型加载函数"""
    model = None

    if model_type == 'mobV3':
        from models.MobileNetV3 import MobileNetV3
        model = MobileNetV3(version='small', n_classes=num_classes, in_ch=in_channels)
    elif model_type == 'mobV2':
        from models.MobileNetV2 import MobileNetV2
        model = MobileNetV2(n_classes=num_classes, in_ch=in_channels)
    elif model_type == 'ghost_paper':
        from models.GhostNet_paper import GhostNet
        model = GhostNet(num_classes=num_classes, in_ch=in_channels)
    elif model_type == 'shufV2':
        from models.ShuffleNetV2 import ShuffleNetV2
        model = ShuffleNetV2(n_classes=num_classes, in_ch=in_channels)
    elif model_type == 'resNet18':
        from models.ResNet18 import resnet18
        model = resnet18(n_classes=num_classes, in_ch=in_channels)
    elif model_type == 'ghost':
        from models.GhostNet import GhostNet
        model = GhostNet(n_classes=num_classes, in_ch=in_channels)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model

def analyze_network_bias(model, dataloader, layer_mapping, device):
    """综合分析网络各层级的归纳偏置"""
    cam = MultiLayerGradCAM(model, list(layer_mapping.values()))
    cam.register_hooks()
    
    results = {
        'shallow': {'iou': [], 'entropy': [], 'fid': [], 'cov_align': []},
        'middle': {'iou': [], 'entropy': [], 'fid': [], 'cov_align': []},
        'deep': {'iou': [], 'entropy': [], 'fid': [], 'cov_align': []}
    }
    
    all_heatmaps = {'shallow': [], 'middle': [], 'deep': []}
    all_masks = []
    
    for batch_idx, (images, masks, labels) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        
        # 生成多层级热力图
        heatmaps, _ = cam.generate_heatmaps(images, class_idx=labels)
        
        # 计算各层级指标
        for level, layer_name in layer_mapping.items():
            if layer_name in heatmaps:
                iou = calculate_iou(heatmaps, masks, layer_name)
                entropy = calculate_heatmap_entropy(heatmaps, layer_name)
                cov_align = compute_covariance_alignment(heatmaps, masks, layer_name)
                
                results[level]['iou'].append(iou)
                results[level]['entropy'].append(entropy)
                results[level]['cov_align'].append(cov_align)
                all_heatmaps[level].append(heatmaps[layer_name].numpy())
        
        all_masks.append(masks.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f'已处理批次 {batch_idx + 1}')
        
        # 清理内存
        del images, masks, labels, heatmaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算FID
    all_masks = np.vstack([m.squeeze(1) if m.ndim == 4 else m for m in all_masks])
    for level in ['shallow', 'middle', 'deep']:
        if all_heatmaps[level]:
            heatmap_features = np.vstack(all_heatmaps[level])
            fid_score = compute_fid(heatmap_features, all_masks, level)
            results[level]['fid'].append(fid_score)
    
    # 计算最终结果
    final_results = {}
    for level in ['shallow', 'middle', 'deep']:
        if results[level]['iou']:
            final_results[level] = {
                'mean_iou': np.mean(results[level]['iou']),
                'mean_entropy': np.mean(results[level]['entropy']),
                'fid': results[level]['fid'][0] if results[level]['fid'] else float('inf'),
                'mean_cov_align': np.mean(results[level]['cov_align'])
            }
        else:
            final_results[level] = {
                'mean_iou': 0.0,
                'mean_entropy': 0.0,
                'fid': float('inf'),
                'mean_cov_align': 0.0
            }
    
    cam.remove_hooks()
    return final_results

def main(
    data_root,
    model_type,
    model_path,
    in_channels,
    layer_mapping,  # 改为层级映射
    mask_threshold=0.5,
    batch_size=10,
    seed=None,
    device='cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"=== 网络归纳偏置分析 ===")
    print(f"设备：{device}")
    print(f"数据集：{data_root}")
    print(f"模型类型：{model_type}")
    print(f"层级映射：{layer_mapping}")

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Grayscale(1) if in_channels == 1 else transforms.Grayscale(3)
    ])
    
    dataset = ImageFolderWithMask(
        root=data_root,
        transform=transform,
        mask_threshold=mask_threshold
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"图像样本数：{len(dataset)}")

    # 模型加载
    model = load_model(model_type, model_path, device, in_channels)

    # 综合分析
    print(f"\n=== 开始综合分析 ===")
    results = analyze_network_bias(model, dataloader, layer_mapping, device)
    
    # 输出结果
    print(f"\n=== 归纳偏置分析结果 ===")
    for level in ['shallow', 'middle', 'deep']:
        if level in results:
            data = results[level]
            print(f"\n{level.upper()}层:")
            print(f"  形状一致性IoU: {data['mean_iou']:.4f}")
            print(f"  关注集中度熵: {data['mean_entropy']:.4f}")
            print(f"  特征分布FID: {data['fid']:.4f}")
            print(f"  协方差对齐: {data['mean_cov_align']:.4f}")
            
            # 偏置类型判断
            if data['mean_iou'] > 0.3 and data['mean_cov_align'] > 0.1:
                bias_type = "强形状偏置"
            elif data['mean_entropy'] < 2.0:
                bias_type = "局部特征偏置"
            elif data['fid'] < 1.0:
                bias_type = "分布一致性偏置"
            else:
                bias_type = "混合偏置"
            
            print(f"  归纳偏置类型: {bias_type}")
    
    # 保存结果
    with open('data_csv/network_bias_results.csv', 'a+') as f:
        for level in ['shallow', 'middle', 'deep']:
            if level in results:
                data = results[level]
                f.write(f"{seed},{model_type},{in_channels},{level},"
                       f"{data['mean_iou']:.4f},{data['mean_entropy']:.4f},"
                       f"{data['fid']:.4f},{data['mean_cov_align']:.4f}\n")
    
    return results

if __name__ == "__main__":
    # 定义不同网络的层级映射
    MOBILENETV2_LAYERS = {
        'shallow': 'features.0.2',   # 第一个卷积层
        'middle': 'features.6.0',    # 中间层
        'deep': 'features.12.0'      # 深层
    }
    
    RESNET18_LAYERS = {
        'shallow': 'layer1.1.conv2',
        'middle': 'layer3.1.conv2', 
        'deep': 'layer4.1.conv2'
    }
    
    MODEL_TYPES = ['mobV3', 'ghost', 'resNet18', 'mobV2', 'shufV2']
    DATA_ROOT = "C:/Users/Administrator/Desktop/datas/test_data_black"
    MODEL_PATH = "save_models/resNet18/resnet18_1C_MD_42.pth"
    MODEL_TYPE = MODEL_TYPES[2]  # mobV2
    
    main(
        data_root=DATA_ROOT,
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        in_channels=1,
        layer_mapping=RESNET18_LAYERS,  # 使用层级映射
        mask_threshold=0.5,
        batch_size=10,
        seed=42,
        device="cuda"
    )