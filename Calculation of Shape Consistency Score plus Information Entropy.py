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

class ImageFolderWithMask(datasets.ImageFolder):
    def __init__(self, root, transform=None, mask_threshold=0.5):
        super().__init__(root, transform)
        self.mask_threshold = mask_threshold
        # 单独定义mask的Resize变换（和图像保持一致的尺寸）
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 和图像Resize尺寸一致
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # 保存原始图像用于生成mask
        original_image = sample.copy()
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        # 使用原始图像生成mask（二值化）
        original_array = np.array(original_image.convert('L'))  # 转为灰度图
        mask = (original_array > self.mask_threshold * 255).astype(np.uint8)  # 二值化
        mask = Image.fromarray(mask * 255)  # 转为PIL图像
        
        # 对mask应用Resize，确保和图像尺寸一致（224x224）
        mask = self.mask_transform(mask).float()
        mask = (mask > 0.5).float()  # 确保是0/1二值图
        
        return sample, mask, target

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

def calculate_iou(heatmap, mask, threshold=0.5):
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
    if mask.ndim == 4:  # 若输入为[B, 1, H, W]，移除通道维度
        mask = mask.squeeze(1)  # 变为[B, H, W]
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
    # 3. 可视化（批次第一张图）
    # --------------------------
    # plt.figure(figsize=(12, 5))
    # # 原始热力图
    # plt.subplot(1, 3, 1)
    # plt.imshow(heatmap[0].cpu().detach().numpy(), cmap='jet')
    # plt.title('Grad-CAM Heatmap')
    # plt.axis('off')
    # # 二值化热力图
    # plt.subplot(1, 3, 2)
    # plt.imshow(heatmap_binary[0].cpu().detach().numpy(), cmap='gray')
    # plt.title('Binary Grad-CAM')
    # plt.axis('off')
    # # 目标mask
    # plt.subplot(1, 3, 3)
    # plt.imshow(mask_binary[0].cpu().detach().numpy(), cmap='gray')
    # plt.title('Target Mask')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('data_img/mobv2/深层.png', dpi=300)
    # plt.show()

    # --------------------------
    # 4. 计算IoU
    # --------------------------
    intersection = torch.sum(heatmap_binary * mask_binary, dim=(1, 2))  # 交集 [B,]
    union = torch.sum(heatmap_binary + mask_binary, dim=(1, 2)) - intersection  # 并集 [B,]
    iou = intersection / torch.clamp(union, min=1e-8)  # 避免除零
    mean_iou = torch.mean(iou).item()  # 批次平均IoU

    # print(f"各样本IoU: {iou.tolist()}")
    # print(f"平均IoU: {mean_iou:.4f}")
    return mean_iou

# def get_model_layers(model):
#         for name, module in model.named_modules():
#             if any(keyword in name.lower() for keyword in ['conv', 'features', 'classifier', 'fc', 'bottleneck', 'block']):
#                 print(name)


def compute_cov_term(heatmaps, masks, encoder= None, eps= 1e-8):
    """
    计算 $\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{\frac{1}{2}}$
    
    参数:
        real_features: 真实样本的特征矩阵，形状为 [N, D]（N为样本数，D为特征维度）
        gen_features: 生成样本的特征矩阵，形状为 [M, D]（M为样本数，D为特征维度）
    
    返回:
        协方差项的矩阵，形状为 [D, D]
    """
    if isinstance(heatmaps, np.ndarray):
        # 若输入是numpy，转为tensor并移至mask所在设备
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32, device=masks.device)
    else:
        # 确保heatmap与mask设备一致
        heatmaps = heatmaps.to(masks.device).float()
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    assert heatmaps.shape == masks.shape, "形状不匹配"
    batch_size = heatmaps.shape[0]
    
    # -------------------------- 新增：降维步骤 --------------------------
    if encoder is not None:
        # 方案1：用传统形状特征提取（无需训练，encoder设为"traditional"）
        # if encoder == "traditional":
        #     heatmaps_lowdim = extract_shape_features(heatmaps)  # [B, D']，D'=20/32
        #     masks_lowdim = extract_shape_features(masks)
        # 方案2：用小型CNN提取（encoder为定义好的HeatmapEncoder实例）
        # else:
        with torch.no_grad():  # 无训练时禁用梯度
            heatmaps_lowdim = encoder(heatmaps)  # [B, 128]
            masks_lowdim = encoder(masks)
        # 校验维度：D' < batch_size（保证协方差矩阵非奇异）
        # assert heatmaps_lowdim.shape[1] < batch_size, f"低维特征维度{D'}需小于批次大小{batch_size}"
        features_r = heatmaps_lowdim
        features_g = masks_lowdim
    else:
        # 兼容原有逻辑（无降维）
        features_r = heatmaps.reshape(batch_size, -1)
        features_g = masks.reshape(batch_size, -1)
    
    # -------------------------- 原有协方差计算逻辑（完全不变） --------------------------
    # 中心化
    features_r_centered = features_r - features_r.mean(dim=0, keepdim=True)
    features_g_centered = features_g - features_g.mean(dim=0, keepdim=True)
    
    # 协方差矩阵（无偏估计）
    sigma_r = (features_r_centered.T @ features_r_centered) / (batch_size - 1 + eps)
    sigma_g = (features_g_centered.T @ features_g_centered) / (batch_size - 1 + eps)
    
    
    # 添加正则化避免奇异矩阵
    # torch.eye() 是 PyTorch 中用于创建单位矩阵（Identity Matrix） 的核心函数，
    # 生成一个「对角线上元素为 1，其余元素为 0」的二维张量（默认是方阵，也可指定非方阵）。
    # 单位矩阵在线性代数、模型初始化（如权重初始化）、掩码生成等场景中常用。
    sigma_r = sigma_r + eps * torch.eye(sigma_r.size(0), device=sigma_r.device)
    sigma_g = sigma_g + eps * torch.eye(sigma_g.size(0), device=sigma_g.device)
    
    # 计算矩阵平方根 (Σ_r Σ_g)^{1/2}
    sigma_product = sigma_r @ sigma_g
    
    # 使用更稳定的矩阵平方根计算
    try:
        # 方法1: SVD分解
        U, S, Vh = torch.linalg.svd(sigma_product)
        S_sqrt = torch.sqrt(torch.clamp(S, min=eps))
        sqrt_sigma_product = U @ torch.diag(S_sqrt) @ Vh
    except:
        # 方法2: 特征值分解（备选）
        L, V = torch.linalg.eig(sigma_product)
        L_sqrt = torch.sqrt(torch.clamp(L.real, min=eps))
        # V.T.conj().real 完全解析（PyTorch）
        # 这个链式操作是 PyTorch 中对矩阵 / 张量的转置 + 共轭 + 实部提取组合操作，
        # 核心用于处理复数张量（或兼容实数张量），
        # 在 SVD 分解、矩阵共轭转置（Hermitian 转置）、复数矩阵重构等场景中高频出现（结合你之前的 SVD 相关问题，V 大概率是 torch.svd() 返回的右奇异向量）。
        sqrt_sigma_product = V @ torch.diag(L_sqrt) @ V.T.conj().real
    
    # 计算协方差项
    cov_term = sigma_r + sigma_g - 2 * sqrt_sigma_product
    
    # 返回Frobenius范数作为标量指标
    return torch.norm(cov_term).item()

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

def main(
    data_root,
    model_type,
    model_path,
    in_channels,
    target_layer_name,
    mask_threshold=0.5,
    heatmap_threshold=0.5,
    batch_size=1,
    seed= None,
    device='cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"=== 实验配置 ===")
    print(f"设备：{device}")
    print(f"数据集：{data_root}")
    print(f"模型类型：{model_type}")
    print(f'目标层名称：{target_layer_name}')


    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Grayscale(1) if in_channels == 1 else transforms.Grayscale(3)
    ])
    
    # 使用自定义数据集
    dataset = ImageFolderWithMask(
        root=data_root,
        transform=transform,
        mask_threshold=mask_threshold
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle= True)
    print(f"图像样本数：{len(dataset)}")

    # 模型加载
    model = load_model(model_type, model_path, device, in_channels)

    # get_model_layers(model)
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model=model, target_layer=target_layer_name)

    # 计算IoU
    print(f"\n=== 开始计算形状一致性IoU ===")
    total_iou = 0.0
    processed_batches = 0
    total_entropy = 0.0
    batch_avg_entropy = 0.0


    # x, mask, y = next(iter(dataloader))
    # plt.imshow(mask[0][0], cmap= 'gray')
    # plt.show()
    encoder = HeatmapEncoder(output_dim=128).to(device)
    encoder.eval()
    
    for batch_idx, (images, masks, labels) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        # plt.imshow(images[0][0].cpu().detach().numpy(), cmap= 'gray')
        # plt.show()

        # try:
        # 生成热力图
        heatmaps = grad_cam(images, class_idx=labels)
        
        # 计算IoU
        batch_iou = calculate_iou(heatmaps, masks, threshold=heatmap_threshold)
        total_iou += batch_iou
        processed_batches += 1

        # 修复熵计算：先求批次平均熵（标量）
        entropy_tensor = calculate_heatmap_entropy(heatmaps)  # [B,]
        batch_avg_entropy = entropy_tensor.mean().item()  # 单批次平均熵（标量）
        total_entropy += batch_avg_entropy

        # cov_term = compute_cov_term(heatmaps, masks, encoder= encoder)
        # print(cov_term)
    
        progress = batch_idx + 1
        if progress % 10 == 0 or progress == len(dataloader):
            print(f"批次 {progress:3d}/{len(dataloader):3d} | 批次IoU：{batch_iou:.4f} | 批次平均熵：{batch_avg_entropy:.4f}")

            # if progress % 50 == 0 or progress == len(dataloader):
            #     # 随机选一个样本（0~batch_size-1）
            #     idx = np.random.randint(0, batch_size)
            #     img = images[idx].cpu().permute(1,2,0)  # [C,H,W]→[H,W,C]
            #     img = img * 0.3081 + 0.1307  # 反归一化
            #     img = img.clamp(0,1)  # 限制像素范围
                
            #     mask = masks[idx].cpu().squeeze()  # [1,H,W]→[H,W]
            #     heatmap = heatmaps[idx] # [H,W]
            #     overlay = img.cpu().detach().numpy() * 0.6 + plt.cm.jet(heatmap)[:, :, :3] * 0.4  # 叠加图
                
                # plt.rcParams['font.family'] = ["SimHei"] 
                # plt.rcParams['axes.unicode_minus'] = False

                # # 绘图
                # fig, axs = plt.subplots(1,4,figsize=(16,6))
                # axs[0].imshow(img.cpu().detach().numpy())
                # axs[0].set_title("原图")
                # axs[0].axis('off')
                
                # axs[1].imshow(mask, cmap='gray')
                # axs[1].set_title("形状掩码")
                # axs[1].axis('off')
                
                # axs[2].imshow(heatmap, cmap='jet')
                # axs[2].set_title(f"热力图（熵：{entropy_tensor[idx]:.4f}）")
                # axs[2].axis('off')
                
                # axs[3].imshow(overlay)
                # axs[3].set_title("热力图叠加原图")
                # axs[3].axis('off')
                
                # plt.tight_layout()
                # plt.show()
        
        # except Exception as e:
        #     print(f"批次 {batch_idx+1} 处理失败: {e}")
        #     return 0, 0
        
        # finally:
        #     # 清理内存
        #     del images, masks, labels
        #     if 'heatmaps' in locals():
        #         del heatmaps
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()

    if processed_batches > 0:
        avg_iou = total_iou / processed_batches
        avg_entropy = total_entropy / processed_batches  # 全局平均熵（标量）
        print(f"\n=== 评估结果 ===")
        print(f"处理批次: {processed_batches}/{len(dataloader)}")
        print(f"处理样本总数: {processed_batches * batch_size}")  # 新增：显示总样本数
        print(f"形状一致性平均IoU：{avg_iou:.4f}")
        print(f"热力图全局平均熵：{avg_entropy:.4f}")  # 标量输出
        
        # 结果解释（不变）
        if avg_iou < 0.1:
            interpretation = "模型主要依赖非形状特征"
        elif avg_iou < 0.3:
            interpretation = "模型对形状特征有一定依赖"
        elif avg_iou < 0.5:
            interpretation = "模型较好地平衡了形状特征和其他特征"
        else:
            interpretation = "模型主要依赖形状特征"

        # 新增：熵的结果解释
        entropy_interpretation = ""
        max_possible_entropy = np.log(224 * 224)  # 理论最大熵（均匀分布）
        if avg_entropy < max_possible_entropy * 0.3:
            entropy_interpretation = "热力图关注点极集中，模型聚焦关键区域"
        elif avg_entropy < max_possible_entropy * 0.5:
            entropy_interpretation = "热力图关注点较集中，模型对目标区域识别明确"
        elif avg_entropy < max_possible_entropy * 0.7:
            entropy_interpretation = "热力图关注点中等分散，模型兼顾局部与全局特征"
        else:
            entropy_interpretation = "热力图关注点极分散，模型依赖全局特征而非局部关键区域"
        
        print(f"模型行为分析(IoU): {interpretation}")
        print(f"模型行为分析(熵): {entropy_interpretation}")
        # with open(f'data_csv/result.csv', 'a+') as f:
        #     f.write(f"{seed},{model_type},{in_channels},{target_layer_name},{avg_iou},{avg_entropy}\n")
        # return avg_iou, avg_entropy
    else:
        print("没有成功处理任何批次")

    # 清理钩子
    grad_cam.remove_hooks()

if __name__ == "__main__":
    MODEL_TYPES = ['mobV3', 'ghost', 'resNet18', 'mobV2', 'shufV2']
    DATA_ROOT = "C:/Users/Administrator/Desktop/datas/test_data_black"
    MODEL_PATH = "save_models/mobV2/MobileNetV2_1C_MD_42.pth"
    MODEL_TYPE = MODEL_TYPES[3]
    TARGET_LAYER_NAME = "last_conv.1"
    MASK_THRESHOLD = 0.5
    HEATMAP_THRESHOLD = 0.5
    BATCH_SIZE = 10
    DEVICE = "cuda"

    main(
        data_root=DATA_ROOT,
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        in_channels= 1,
        target_layer_name=TARGET_LAYER_NAME,
        mask_threshold=MASK_THRESHOLD,
        heatmap_threshold=HEATMAP_THRESHOLD,
        batch_size=BATCH_SIZE,
        seed= 42,
        device=DEVICE
    )
