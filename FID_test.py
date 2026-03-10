import os
import torch
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import inception_v3


def get_inception_model(device):
    """加载预训练的InceptionV3，用钩子提取pool3层特征（avgpool输出）"""
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    model.to(device)
    
    # 定义钩子：用于捕获avgpool层（即pool3）的输出
    features = []
    def hook_fn(module, input, output):
        features.append(output.cpu().detach())  # 将输出保存到列表
    
    # 找到avgpool层并注册钩子（InceptionV3的pool3对应avgpool层）
    # PyTorch的InceptionV3中，avgpool是模型的直接属性
    model.avgpool.register_forward_hook(hook_fn)
    
    # 返回模型和特征列表（特征会在推理时被钩子填充）
    return model, features


def ensure_black_bg_white_fg(tensor):
    if tensor.mean() > 0.5:
        tensor = 1 - tensor
    return tensor


def tensor_transform(x):
    """处理通道数（C, H, W）"""
    if x.shape[0] == 1:
        return x.repeat_interleave(3, dim=0)
    elif x.shape[0] == 4:
        return x[:3, :, :]
    elif x.shape[0] == 3:
        return x
    else:
        raise ValueError(f"不支持的通道数: {x.shape[0]}")


def extract_features(dataloader, model, features_list, device, verbose=True):
    """提取特征（通过钩子捕获的pool3层输出）"""
    all_features = []
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            # 每次推理前清空特征列表
            features_list.clear()
            images = images.to(device)
            # 推理时，钩子会自动将avgpool输出存入features_list
            _ = model(images)  # 模型输出（logits）无用，忽略
            # 获取钩子捕获的特征（shape: [batch_size, 2048, 1, 1]）
            feat = features_list[0].view(images.size(0), -1).numpy()  # 展平为[bs, 2048]
            all_features.append(feat)
            
            if verbose and (i+1) % 10 == 0:
                print(f"已处理 {i+1}/{len(dataloader)} 批数据")
    
    return np.concatenate(all_features, axis=0)


def calculate_fid(features_real, features_fake, eps=1e-6):
    mu_real = np.mean(features_real, axis=0)
    mu_fake = np.mean(features_fake, axis=0)
    mean_diff_squared = np.sum((mu_real - mu_fake) **2)
    
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)
    
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])
    
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    trace_term = np.trace(sigma_real + sigma_fake - 2 * covmean)
    return mean_diff_squared + trace_term


def set_random_seed(seed=42):
    """设置随机种子确保可重复性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(train_dir, test_dir, batch_size=32, device='cuda', seed=42):
    set_random_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 预处理
    transform_train = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(tensor_transform),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Grayscale(num_output_channels= 1)
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Lambda(ensure_black_bg_white_fg),
        # transforms.Lambda(tensor_transform),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Grayscale(num_output_channels= 1)
    ])
    
    # 加载数据
    train_dataset = datasets.MNIST(
        root=train_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
    
    # num_workers = 4 if os.name != 'nt' else 0  # Windows系统禁用多进程
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练域样本数: {len(train_dataset)}, 测试域样本数: {len(test_dataset)}")
    
    # 加载模型并提取特征
    inception_model, features_hook = get_inception_model(device)
    
    print("提取训练域特征...")
    train_features = extract_features(train_loader, inception_model, features_hook, device)
    print("提取测试域特征...")
    test_features = extract_features(test_loader, inception_model, features_hook, device)
    
    # 计算FID
    fid_score = calculate_fid(train_features, test_features)
    print(f"训练域与测试域的FID分数: {fid_score:.4f}")


if __name__ == "__main__":
    TRAIN_DIR = "dataFolder/"  # MNIST数据目录
    TEST_DIR = "C:/Users/Administrator/Desktop/datas/test_data_black"  # 测试集目录（ImageFolder格式）
    main(train_dir=TRAIN_DIR, test_dir=TEST_DIR, batch_size=32, device='cuda')