import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import Tensor
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from models.ShuffleNetV2 import ShuffleNetV2 as shuf
from models.GhostNet import GhostNet
from models.ShuffleNetV2_exp_channel_shuffle import ShuffleNetV2 as shuf_exp
from models.MobileNetV2 import MobileNetV2
from models.MobileNetV3 import MobileNetV3

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # 注册前向和反向钩子，以便在计算梯度时记录特征图
        self.hook_handles = []

        def forward_hook(module, input, output):
            # 记录特征图
            self.feature_maps = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            # 记录梯度
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == target_layer:
                # 记录目标层的特征图和梯度
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                break
    
    def __call__(self, x, class_idx= None):
        """
        生成GradCAM热力图
        x: 输入图像张量（shape: [1, C, H, W]）
        class_idx: 目标类别索引（None则用模型预测的最高概率类别）
        """
        self.model.eval()
        # requires_grad_() 方法用于将张量的requires_grad属性设置为True，表示该张量需要计算梯度。
        x.requires_grad_()

        output = self.model(x)
        # print(output.shape)
        if class_idx is None:
            # 获取预测概率最高的类别索引
            class_idx = torch.argmax(output, dim= 1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # 计算权重
        weights = F.adaptive_avg_pool2d(self.gradients, 1) # [1, C, 1, 1]
        weights = weights.squeeze(-1).squeeze(-1) # [1, C]

        # 计算CAM
        cam = torch.sum(weights[:, :, None, None] * self.feature_maps, dim= 1) # [1, H, W]
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0), # [1, 1, H, W]
            size= (x.shape[2], x.shape[3]),
            mode= 'bilinear',
            align_corners= False
        ).squeeze() # [H, W]

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def preprocess_image(img_paths, in_channels):
    """将图像转换为模型输入格式（归一化、转张量）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
        transforms.Resize((224, 224))
    ])
    if in_channels == 3: img = Image.open(img_paths).convert('RGB')
    else: img = Image.open(img_paths).convert('L')
    img_tensor = transform(img).unsqueeze(0)
	# transform = transforms.Compose([
	# 	transforms.Resize((224, 224)),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize((0.1307,), (0.3081,)),
	# ])

	# if isinstance(img, Tensor):
	# 	img_tensor = img.unsqueeze(0) # [1, 3, 224, 224]
	# else:
	# 	img = Image.open(img).convert('RGB')
	# 	img_tensor = transform(img).unsqueeze(0) # [1, 3, 224, 224]

    # img = Image.open(img_paths).convert('RGB')
    # img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

def visualize_cam(img, cam, name, in_channels, epochs):
    """叠加原图与热力图"""
    plt.figure(figsize=(3, 3))
    
    # 显示原图
    # plt.subplot(121)
    # plt.imshow(img[0])
    # plt.title('Original Image')
    # plt.axis('off')
    
    # 显示叠加热力图
    # plt.subplot(111)
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)  # 热力图叠加（alpha控制透明度）
    plt.title(name.replace('.', '_'), fontsize= 10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'data_imgs/grad_cam/ShuffleNetV2/E{epochs}/{in_channels}C/{name}.png',
    pad_inches=0.01         # 裁剪后保留的最小边距（可选）
    )
    # plt.show()

def ensure_black_bg_white_fg(tensor):
    # 计算图像的平均像素值，如果平均像素值大于0.5，我们认为当前是白底黑字，需要反转
    if tensor.mean() > 0.5:
        tensor = 1 - tensor
    return tensor

if __name__ == '__main__':

    in_channels = 1
    epochs = 10

    # 加载预训练模型（以ResNet50为例）
    model = shuf(in_ch= in_channels)
    model.load_state_dict(torch.load(f'save_models/E{epochs}/shuf_{in_channels}C_MD_1_E{epochs}.mdl'))
    model.eval()

    img_paths = 'grad_cam_img_test.png'
    
     # 加载并预处理图像（替换为你的图像路径）
    
    # MNIST_db = torchvision.datasets.MNIST(root= 'dataFolder/', train=True, transform= transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307, ), (0.3081, )),
    #     transforms.Resize((224, 224))
    # ]))

    # MNIST_Loader = DataLoader(MNIST_db, batch_size= 32, shuffle= True)

    # num_db =  torchvision.datasets.ImageFolder(root= 'dataFolder/num', transform= transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Lambda(ensure_black_bg_white_fg),
    #     transforms.Normalize((0.1307,), (0.3081,)),
    #     # transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
    #     transforms.Grayscale(1)
    # ]))

    # num_loader = DataLoader(dataset= num_db, batch_size= 100, shuffle= True)

    # x, y = next(iter(num_loader))

    # print(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"卷积层名称: {name}")

            # if name.count('primary_conv') == 0: continue
            if name.count('branch_main.5') == 0 and name.count('branch_proj.2') == 0: continue
    
            # target_layer = 'features.1.ghost1.primary_conv.0'  # 需根据模型结构调整（可打印模型查看）
            # s = name.split('.')[3]
            # if s == 'primary_conv':
            
            # 初始化GradCAM
            grad_cam = GradCAM(model, name)

            img, img_tensor = preprocess_image(img_paths, in_channels)
            
            # 生成GradCAM热力图
            cam = grad_cam(img_tensor)
            
            # 可视化
            visualize_cam(img, cam, name, in_channels, epochs)
            
            # 移除钩子
            grad_cam.remove_hooks()