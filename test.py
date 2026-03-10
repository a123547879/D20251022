import os
import tkinter as tk
import torch
import torch.nn as nn
import torchvision
from GradCAM import GradCAM
from PIL import Image, ImageTk
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from models.GhostNet import GhostNet
from models.MobileNetV2 import MobileNetV2
from models.MobileNetV3 import MobileNetV3
from models.ShuffleNetV2 import ShuffleNetV2
from tkinter import ttk, filedialog, messagebox
from 形状一致性分数加信息熵计算 import *


class DataListApp:
    def __init__(self, root):
        """初始化应用程序"""
        self.root = root
        self.root.title("数据列表管理器")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # 确保中文显示正常
        self.style = ttk.Style()
        
        # 存储数据列表
        self.model = None
        self.orignal_dataset = None
        self.target_dataset = None
        self.target_index= None
        self.orginal_index = None
        self.test_img = None
        self.orignal_img = None
        self.model_type = None
        self.model_path = None
        self.in_channels = 0
        self.model_layers = []
        self.model_avg_ious = []
        self.model_avg_entropys = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建UI
        self.create_widgets()

    def get_model_layers(self):
        if not self.model:
            messagebox.showinfo("提示", "请先选择模型文件！")
            return
        for name, module in self.model.named_modules():
            # if any(keyword in name.lower() for keyword in ['conv', 'features', 'classifier', 'fc', 'bottleneck', 'block']):
            self.model_layers.append(name)
        # for layer_name in self.model_layers:
        #     avg_iou, avg_entropy = main(self.target_dataset, self.model_type, self.model_path, self.in_channels, layer_name, batch_size= 10)
        #     self.model_avg_ious.append(avg_iou)
        #     self.model_avg_entropys.append(avg_entropy)
         
        # 更新列表框
        self.update_listbox()
        
    def create_widgets(self):
        """创建界面组件"""
        # 创建顶部操作栏
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        # 导入模型文件按钮
        ttk.Button(
            top_frame, 
            text="导入模型文件", 
            command=self.import_model_file
        ).pack(side=tk.LEFT, padx=5)
        
        # # 导入数据集路径按钮
        # ttk.Button(
        #     top_frame, 
        #     text="导入原数据集路径", 
        #     command=self.import_dataset_path
        # ).pack(side=tk.LEFT, padx=5)

        # # 导入目标数据集路径按钮
        # ttk.Button(
        #     top_frame, 
        #     text="导入目标数据集路径", 
        #     command=self.import_dataset_path
        # ).pack(side=tk.LEFT, padx=5)

        # 导入测试图片
        ttk.Button(
            top_frame, 
            text="导入测试图片", 
            command=self.import_img_file
        ).pack(side=tk.LEFT, padx=5)
        
        # 清空列表按钮
        ttk.Button(
            top_frame, 
            text="清空列表", 
            command=self.clear_list
        ).pack(side=tk.RIGHT, padx=5)
        
        # 创建列表区域
        list_frame = ttk.Frame(self.root, padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 列表标题（修正了这里的参数错误）
        ttk.Label(list_frame, text= "数据列表", font=("SimHei", 12)).pack(anchor=tk.W, pady=5)  # 将mb=5改为pady=5
        
        # 创建列表框
        self.listbox_frame = ttk.Frame(list_frame)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(self.listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 列表框
        self.listbox = tk.Listbox(
            self.listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("SimHei", 10),
            selectbackground="#a6a6a6",
            height=15
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        # 绑定列表项点击事件
        self.listbox.bind('<Double-1>', self.on_item_click)

        # 数据集区域
        self.data_sets = ['MNIST', 'SVHN', 'Other']
        # 滚动条
        scrollbar2 = ttk.Scrollbar(self.listbox_frame)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox2 = tk.Listbox(
            self.listbox_frame,
            yscrollcommand= scrollbar2.set,
            font=("SimHei", 10),
            selectbackground="#a6a6a6",
            height=15
        )
        for i, dataset in enumerate(self.data_sets):
            self.listbox2.insert(i, dataset)
        self.listbox2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.config(command=self.listbox2.yview)
        self.listbox2.bind('<Double-1>', lambda event: self.on_item_click2())
        
        # 状态区域
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_listbox(self):
        """更新列表框内容"""
        # 清空列表框
        self.listbox.delete(0, tk.END)
        for item in self.model_layers:
            self.listbox.insert(tk.END, item)

    def update_listbox2(self):
        """更新列表框2内容"""
        # 清空列表框
        self.listbox2.delete(0, tk.END)
        for i, item in enumerate(self.data_sets):
            print(self.target_index, self.orginal_index)
            if i == self.target_index:
                self.listbox2.insert(tk.END, f'{item} target')
            elif i == self.orginal_index:
                self.listbox2.insert(tk.END, f'{item} orignal')
            else:
                self.listbox2.insert(tk.END, item)

    def import_model_file(self):
        if self.target_dataset is None:
            messagebox.showerror("错误", "请先选择目标数据集！")
            return

        """导入模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[
                ("模型文件", "*.h5 *.pkl *.pt *.pth *.model *.mdl"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.listbox.delete(0, tk.END)
            self.model_layers = []
            # 提取文件名作为显示名称
            file_name = os.path.basename(file_path)
            display_text = f"[模型] {file_name}"

            s = file_name.split('_')
            typename = s[0]
            self.in_channels = int(s[1][0])
            if typename == 'GhostNet':
                self.model_type = 'ghost'
                self.model = GhostNet(in_ch= self.in_channels)
            elif typename == 'MobileNetV3':
                self.model_type = 'mobV3'
                self.model = MobileNetV3(version='small', n_classes=10, in_ch= self.in_channels)
            elif typename == 'MobileNetV2':
                self.model_type = 'mobV2'
                self.model = MobileNetV2(in_ch= self.in_channels, n_classes=10)
            elif typename == 'ShuffleNetV2':
                self.model_type = 'shufV2'
                self.model = ShuffleNetV2(in_ch= self.in_channels)

            # 添加到列表
            # self.model = GhostNet(in_ch= 3)
            self.model_path = file_path
            self.model.load_state_dict(torch.load(file_path))
            self.model.to(self.device)
            self.model.eval()
            self.get_model_layers()


            
            # 更新状态
            self.status_var.set(f"已导入模型文件: {file_name}")

    def on_item_click2(self, event=None):
        is_target = messagebox.askquestion("源数据集", "目标数据集")  
          
        """列表项点击事件处理"""
        # 获取选中的索引
        index = self.listbox2.curselection()
        if not index:
            return
            
        index = index[0]
        if is_target == 'yes': self.target_index = index
        else: self.orginal_index = index
        selected_item = self.listbox2.get(index)
        if selected_item == 'Other':
            dir_path = filedialog.askdirectory(title="选择数据集文件夹")
            
            if dir_path:
                # 提取文件夹名作为显示名称
                dir_name = os.path.basename(dir_path)
                display_text = f"[数据集] {dir_name}"
                
                if is_target == 'yes':
                    self.target_dataset = dir_path
                else:
                    self.orignal_dataset = dir_path
        elif selected_item == 'MNIST':
            dir_path = 'dataFolder/'
                
            # 添加到列表
            MNIST_db = torchvision.datasets.MNIST(root= dir_path, train=True, transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, )),
                transforms.Resize((224, 224))
            ]))

            MNIST_Loader = DataLoader(MNIST_db, batch_size= 32, shuffle= True)

            if is_target == 'yes':
                self.target_dataset = MNIST_Loader
            else:
                self.orignal_dataset = MNIST_Loader
        elif selected_item == 'SVHN':
            dir_path = 'dataFolder/svhn'
                
            # 添加到列表
            svhn_db = torchvision.datasets.SVHN(root= dir_path, split='train', download=False, transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, )),
                transforms.Resize((224, 224))
                ]))

            svhn_loader = DataLoader(svhn_db, batch_size= 32, shuffle= True)

            if is_target == 'yes':
                self.target_dataset = svhn_loader
            else:
                self.orignal_dataset = svhn_loader

        self.update_listbox2()
        # 更新状态
        self.status_var.set("已导入数据集路径")

    def import_img_file(self):
        dir_path = filedialog.askopenfilename(title="选择测试图片")
        if dir_path:
            # 提取文件夹名作为显示名称
            dir_name = os.path.basename(dir_path)
            display_text = f"[图片名] {dir_name}"
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, )),
                transforms.Resize((224, 224))
            ])
            
            if self.in_channels == 0:
                messagebox.showerror("错误", "请先选择模型")
                return

            if self.in_channels == 3: img = Image.open(dir_path).convert('RGB')
            else: img = Image.open(dir_path).convert('L')
            # 添加到列表
            self.orignal_img = img
            self.test_img = transform(img)
            self.test_img = self.test_img.unsqueeze(0).to(self.device)
            
            # 更新状态
            self.status_var.set(f"已导入测试图片: {dir_name}") 


    def clear_list(self):
        """清空列表"""
        if messagebox.askyesno("确认", "确定要清空所有列表项吗?"):
            self.listbox.delete(0, tk.END)
            self.model = None
            self.dataset = None
            self.status_var.set("列表已清空")
    
    def visualize_cam(self, img, cam, name):
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
        plt.show()
        # plt.savefig(f'data_imgs/grad_cam/ShuffleNetV2/E{epochs}/{in_channels}C/{name}.png',
        # pad_inches=0.01         # 裁剪后保留的最小边距（可选）
        # )

    def on_item_click(self, event):
        """列表项点击事件处理"""
        # 获取选中的索引
        index = self.listbox.curselection()
        if not index:
            return
        
        # if self.test_img is None:
        #     messagebox.showerror("错误", "请先导入测试图片")
        #     return
            
        # index = index[0]
        # selected_item = self.listbox.get(index)
        # print(selected_item)
        # grad_cam = GradCAM(self.model, selected_item)
        # cam = grad_cam(self.test_img)
        # self.visualize_cam(self.orignal_img, cam, selected_item)
        # grad_cam.remove_hooks()

        index = index[0]
        selected_item = self.listbox.get(index)
        print(selected_item)
        avg_iou, avg_entropy = main(self.target_dataset, self.model_type, self.model_path, self.in_channels, selected_item, batch_size= 10)
        messagebox.showinfo("结果", f"平均IoU: {avg_iou:.4f}\n平均熵: {avg_entropy:.4f}")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = DataListApp(root)
    root.mainloop()