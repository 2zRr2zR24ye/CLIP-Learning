import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ClassificationDataset(Dataset):   
    def __init__(self, data_root, transform=None, shots=None):
        """
        参数:
            data_root: 数据根目录
            transform: 图像预处理（可以是函数或 CLIPImageProcessor）
            shots: 每个类别取多少张图片（None 表示全部使用）
        """
        self.data_root = data_root
        self.transform = transform
        self.shots = shots
        
        self.samples = []
        self.class_names = []
        
        # 遍历子文件夹
        for class_idx, class_name in enumerate(sorted(os.listdir(data_root))):
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_names.append(class_name)
            
            # 获取该类别所有图片
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            # 如果设置了 shots，只取前 shots 张
            if shots is not None and len(images) > shots:
                images = images[:shots]
            
            for img_name in images:
                self.samples.append((
                    os.path.join(class_dir, img_name),
                    class_idx
                ))
        
        print(f"加载数据集: {data_root}")
        print(f"  类别数: {len(self.class_names)}")
        print(f"  样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            # 应用 transform
            output = self.transform(image)
            # 如果 transform 返回的是字典（如 CLIPImageProcessor），提取 pixel_values
            if isinstance(output, dict) and 'pixel_values' in output:
                image = output['pixel_values'].squeeze(0)  # [1, C, H, W] -> [C, H, W]
            else:
                image = output
        else:
            # 没有 transform，转为 tensor（归一化到 0-1）
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image, label