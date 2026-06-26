import os
import tarfile
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    """加载 CIFAR-10 的 pickle 文件"""
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def convert_cifar10_from_gz(gz_path="./data/cifar-10-python.tar.gz", 
                            output_dir="./data", 
                            val_ratio=0.2):
    """
    从本地的 cifar-10-python.tar.gz 解压并转换为文件夹结构
    
    参数:
        gz_path: tar.gz 文件路径
        output_dir: 输出根目录
        val_ratio: 验证集比例（从训练集中分出来的比例）
    """
    
    # 1. 解压
    print(f"解压 {gz_path}...")
    extract_dir = os.path.join(output_dir, "cifar-10-batches-py")
    
    # 如果已经解压过，跳过解压
    if not os.path.exists(extract_dir):
        with tarfile.open(gz_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        print(f"✅ 解压到: {extract_dir}")
    else:
        print(f"✅ 已存在解压目录: {extract_dir}")
    
    # 2. 加载所有数据
    print("加载 CIFAR-10 数据...")
    
    # 类别名称（CIFAR-10 官方顺序）
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 加载训练集 (data_batch_1 到 data_batch_5)
    all_images = []
    all_labels = []
    
    for i in range(1, 6):
        batch_path = os.path.join(extract_dir, f'data_batch_{i}')
        batch = unpickle(batch_path)
        # batch[b'data'] 形状: [10000, 3072]，每张图是 32x32x3 (RGB)
        images = batch[b'data']
        labels = batch[b'labels']
        all_images.append(images)
        all_labels.extend(labels)
    
    train_images = np.concatenate(all_images, axis=0)  # [50000, 3072]
    train_labels = np.array(all_labels)                # [50000]
    
    # 加载测试集
    test_batch_path = os.path.join(extract_dir, 'test_batch')
    test_batch = unpickle(test_batch_path)
    test_images = test_batch[b'data']   # [10000, 3072]
    test_labels = test_batch[b'labels'] # [10000]
    
    print(f"训练集: {len(train_images)} 张")
    print(f"测试集: {len(test_images)} 张")
    
    # 3. 创建输出文件夹
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    # 4. 按类别分组训练集
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(train_labels):
        class_indices[label].append(idx)
    
    # 5. 保存图片（训练集 -> train/ + val/）
    print("保存训练集和验证集...")
    
    for class_idx, indices in class_indices.items():
        class_name = class_names[class_idx]
        n_total = len(indices)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        
        # 取前 n_train 张作为训练，后 n_val 张作为验证
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # 保存训练集
        for i, idx in enumerate(train_indices):
            # 将 3072 维向量转成 32x32x3 的 RGB 图片
            img_data = train_images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img_data)
            img.save(os.path.join(train_dir, class_name, f"{i+1:05d}.jpg"))
        
        # 保存验证集
        for i, idx in enumerate(val_indices):
            img_data = train_images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img_data)
            img.save(os.path.join(val_dir, class_name, f"{i+1:05d}.jpg"))
        
        print(f"  {class_name}: train={n_train}, val={n_val}")
    
    # 6. 保存测试集（可选，这里命名为 test 文件夹）
    print("保存测试集...")
    test_dir = os.path.join(output_dir, 'test')
    for class_name in class_names:
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    for idx, (img_data, label) in enumerate(zip(test_images, test_labels)):
        img_data = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img_data)
        class_name = class_names[label]
        img.save(os.path.join(test_dir, class_name, f"{idx+1:05d}.jpg"))
    
    print(f"\n✅ 转换完成！")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  测试集: {test_dir}")
    print(f"  类别数: {len(class_names)}")
    print(f"  类别: {class_names}")


if __name__ == "__main__":
    # 请根据你的实际路径修改
    gz_path = "/root/autodl-pub/cifar-10/cifar-10-python.tar.gz"  # 你下载的 gz 文件路径
    output_dir = "/root/autodl-tmp/my_coop/data"
    
    convert_cifar10_from_gz(gz_path, output_dir, val_ratio=0.2)