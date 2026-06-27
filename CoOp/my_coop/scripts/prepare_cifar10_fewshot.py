
import argparse
import os
import pickle
import tarfile
import random
import shutil
import numpy as np
from PIL import Image


  # CIFAR-10 的类别名（按顺序 0~9）
CLASSNAMES = [
      "airplane", "automobile", "bird", "cat", "deer",
      "dog", "frog", "horse", "ship", "truck",
  ]


def parse_args():
      parser = argparse.ArgumentParser()
      parser.add_argument("--tar-path", type=str, required=True,
                          help="cifar-10-python.tar.gz 路径")
      parser.add_argument("--output-dir", type=str, default="./data/cifar10_fewshot",
                          help="输出目录（ImageFolder 格式）")
      parser.add_argument("--shots", type=int, default=16,
                          help="每类取多少张")
      parser.add_argument("--seed", type=int, default=1,
                          help="随机种子")
      return parser.parse_args()


def extract_tar(tar_path, tmp_dir):
      """解压 tar.gz 到临时目录"""
      with tarfile.open(tar_path, "r:gz") as tar:
          tar.extractall(tmp_dir)


def load_cifar10_batch(filepath):
      """加载单个 CIFAR-10 batch 文件"""
      with open(filepath, "rb") as f:
          batch = pickle.load(f, encoding="latin1")
      images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
      labels = batch["labels"]
      return images, labels


def load_all_cifar10(data_dir):
      """加载所有 CIFAR-10 训练 batch，返回 (images, labels)"""
      all_images, all_labels = [], []
      for i in range(1, 6):  # data_batch_1 ~ data_batch_5
          imgs, lbls = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
          all_images.append(imgs)
          all_labels.extend(lbls)
      all_images = np.concatenate(all_images, axis=0)   # (50000, 32, 32, 3)
      all_labels = np.array(all_labels)                  # (50000,)
      return all_images, all_labels


def save_fewshot(images, labels, output_dir, shots, seed):
      """
      每个类别随机抽 shots 张，存为 ImageFolder 格式:
          output_dir/
          ├── airplane/
          │   ├── 001.png
          │   └── ...
          ├── automobile/
          │   └── ...
          └── truck/
              └── ...
      """
      random.seed(seed)
      os.makedirs(output_dir, exist_ok=True)

      for cls_id, cls_name in enumerate(CLASSNAMES):
          # 找到该类的所有图片索引
          indices = np.where(labels == cls_id)[0]
          # 随机抽 shots 张
          selected = random.sample(list(indices), shots)

          cls_dir = os.path.join(output_dir, cls_name)
          os.makedirs(cls_dir, exist_ok=True)

          for i, idx in enumerate(selected):
              img = Image.fromarray(images[idx])
              img.save(os.path.join(cls_dir, f"{i+1:04d}.png"))

          print(f"  {cls_name}: {len(selected)} 张")


def main():
      args = parse_args()

      # 1. 解压到临时目录
      tmp_dir = "/tmp/cifar10_tmp"
      print(f"解压 {args.tar_path} ...")
      extract_tar(args.tar_path, tmp_dir)

      # 2. 找到解压后的实际目录（cifar-10-batches-py/）
      cifar_dir = os.path.join(tmp_dir, "cifar-10-batches-py")
      if not os.path.isdir(cifar_dir):
          # 有时候 tar 里没有外层目录，直接就是 data_batch_1 等
          cifar_dir = tmp_dir

      # 3. 加载全部训练数据
      print("加载 CIFAR-10 训练数据...")
      images, labels = load_all_cifar10(cifar_dir)
      print(f"  总图片数: {len(images)}")

      # 4. 每类抽 shots 张，存为 ImageFolder
      print(f"每类抽取 {args.shots} 张，并存到 {args.output_dir} ...")
      save_fewshot(images, labels, args.output_dir, args.shots, args.seed)

      # 5. 清理临时目录
      shutil.rmtree(tmp_dir)
      print("完成。")


if __name__ == "__main__":
      main()