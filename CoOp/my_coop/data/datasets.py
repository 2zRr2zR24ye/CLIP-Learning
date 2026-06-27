from torch.utils.data import DataLoader
from torchvision import datasets

def build_data_loader(data_root: str, preprocess, batch_size: int = 32):
      """
      Args:
          data_root:   数据集根目录，子文件夹名 = 类别名
          preprocess:  CLIP 的图像预处理 pipeline
          batch_size:  batch size

      Returns:
          train_loader: DataLoader，每次迭代返回 (images, labels)
          classnames:   list[str]，如 ["accordion", "anchor", "ant"]
      """
      train_dataset = datasets.ImageFolder(
          root=data_root,
          transform=preprocess,
      )

      classnames = train_dataset.classes

      train_loader = DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=True,
      )

      return train_loader, classnames