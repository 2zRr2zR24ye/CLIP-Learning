import argparse
import torch
import torch.nn as nn
import clip

from .model.clip_utils import load_clip
from .model.custom_clip import customclip
from .data.datasets import build_data_loader

def parse_args():
      parser = argparse.ArgumentParser()
      parser.add_argument("--data-root", type=str, required=True,
                          help="数据集根目录（ImageFolder 格式）")
      parser.add_argument("--backbone", type=str, default="ViT-B/16",
                          help="CLIP backbone，如 ViT-B/16, RN50")
      parser.add_argument("--n-ctx", type=int, default=16,
                          help="可学习 context token 数量")
      parser.add_argument("--lr", type=float, default=0.002,
                          help="学习率")
      parser.add_argument("--epochs", type=int, default=50,
                          help="训练轮数")
      parser.add_argument("--batch-size", type=int, default=32,
                          help="batch size")
      parser.add_argument("--device", type=str, default="cuda",
                          help="设备")
      parser.add_argument("--output-dir", type=str, default="./outputs",
                          help="输出目录")
      return parser.parse_args()

@torch.no_grad()
def evaluate(model, loader, device):
      model.eval()
      correct, total = 0, 0
      for images, labels in loader:
          images = images.to(device).type(model.dtype)
          labels = labels.to(device)

          logits = model(images)                     # (B, n_cls)
          preds = logits.argmax(dim=1)               # (B,)
          correct += (preds == labels).sum().item()  # 累加正确的
          total += len(labels)                       # 累加总数

      return correct / total                         # 0~1 的 float

def train_one_epoch(model, loader, optimizer, criterion, device):
      model.train()
      total_loss = 0

      for images, labels in loader:
          images = images.to(device).type(model.dtype)
          labels = labels.to(device)

          logits = model(images)                     # (B, n_cls)
          loss = criterion(logits, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          total_loss += loss.item()

      return total_loss / len(loader)                # 平均 loss

def main():
      args = parse_args()

      # 1. 加载 CLIP preprocess（直接用 clip.load）
      _, preprocess = clip.load(args.backbone, device="cpu")

      # 2. 数据
      train_loader, classnames = build_data_loader(
          args.data_root, preprocess, args.batch_size
      )

      # 3. 模型
      clip_model, _, _, _, _ = load_clip(args.backbone, device=args.device)
      model = customclip(classnames, clip_model, n_ctx=args.n_ctx, device=args.device)
      model = model.to(args.device)

      # 4. 优化器（只更新 prompt_learner 的参数）
      optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
      criterion = nn.CrossEntropyLoss()

      # 5. 训练循环
      for epoch in range(args.epochs):
          avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
          acc = evaluate(model, train_loader, args.device)
          print(f"Epoch {epoch+1:3d}/{args.epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.2%}")

      # 6. 保存
      torch.save({
          "model_state_dict": model.state_dict(),
          "classnames": classnames,
      }, f"{args.output_dir}/checkpoint.pth")
      print("Done.")


if __name__ == "__main__":
      main()