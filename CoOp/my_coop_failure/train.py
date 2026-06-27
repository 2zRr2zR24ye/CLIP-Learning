import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from CoOp.my_coop_failure.model import CLIP, promptlearner
from CoOp.my_coop_failure.datasets import ClassificationDataset
from CoOp.my_coop_failure.utils import set_seed, load_config, save_checkpoint, accuracy


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    set_seed(config['seed'])
    
    # 设备
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    
    #加载 CLIP
    clip_model = CLIP(config['clip_model_name'], device)
    
    #准备数据集
    def preprocess(image):
        return clip_model.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

    train_transform = preprocess
    val_transform = preprocess
    
    train_dataset = ClassificationDataset(
        os.path.join(config['data_root'], 'train'),
        transform=train_transform,
        shots=config.get('shots', None)
    )
    val_dataset = ClassificationDataset(
        os.path.join(config['data_root'], 'val'),
        transform=val_transform
    ) if os.path.exists(os.path.join(config['data_root'], 'val')) else None #若有训练集则使用
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('test_batch_size', config['batch_size']),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    class_names = train_dataset.class_names
    print(f"类别：{class_names}")
    
    #初始化 PromptLearner
    prompt_learner = promptlearner(
        clip_model,
        class_names,
        n_ctx=config['n_ctx'],
        ctx_init=config.get('ctx_init', None)
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(
        prompt_learner.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    #学习率调度器 
    total_steps = len(train_loader) * config['epochs']
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config['lr'] * 0.01
    )
    
    #训练
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    
    print(f"\nstart train step: {total_steps}")
    
    for epoch in range(1, config['epochs'] + 1):
        # 训练
        prompt_learner.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取图像特征
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 构建 prompt 并提取文本特征
            prompt_embeddings = prompt_learner()  # [num_classes, seq_len, dim]
            seq_len = prompt_embeddings.size(1)
            attention_mask = torch.ones(prompt_embeddings.size(0), seq_len, device=device)

            text_features = clip_model.encode_text_from_embeddings(
                prompt_embeddings,
                attention_mask
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度和损失
            logits = image_features @ text_features.T  # [batch, num_classes]
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 统计
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            
            # 打印日志
            if (batch_idx + 1) % config.get('log_interval', 50) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                acc = total_correct / total_samples
                print(
                    f"Epoch [{epoch}/{config['epochs']}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {acc:.4f} "
                    f"LR: {current_lr:.6f}"
                )
        
        train_acc = total_correct / total_samples
        train_loss = total_loss / len(train_loader)
        
        # 验证
        val_acc = 0.0
        if val_loader:
            prompt_learner.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    image_features = clip_model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    prompt_embeddings = prompt_learner()
                    seq_len = prompt_embeddings.size(1)
                    attention_mask = torch.ones(
                        prompt_embeddings.size(0), seq_len, device=device
                    )
                    text_features = clip_model.encode_text_from_embeddings(
                        prompt_embeddings, attention_mask
                    )
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    logits = image_features @ text_features.T
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            
            val_acc = correct / total
            print(f"Epoch [{epoch}/{config['epochs']}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': prompt_learner.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config,
                }, os.path.join(config['output_dir'], 'best_model.pth'))
                print(f"best model saved acc: {best_acc:.4f}")
        else:
            print(f"Epoch [{epoch}/{config['epochs']}] Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")
    
    print(f"\nfinish best acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()