# Generate from deepseek v4 pro
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import CLIP, promptlearner
from datasets import ClassificationDataset
from utils import load_config, load_checkpoint


def inference_single_image(clip_model, prompt_learner, image_path, class_names, device, topk=5):
    """
    单张图片推理
    
    参数:
        clip_model: CLIP 模型实例
        prompt_learner: 训练好的 PromptLearner
        image_path: 图片路径
        class_names: 类别名称列表
        device: 设备
        topk: 显示前 k 个预测结果
    
    返回:
        预测结果列表 [(class_name, probability), ...]
    """
    # 1. 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    pixel_values = clip_model.processor.image_processor(
        image, return_tensors="pt"
    )['pixel_values'].to(device)
    
    # 2. 提取图像特征
    with torch.no_grad():
        image_features = clip_model.encode_image(pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 3. 获取所有类别的文本特征
    with torch.no_grad():
        prompt_embeddings = prompt_learner()  # [num_classes, seq_len, dim]
        seq_len = prompt_embeddings.size(1)
        attention_mask = torch.ones(
            prompt_embeddings.size(0), seq_len, device=device
        )
        text_features = clip_model.encode_text_from_embeddings(
            prompt_embeddings, attention_mask
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 4. 计算相似度
    logits = image_features @ text_features.T  # [1, num_classes]
    probs = logits.softmax(dim=-1)  # [1, num_classes]
    
    # 5. 获取 top-k 结果
    topk_probs, topk_indices = probs[0].topk(topk)
    
    results = []
    for prob, idx in zip(topk_probs, topk_indices):
        results.append((class_names[idx.item()], prob.item()))
    
    return results


def evaluate_on_val_set(clip_model, prompt_learner, val_loader, device):
    """
    在验证集上评估模型
    
    返回:
        accuracy: 准确率
    """
    prompt_learner.eval()
    clip_model.clip_model.eval()
    
    # 预计算文本特征（只算一次，因为类别固定）
    with torch.no_grad():
        prompt_embeddings = prompt_learner()
        seq_len = prompt_embeddings.size(1)
        attention_mask = torch.ones(
            prompt_embeddings.size(0), seq_len, device=device
        )
        text_features = clip_model.encode_text_from_embeddings(
            prompt_embeddings, attention_mask
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = image_features @ text_features.T
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def main():
    parser = argparse.ArgumentParser(description='CoOp 推理/评估')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image', type=str, help='单张图片路径（可选）')
    parser.add_argument('--topk', type=int, default=5, help='显示 Top-K 预测')
    parser.add_argument('--eval', action='store_true', help='在验证集上评估模型')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载 CLIP
    print("加载 CLIP 模型...")
    clip_model = CLIP(config['clip_model_name'], device)
    
    # 获取类别名称（从训练集）
    print("加载类别信息...")
    train_dataset = ClassificationDataset(
        os.path.join(config['data_root'], 'train'),
        transform=clip_model.processor.image_processor
    )
    class_names = train_dataset.class_names
    print(f"类别数: {len(class_names)}")
    print(f"类别: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
    
    # 初始化 PromptLearner
    prompt_learner = promptlearner(
        clip_model,
        class_names,
        n_ctx=config['n_ctx'],
        ctx_init=config.get('ctx_init', None)
    ).to(device)
    
    # 加载训练好的权重
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    
    # 处理不同的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  从完整 checkpoint 加载 (epoch {checkpoint.get('epoch', 'unknown')})")
    elif 'prompt_learner_state_dict' in checkpoint:
        state_dict = checkpoint['prompt_learner_state_dict']
        print(f"  从 PromptLearner checkpoint 加载")
    else:
        state_dict = checkpoint
        print(f"  直接加载 state_dict")
    
    prompt_learner.load_state_dict(state_dict)
    prompt_learner.eval()
    
    print("✅ 模型加载完成\n")
    
    # ----- 模式选择 -----
    
    # 模式 1：单张图片推理
    if args.image:
        print(f"🔍 推理图片: {args.image}")
        results = inference_single_image(
            clip_model, prompt_learner, args.image, class_names, device, args.topk
        )
        
        print(f"\n📊 Top-{args.topk} 预测:")
        print("-" * 40)
        for i, (class_name, prob) in enumerate(results, 1):
            bar = "█" * int(prob * 40)
            print(f"{i:2d}. {class_name:20s} {prob*100:6.2f}%  {bar}")
        return
    
    # 模式 2：验证集评估
    if args.eval:
        print("📊 在验证集上评估...")
        val_path = os.path.join(config['data_root'], 'val')
        if not os.path.exists(val_path):
            print(f"❌ 验证集不存在: {val_path}")
            return
        
        val_dataset = ClassificationDataset(
            val_path,
            transform=clip_model.processor.image_processor
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('test_batch_size', config['batch_size']),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        accuracy = evaluate_on_val_set(clip_model, prompt_learner, val_loader, device)
        print(f"\n🎯 验证集准确率: {accuracy*100:.2f}% ({int(accuracy * len(val_dataset))}/{len(val_dataset)})")
        return
    
    # 如果既没有指定图片也没有指定 --eval，提示用户
    print("❌ 请指定 --image 进行单张推理，或 --eval 进行验证集评估")
    print("示例:")
    print("  python inference.py --checkpoint outputs/best_model.pth --image test.jpg")
    print("  python inference.py --checkpoint outputs/best_model.pth --eval")


if __name__ == '__main__':
    main()