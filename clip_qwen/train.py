import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 确保能找到项目根目录

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from clip_qwen.model import VLMModel          
from data.datasets import VLMDataset            
from torch.cuda.amp import autocast, GradScaler

# 1. 配置参数
DEVICE = "cuda"
CLIP_MODEL = "openai/clip-vit-base-patch32"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "/root/autodl-tmp/my_vlm/dataset/image"  # 你的图片存放路径
ANNO_FILE = "/root/autodl-tmp/my_vlm/dataset/annotations.jsonl"  # 你的标注文件
BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-3
OUTPUT_DIR = "/root/autodl-tmp/my_vlm/output"
SAVE_STEPS = 1000  # 每多少步保存一次 checkpoint
LOG_STEPS = 10     # 每多少步打印一次 loss

# 2. 初始化 Tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

# 3. 初始化模型
model = VLMModel(CLIP_MODEL, LLM_MODEL, DEVICE)

# 检查 <|image|> 是否在词汇表中
if '<|image|>' not in llm_tokenizer.get_vocab():
    num_added = llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|image|>']})
    print(f"add {num_added} special token: <|image|>")
    model.llm.resize_token_embeddings(len(llm_tokenizer))
    print(f"The model embedding layer has been adjusted to {len(llm_tokenizer)}")
else:
    print("<|image|> already exists in the vocabulary")

# 4. 创建数据集和 DataLoader
dataset = VLMDataset(DATA_DIR, ANNO_FILE, CLIP_MODEL, llm_tokenizer)

# 【可选】取子集测试，如果不需要可以注释掉下面两行
# dataset = Subset(dataset, range(1000))  # 先跑 1000 条

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# 5. 设置优化器 (只训练 Projector)
optimizer = torch.optim.AdamW(model.projector.parameters(), lr=LR)

# 6. 获取 <|image|> 的 token ID
image_token_id = llm_tokenizer.convert_tokens_to_ids('<|image|>')

# 7. 混合精度缩放器
scaler = GradScaler()

# 8. 训练循环
model.train()
global_step = 0
total_samples = len(dataloader)

for epoch in range(EPOCHS):
    total_loss = 0
    print(f"\n Epoch {epoch+1}/{EPOCHS} starts，total batch:{total_samples}")
    
    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch['pixel_values'].to(DEVICE, non_blocking=True)
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)

        # 前向传播（混合精度）
        with autocast():
            outputs = model(pixel_values, input_ids, attention_mask, image_token_id)
            loss = outputs.loss

        # 反向传播（混合精度）
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        global_step += 1

        # 【新增】每 LOG_STEPS 步打印一次当前 loss
        if global_step % LOG_STEPS == 0:
            avg_loss_so_far = total_loss / (batch_idx + 1)
            print(f"  Step {global_step}/{total_samples} (Batch {batch_idx+1}), "
                  f"Loss: {loss.item():.4f}, 平均 Loss: {avg_loss_so_far:.4f}")

        # 每 SAVE_STEPS 步保存一次 checkpoint
        if global_step % SAVE_STEPS == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': global_step,
            }, f"{OUTPUT_DIR}/checkpoint_{global_step}.pt")
            print(f"Checkpoint saved at step {global_step}")

    # Epoch 结束后打印平均 loss
    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1}/{EPOCHS} finish，Average Loss: {avg_loss:.4f}\n")

print(" finish ")