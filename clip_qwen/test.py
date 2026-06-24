import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer

#加载一个clip模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#冻结clip所有参数
for param in clip_model.parameters():
    param.requires_grad = False

# 将模型设置为评估模式
clip_model.eval()
# 将模型放到 GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

# 2. 加载一个轻量级 LLM (作为文本生成器)
# 这里使用 Qwen2.5-1.5B-Instruct，它性能优秀且对消费级显卡友好
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # 使用 bfloat16 节省显存
    device_map="auto"           # 让 accelerate 自动分配设备
)
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

# 我们暂时冻结 LLM，稍后可以选择性地用 LoRA 微调
for param in llm_model.parameters():
    param.requires_grad = False

# 制造一张假的 224x224 图片 (Batch=1, 通道=3, 高=224, 宽=224)
dummy_image = torch.randn(1, 3, 224, 224).to("cuda")
# 放进 CLIP 的视觉编码器（只取视觉部分，不要整个 CLIP 模型）
with torch.no_grad():
    vision_outputs = clip_model.vision_model(dummy_image)
    
print(f"CLIP_shape: {vision_outputs.last_hidden_state.shape}")