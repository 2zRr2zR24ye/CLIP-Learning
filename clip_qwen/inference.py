import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
from clip_qwen.proj import SimpleProj

#1. 定义 Projector
class SimpleProj(nn.Module):
    def __init__(self, input_dim=768, output_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),  # 升维
            nn.GELU(),                   # 非线性激活
            nn.Linear(2048, output_dim)  # 降维到目标维度
        )
        # 更好的初始化：使用 xavier 或 kaiming
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.mlp(x)

#2. 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = "/root/autodl-tmp/my_vlm/output/checkpoint_10000.pt"  # 改成实际路径
IMAGE_PATH = "/root/autodl-tmp/my_vlm/cat.jpg"
QUESTION = "Please describe this image in detail."

# 3. 加载模型
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

print("Loading Tokenizer and adding special token...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
if '<|image|>' not in llm_tokenizer.get_vocab():
    print("Adding <|image|> token to tokenizer")
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|image|>']})

print("Loading LLM...")
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# 调整嵌入层以匹配 tokenizer（因为添加了新 token）
llm_model.resize_token_embeddings(len(llm_tokenizer))
llm_model.eval()
for param in llm_model.parameters():
    param.requires_grad = False

print("Loading Projector checkpoint...")
projector = SimpleProj().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# 方法1：如果保存的是完整模型 state_dict
if 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
    # 只提取 projector 部分（键名以 'projector.' 开头）
    projector_state_dict = {k.replace('projector.', ''): v for k, v in model_state_dict.items() if k.startswith('projector.')}
    projector.load_state_dict(projector_state_dict)
    print("loading Projector weights from checkpoint")
else:
    # 方法2：如果保存的是纯 projector state_dict
    projector.load_state_dict(checkpoint)
    print("loading Projector weights")
projector.eval()
# 注意：Projector 保持 float32，因为 CLIP 输出是 float32

print("All models loaded successfully!")

# 4. 推理函数
def inference(image_path, question):
    # 4.1 图片处理
    image = Image.open(image_path).convert('RGB')
    pixel_values = clip_processor(images=image, return_tensors="pt")['pixel_values'].to(DEVICE)

    # 4.2 提取 CLIP 特征
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values)
        image_features = vision_outputs.last_hidden_state  # [1, 50, 768]

    # 4.3 投影（输出为 float32）
    projected_features = projector(image_features)  # [1, 50, 1536] float32

    # 4.4 构建 prompt
    conversation = [
        {"role": "user", "content": f"<|image|>\n{question}"},
    ]
    prompt = llm_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    print(f"Prompt: {prompt[:200]}...")  # 调试

    # 4.5 Tokenize
    text_inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    input_ids = text_inputs['input_ids'].to(DEVICE)
    attention_mask = text_inputs['attention_mask'].to(DEVICE)
    print(f"input_ids shape: {input_ids.shape}")

    # 4.6 替换 <|image|>
    image_token_id = llm_tokenizer.convert_tokens_to_ids('<|image|>')
    print(f"image_token_id: {image_token_id}")

    if image_token_id is None or image_token_id == llm_tokenizer.unk_token_id:
        raise ValueError("Tokenizer does not contain <|image|> token. Please add it.")

    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        print("Warning: <|image|> token not found in prompt. Using original input.")
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)
    else:
        pos = pos[0]
        inputs_embeds = llm_model.get_input_embeddings()(input_ids)  # [1, seq_len, 1536] float32
        # 将 projected_features 转换为与 llm_model 相同的 dtype (bfloat16)
        projected_features = projected_features.to(dtype=llm_model.dtype)
        # 替换
        inputs_embeds = torch.cat([
            inputs_embeds[:, :pos],
            projected_features,
            inputs_embeds[:, pos+1:]
        ], dim=1)
        # 更新 attention_mask
        attention_mask = torch.cat([
            attention_mask[:, :pos],
            torch.ones(1, projected_features.size(1), device=DEVICE, dtype=attention_mask.dtype),
            attention_mask[:, pos+1:]
        ], dim=1)

    # 确保 inputs_embeds 的 dtype 与模型一致
    inputs_embeds = inputs_embeds.to(dtype=llm_model.dtype)

    # 4.7 生成
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )

    # 4.8 解码
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#5. 运行
if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        print(f"\n--- Testing image: {IMAGE_PATH} ---")
        print(f"Question: {QUESTION}")
        result = inference(IMAGE_PATH, QUESTION)
        print(f"\n Model Answer:\n{result}")
    else:
        print(f"Error: Image file '{IMAGE_PATH}' not found.")