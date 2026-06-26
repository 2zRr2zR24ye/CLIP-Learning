import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor


class CLIP:
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = device
        self.clip_model_name = clip_model_name

        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(clip_model_name)

        self.text_encoder = self.clip_model.text_model
        self.vision_encoder = self.clip_model.vision_model

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.token_embedding = self.clip_model.text_model.embeddings.token_embedding
        self.dtype = self.clip_model.dtype

        self.clip_model.eval()

    # ========== 图像编码 ==========
    def encode_image(self, images):
        if not isinstance(images, torch.Tensor):
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        else:
            pixel_values = images.to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values)
        return image_features

    # ========== 标准文本编码 ==========
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return text_features

    # ========== CoOp 专用：从嵌入向量提取文本特征 ==========
    def encode_text_from_embeddings(self, inputs_embeds, attention_mask=None):
        """
        手动实现文本前向传播，支持自定义嵌入
        inputs_embeds: [batch_size, seq_len, embed_dim]
        attention_mask: [batch_size, seq_len] (可选)
        """
        inputs_embeds = inputs_embeds.to(self.device).to(self.dtype)
        batch_size, seq_len, _ = inputs_embeds.shape

        # 1. 添加位置编码
        pos_embed = self.clip_model.text_model.embeddings.position_embedding.weight
        if seq_len > pos_embed.shape[0]:
            pos_embed = torch.cat([pos_embed, pos_embed[-1:].repeat(seq_len - pos_embed.shape[0], 1)], dim=0)
        else:
            pos_embed = pos_embed[:seq_len, :]
        x = inputs_embeds + pos_embed.unsqueeze(0)  # [batch, seq_len, dim]

        # 2. 调整维度：LND -> 序列优先
        x = x.permute(1, 0, 2)  # [seq_len, batch, dim]

        # 3. 经过 Transformer 编码器（不传 attention_mask，避免维度问题）
        # CLIPEncoder 内部默认处理全1掩码
        x = self.text_encoder.encoder(
            inputs_embeds=x,
            attention_mask=None,  # 关键：避免掩码维度错误
        ).last_hidden_state  # [seq_len, batch, dim]

        # 4. 调整回 [batch, seq_len, dim]
        x = x.permute(1, 0, 2)

        # 5. 最终 Layer Norm
        x = self.text_encoder.final_layer_norm(x)

        # 6. 取 EOS token 位置的特征
        if attention_mask is not None:
            # 使用 attention_mask 找到最后一个有效 token
            eos_indices = attention_mask.sum(dim=1) - 1
        else:
            # 默认取最后一个 token
            eos_indices = torch.full((batch_size,), seq_len - 1, device=self.device)
        eos_indices = eos_indices.long()
        text_features = x[torch.arange(batch_size, device=self.device), eos_indices]  # [batch, dim]

        # 7. 投影到共享空间
        text_features = text_features @ self.clip_model.text_model.projection

        return text_features

    # ========== 辅助方法 ==========
    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_embed_dim(self):
        return self.token_embedding.weight.shape[1]

    def get_token_embedding(self, input_ids):
        return self.token_embedding(input_ids)