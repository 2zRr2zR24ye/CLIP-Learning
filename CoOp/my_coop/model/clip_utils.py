"""
clip模型加载工具
"""

import torch.nn as nn
import clip

def load_clip(backbone_name: str, device: str = "cuda"):
    
    clip_model, preprocessor = clip.load(backbone_name, device)
    token_embedding = clip_model.token_embedding
    image_encoder = clip_model.visual
    text_encoder = clip_model.transformer
    embed_dim = token_embedding.embedding_dim

    return clip_model, token_embedding, image_encoder, text_encoder, embed_dim


