"""
这个文件负责将promptlearner和clip拼接起来组装成coop模型
"""
import clip
import torch
import torch.nn as nn
from .prompt_learner import PromptLearner

class customclip(nn.Module):
    def __init__(self, classnames: list[str], clip_model: nn.Module, n_ctx: int = 16, device: str = "cuda"):
        super().__init__()
        #加载组件   
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.device = device

        #冻结clip的所有参数
        for param in clip_model.parameters():
            param.requires_grad = False

        #创建promptlearner
        self.promptlearner = PromptLearner(classnames, clip_model, n_ctx, device)

        # 为了拿到sos和eos的token，这里随便取一个classnames来拿
        tokenized = clip.tokenize(self.classnames[0]).to(self.device)

        with torch.no_grad():
            embed = self.token_embedding(tokenized)
        #获取eos的位置
        eos_idx = (tokenized[0] == 49407).int().argmax().item()
        self.register_buffer("sos", embed[0, 0:1, :])
        self.register_buffer("eos", embed[0, eos_idx:eos_idx+1, :])

    
    #文本编码器
    def encode_text(self):
        
        prompt = self.promptlearner()
        sos = self.sos.unsqueeze(0).expand(len(self.classnames), -1, -1)
        pad_len = 77 - 1 - (self.n_ctx + 1)
        eos_pad = self.eos.unsqueeze(0).expand(len(self.classnames), pad_len, -1)
        
        full_text = torch.cat([sos, prompt, eos_pad], dim = 1)
        # CLIP 的 transformer 期望 (seq, batch, dim) 即 (77, n_cls, dim)
        x = full_text.permute(1, 0, 2).type(self.dtype)                # (77, n_cls, dim)
        x = self.text_encoder(x)                                       # (77, n_cls, dim)
        x = x.permute(1, 0, 2)                                         # (n_cls, 77, dim)

        #编码后取eos的位置作为文本编码器的输出
        eos_pos = self.n_ctx + 2
        text_features = x[:, eos_pos, :]

        text_features = text_features / text_features.norm(dim = 1, keepdim = True)

        return text_features

    #图像编码器
    def encode_image(self, image):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim = 1, keepdim = True)
        return image_features

    def forward(self, image):
        image_features = self.encode_image(image)
        text_features = self.encode_text()

        #计算余弦相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

