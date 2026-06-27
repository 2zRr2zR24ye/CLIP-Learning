"""
promptlearner: 可学习的连续prompt context vectors
在这里我们只实现最简版本，所有class共享一组上下文，class token在文末
"""
import torch
import torch.nn as nn
import clip

class PromptLearner(nn.Module):
    def __init__(self, classnames: list[str], clip_model: nn.Module, n_ctx: int = 16, device: str = "cuda"):
        super().__init__()

        token_embedding = clip_model.token_embedding
        ctx_dim = token_embedding.embedding_dim
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device

        #创建可学习的文本向量
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        ctx_vectors.normal_(mean = 0, std = 0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        #为每个class实现class token embedding
        tokenized = clip.tokenize(classnames).to(self.device)

        with torch.no_grad():
            embedding = token_embedding(tokenized)

        #这里我已经拿到了每个class的embedding，格式大概是[sos, class, eos],这时我只需要找到sos和eos，取其中间的即可

        eos_idx = (tokenized == 49407).int().argmax(dim = 1)

        class_token_embeddings_list = []
        for i in range(self.n_cls):
            token_i = embedding[i, 1:eos_idx[i], :]
            class_token_embeddings_list.append(token_i.mean(dim=0))
        # 堆叠并注册为缓冲区（而非直接赋值）
        class_token_embeddings = torch.stack(class_token_embeddings_list)
        self.register_buffer("class_token_embeddings", class_token_embeddings)
        
    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        class_tok = self.class_token_embeddings.unsqueeze(1)

        prompts = torch.cat([ctx, class_tok], dim = 1)

        return prompts
