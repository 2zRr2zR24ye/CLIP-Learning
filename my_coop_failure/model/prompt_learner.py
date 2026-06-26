import torch
import torch.nn as nn


class promptlearner(nn.Module):
    def __init__(self, clip_model, class_names, n_ctx = 4, ctx_init = None):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.device = clip_model.device
        self.dtype = clip_model.dtype

        #获取tokenizer
        tokenizer = clip_model.get_tokenizer()
        ctx_dim = clip_model.get_embed_dim()

        #获取类别名称的token id
        tokenized = tokenizer(class_names, return_tensors = "pt", padding = True)
        self.class_token_ids = tokenized['input_ids'][:, 1:-1]
        self.num_class = len(class_names)

        with torch.no_grad():
            class_embeddings = clip_model.get_token_embedding(
                self.class_token_ids.to(self.device)
            )
        
        self.register_buffer('class_embeddings', class_embeddings)

        #初始化可学习的上下文向量
        if ctx_init is not None:
            ctx_tokens = tokenizer(ctx_init, return_tensors="pt")['input_ids']
            with torch.no_grad():
                ctx_embeddings = clip_model.get_token_embedding(ctx_tokens.to(self.device))
            self.ctx = nn.Parameter(ctx_embeddings.squeeze(0)[:n_ctx].clone())
        else:
            self.ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim) * 0.02)

    def forward(self):

        #扩展上下文向量到所有类别
        ctx = self.ctx.unsqueeze(0).expand(self.num_class, -1, -1)

        #拼接上下词+类别词
        prompt_embeddings = torch.cat([ctx, self.class_embeddings], dim = 1)

        return prompt_embeddings

    


