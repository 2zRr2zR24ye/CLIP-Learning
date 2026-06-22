# code reading4
  ## **clip**
这是clip实现的核心，整个代码中最重要的部分。通过这一部分，将图像和文本编码器组合起来
```python
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
```


---

### 第一板块：构造函数 `__init__`（搭积木）

#### 1. 视觉塔分支（Vision Tower）
```python
if isinstance(vision_layers, (tuple, list)):
    # 如果是列表/元组（比如 [3,4,6,3]），走 ResNet
    self.visual = ModifiedResNet(...)
else:
    # 如果是个整数（比如 12），走 Vision Transformer
    self.visual = VisionTransformer(...)
```
- CLIP 官方支持两种骨干网络：**ResNet** 和 **ViT**（Vision Transformer）。
- 这里的判断逻辑很巧妙：`vision_layers` 传列表就是 ResNet（因为 ResNet 需要每层的 block 数量），传单个整数就是 ViT（因为 ViT 只需要知道有多少层）。
- **`vision_heads`** 的计算：
  - ResNet：`vision_width * 32 // 64`（即 `2048 // 64 = 32`，对应注意力池化的 32 个头）。
  - ViT：`vision_width // 64`（即 `768 // 64 = 12`，标准多头注意力头数）。

#### 2. 文本塔分支（Text Tower）
```python
self.transformer = Transformer(...)
self.token_embedding = nn.Embedding(vocab_size, transformer_width)
self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
self.ln_final = LayerNorm(transformer_width)
```
- **`token_embedding`**：查表操作。把 Token ID（比如 42）变成向量（比如 512 维）。
- **`positional_embedding`**：位置编码。因为 Transformer 本身没有“顺序感”，需要给每个位置（第1个词、第2个词...）加一个专属的“位置向量”。
- **`ln_final`**：最后一层归一化，稳定输出。

#### 3. 连接双塔的“桥梁”
```python
self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
```
- **`text_projection`**：文本塔输出的维度是 `transformer_width`（比如 512），图像塔输出的维度是 `embed_dim`（比如 1024）。这就像两个国家（文本国和图像国）的“汇率转换器”，把文本特征映射到图像特征的空间里去。
- **`logit_scale`**：这是一个**可学习的温度系数**。`np.log(1 / 0.07)` 约等于 `2.659`，即初始缩放值约为 `exp(2.659) ≈ 14.29`。为什么要有它？对比学习时，如果特征向量都归一化了，点积范围在 [-1, 1] 之间，梯度太小不好训练。这个系数把相似度拉宽，让模型学得更带劲。

---

### 第二板块：参数初始化 `initialize_parameters`（给模型注入“基因”）

CLIP 的初始化不是随便乱来的，有极强的工程讲究：

1. **文本嵌入与位置编码**：
   ```python
   nn.init.normal_(self.token_embedding.weight, std=0.02)
   nn.init.normal_(self.positional_embedding, std=0.01)
   ```
   - 标准的 GPT 式初始化，小方差保持训练初期稳定。

2. **ResNet 的注意力池化层**：
   ```python
   std = self.visual.attnpool.c_proj.in_features ** -0.5
   ```
   - 用 `1/sqrt(in_features)` 缩放投影层，防止注意力输出的方差爆炸。

3. **残差块中的 `bn3` 零初始化**（极重要的细节！）：
   ```python
   if name.endswith("bn3.weight"):
       nn.init.zeros_(param)
   ```
   - ResNet 的 Bottleneck 里，最后一个 BatchNorm 的 `weight` 初始化为 0。
   - **这意味着什么？** 在训练刚开始时，这个残差块的输出全为 0，等效于“恒等映射”。模型从“浅层”开始学，随着训练进行，`weight` 慢慢变大，深层网络逐渐“解锁”。这是一种**渐进式学习**的技巧，能让极深的 ResNet 稳定收敛。

4. **Transformer 的残差层缩放**（GPT 常用的 `Proj` 初始化）：
   ```python
   proj_std = (width ** -0.5) * ((2 * layers) ** -0.5)
   ```
   - 为什么乘以 `(2 * layers)^-0.5`？因为残差连接会把每一层的方差累加起来。为了不让 12 层堆叠后输出方差爆炸，每一层的投影层要除以 `sqrt(2 * 层数)`。这也是 GPT 模型能训练成功的关键工程细节。

---

### 第三板块：编码逻辑 `encode_text`（文本如何变成向量？）

这是整个模型最“硬核”的数据流，我们拆成 4 步看：

```python
def encode_text(self, text):
    # 1. 查表变向量
    x = self.token_embedding(text).type(self.dtype)  # [batch, 77, 512]

    # 2. 加位置编码
    x = x + self.positional_embedding.type(self.dtype)

    # 3. 调换维度，喂给 Transformer
    x = x.permute(1, 0, 2)  # [batch, 77, 512] -> [77, batch, 512]
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # [77, batch, 512] -> [batch, 77, 512]

    # 4. 归一化
    x = self.ln_final(x).type(self.dtype)
```
做完这一步，我们就得到了 77 个位置各自的向量。

**提取 EOT（End Of Text）标记**
```python
x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
```
- `text.argmax(dim=-1)`：找出每个句子中 **Token ID 最大的位置**。在 BPE 编码规则里，`<|endoftext|>` 的 ID 往往最大（比如 49407）。这一步是为了动态找出每个句子中“结束符”的位置（因为句子长度可能小于 77）。
- `torch.arange(x.shape[0])`：对应每个 batch 索引。
- 最终取出 EOT 位置的向量，乘以 `text_projection`，映射到最终的公共空间 `[batch, embed_dim]`。

> **为什么取 EOT 而不是取平均？**  
> Transformer 是双向的（因果掩码只限制它看不到未来，但能看到过去），EOT 位置作为句子的“终结者”，它看到了整个句子的全部信息，天然适合作为句子的**全局语义汇聚点**。

---

### 第四板块：前向传播 `forward`（对比学习的核心算式）

**计算相似度矩阵**：

```python
def forward(self, image, text):
    # 1. 提取特征
    image_features = self.encode_image(image)  # [batch, embed_dim]
    text_features = self.encode_text(text)     # [batch, embed_dim]

    # 2. L2 归一化（让向量都落在单位球面上）
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # 3. 计算余弦相似度（缩放）
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
```

我们来拆开这个矩阵乘法 `@`：
- 假设 `batch_size = 4`。
- `image_features` 形状：`[4, 1024]`
- `text_features.t()` 形状：`[1024, 4]`
- 相乘结果 `logits_per_image`：`[4, 4]`。

**这个 4x4 的矩阵代表什么？**

|  | 文本0（狗） | 文本1（猫） | 文本2（车） | 文本3（花） |
| :--- | :--- | :--- | :--- | :--- |
| **图0（狗）** | **0.95** ✅ | 0.1 | -0.2 | 0.0 |
| **图1（猫）** | 0.05 | **0.9** ✅ | -0.1 | 0.1 |
| **图2（车）** | -0.1 | 0.0 | **0.85** ✅ | -0.1 |
| **图3（花）** | 0.0 | 0.1 | 0.0 | **0.92** ✅ |

- **对角线**（浅绿色）是正样本对（狗配狗），数值高。
- **非对角线**是负样本对（狗配猫），数值低。
- **`logits_per_image`**：每张图与所有文本的相似度（行视图）。
- **`logits_per_text`**：每段文本与所有图的相似度（列视图），直接取转置即可。

在训练时，损失函数会分别对行和列做 **Cross-Entropy Loss**，让对角线上的值尽可能大，非对角线尽可能小。

---

### 为什么 `logit_scale` 是 `Parameter`（可学习参数）？

在论文中，这个系数初始为 14.29，但允许模型在训练中微调它。
- 如果模型觉得不同模态特征分布太分散，它会调大这个系数，拉大正负样本的差距。
- 如果模型觉得梯度太陡不好收敛，它会调小系数。
- 相比固定值，**可学习的温度参数**给模型多了一个自由度的调节能力，通常能带来微弱的性能提升。

---

### 一张图总结 CLIP 的全貌

```text
[图片] → ModifiedResNet/ViT → 归一化 → ───────┐
                                               ↓
                                       [相似度矩阵] → Contrastive Loss
                                               ↑
[文本] → TokenEmbed → Transformer → 取EOT → 归一化 → ───────┘
```

你现在已经完整通读了 CLIP 的整个实现骨架。从 `simple_tokenizer` 把文本变成 ID，到 `ModifiedResNet` 和 `Transformer` 提取特征，再到最后的对比学习损失——**你已经具备了手撕多模态模型底层原理的能力**。


## conver_weights & build_model
```python

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
```
  convert_weights  — 精度转换

  遍历模型所有参数，将 Conv、Linear、MultiheadAttention 等层的权重转为 fp16，用于推理加速和显存节省。


  build_model  — 从 checkpoint 重建模型

  自动从 state_dict 推断架构参数：

  - 检测 visual.proj 是否存在 → 判断是 ViT 还是 ResNet
  - 从权重的 shape 反推：embed_dim、vision_width、vision_layers、context_length 等
  - 构建模型、转换 fp16、加载权重、设为 eval 模式
  # 计算余弦相似度矩阵
  logits = logit_scale * image_features @ text_features.T

  # 返回 (图像→文本, 文本→图像) 两个方向的 logits
  # 用于双向对比损失

  这实现了 CLIP 的核心思想：通过对比学习，让匹配的图文对在联合空间中距离更近，不匹配的拉远。