# code reading 
# model.py

这个文件是clip模型的核心实现，包含图像编码器和文本编码器两部分

## bottleneck
```python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out
```
  输入 → 1×1降维卷积 → 3×3空间卷积 → 平均池化(下采样) → 1×1升维卷积 → +残差连接 → 输出

  - expansion=4: 先降维再升维 4 倍，减少计算量（经典 ResNet 设计）
  - 反锯齿下采样（Anti-aliasing）: 当 stride>1 时，不直接在卷积中 stride=2，而是在卷积后用 AvgPool2d
  做下采样。这能消除锯齿伪影，提升平移不变性
  - downsample shortcut: 当输入/输出维度不匹配时，shortcut 路径也会经过 AvgPool + 1×1 卷积来对齐维度

  防止信息丢失（抗锯齿）：
常规的 stride=2 卷积在降采样时会跳过大量像素，这类似于信号处理中的“欠采样”，容易造成高频信息的混叠（Aliasing）。先做平均池化再卷积，相当于先对信号做“低通滤波”再降采样，能更好地保留背景和细节信息，通常能带来 0.5%~1% 的精度提升。

  避免计算瓶颈（信息瓶颈）：
在瓶颈块（Bottleneck）中，inplanes（输入通道）通常很大（例如 1024），而 planes（中间通道）通常较小（例如 256）。如果直接在 1x1 卷积上设置 stride=2，意味着卷积核要在只有 1 个像素的视野内，强行将大量的输入通道信息压缩并丢弃 75% 的空间位置，这很容易造成信息瓶颈。
先池化（空间压缩）再卷积（通道变换），把“空间降采样”和“通道升维”解耦，让每一步只专注做好一件事，计算更科学。

## attentionpool2d
```python
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
```
  核心作用：替代传统的 Global Average Pooling，用 多头自注意力 来聚合空间特征。

  - 在空间维度展平后，额外拼接一个可学习的全局 query token（类似 CLS token）
  - 这个 query token 对所有空间位置做 cross-attention
  - 最终输出就是这个 query token 的注意力聚合结果
  - 比平均池化更能捕捉全局上下文关系

  x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
  这一步的目的是：flatten：把二维图片 扯成一维长条，方便当句子处理。permute：把排列顺序从 “按通道排” 改成 “按像素位置排”，让每个位置的 RGB 值聚在一起。

  我们用一个实际数值来走一遍流程：

设定：

batch_size = 2

embed_dim = 512

spacial_dim = 7（即特征图 7×7 = 49 个空间位置）

num_heads = 8

output_dim = 768（假设我们想得到 768 维向量）

数据流：

输入：从 CNN 得到的特征图 x 形状为 (2, 512, 7, 7)。

展平与置换：

x.flatten(start_dim=2) → (2, 512, 49)

.permute(2, 0, 1) → (49, 2, 512)
现在，有 49 个位置，每个位置是 512 维向量，batch 为 2。

拼接查询向量：

x.mean(dim=0, keepdim=True) → 对 49 个位置取平均 → (1, 2, 512)

torch.cat → (50, 2, 512)
第 0 个元素是平均向量，第 1~49 是原始空间特征。

加位置编码：

self.positional_embedding 形状 (50, 512)，扩展为 (50, 1, 512)，广播后与 x 相加，结果仍为 (50, 2, 512)。

多头注意力：

query = x[:1] → (1, 2, 512)

key = value = x → (50, 2, 512)

多头注意力（8 头，每个头维度 512/8=64）计算 query 与所有 key 的相似度，得到注意力权重（形状 (1, 2, 50)），然后加权平均 value，得到输出 (1, 2, 512)。

该输出经过 c_proj 线性层，从 512 映射到 768，结果仍为 (1, 2, 768)。

最终输出：

x.squeeze(0) → (2, 768)
每个样本得到一个 768 维的全局图像特征向量。

这个向量的意义：它不再是简单粗暴的全局平均，而是通过注意力机制“挑选”了最值得关注的空间区域（例如，如果图像包含一只猫，则注意力可能集中在猫的区域），因此能为后续对比学习提供更有判别力的信息。

## modifiedresnet
```python
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
```

修正后的resnet，不同点主要在于stem，这里stem不是利用7x7的大核卷积，而是用了3层3x3卷积（第一层stride=2，后两层stride=1）。第二点在于所有的下采样前面都加了avgpool。最后一层用的attentionpool2d，而非简单的avgpool


## layernorm & quickgelu
  - LayerNorm: 解决 fp16 精度问题——内部始终用 float32 计算，再转回原始精度
  - QuickGELU: GELU 激活函数的快速近似：x · σ(1.702x)，比标准 GELU 更快

## ResidualAttentionBlock
```python
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(zs * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```
  标准的 Pre-LN Transformer 结构：

  x = x + MultiheadAttention(LayerNorm(x))   ← 自注意力 + 残差
  x = x + MLP(LayerNorm(x))                  ← FFN + 残差

  MLP 的组成：Linear → QuickGELU → Linear，中间维度是 4 倍。

假设 d_model = 512，输入序列长度 seq_len = 77（CLIP 的最大上下文长度），batch = 2。

输入 x 形状：[77, 2, 512]（注意 PyTorch 的 MultiheadAttention 输入格式是 [序列长度, Batch, 特征维度]）。

Step 1：注意力分支

self.ln_1(x) → 形状不变 [77, 2, 512]

self.attn → [77, 2, 512]（多头内部会切分，最后拼回 512）

残差相加 x + ... → [77, 2, 512]

Step 2：MLP 分支

self.ln_2(x) → [77, 2, 512]

c_fc (Linear 512→2048) → [77, 2, 2048]

QuickGELU → [77, 2, 2048]（激活函数不改变形状）

c_proj (Linear 2048→512) → [77, 2, 512]

残差相加 x + ... → [77, 2, 512]

输出：形状与输入严格保持一致 [77, 2, 512]。

## transformer
```python
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
```
简单地将 N 个 ResidualAttentionBlock 串起来，形成 Transformer encoder。

## vision transformer
```python
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
```
  标准的 Vision Transformer 流程：

  图像 → Patch卷积 → Flatten → 拼接CLS token → 加位置编码 → LayerNorm → Transformer → 提取CLS → 线性投影

  - Patch Embedding: 用 conv2d(stride=patch_size) 直接将图像切成 patch
  - Class Embedding + Positional Embedding: 可学习的 CLS token 和位置编码
  - 输出: 取 CLS token 对应的输出，经过 LayerNorm 后通过投影矩阵 proj 映射到 output_dim

