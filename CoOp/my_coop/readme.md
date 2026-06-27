  CoOp（Context Optimization）的最小化复现——用可学习的连续 prompt
  向量替代手工设计的文本提示，实现视觉-语言模型的高效少样本适配。

  > **论文**: [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) (IJCV'22 / CVPR'22)

  > **原版代码**: [KaiyangZhou/CoOp](https://github.com/KaiyangZhou/CoOp)

  本项目从零复现了 CoOp 的核心思想，不依赖 Dassl.pytorch，全部核心代码约 200 行。

  ---

  ## 特点

  - **轻量自包含** — 不依赖 Dassl.pytorch，纯 PyTorch 实现
  - **模块化设计** — models / data / training 清晰分离
  - **即插即用** — 任何 ImageFolder 格式的数据集直接跑
  - **少样本友好** — 每类 16 张图即可达到高准确率

  ---

  ## 项目结构
```bash
  my_coop/
  ├
  │
  │── models/
  │   ├── init.py
  │   ├── clip_utils.py          # 加载 CLIP 并拆解组件
  │   ├── prompt_learner.py      # 可学习的 context vectors（CoOp 核心）
  │   └── custom_clip.py         # 将 CLIP + PromptLearner 组装为 CoOp 模型
  │── data/
  │   ├── init.py
  │   └── dataset.py             # 基于 ImageFolder 的数据加载器
  ├
  │── train.py                   # 训练入口
  ├── scripts/
  │   └── prepare_cifar10_fewshot.py # 从 tar.gz 生成 CIFAR-10 少样本数据集
  ├── outputs/                       # 存放 checkpoint 和日志
  ├── requirements.txt
  └── README.md
```
  ---

  ## 安装

  ```bash
  # 1. 安装 PyTorch
  pip install torch torchvision

  # 2. 安装 CLIP
  pip install git+https://github.com/openai/CLIP.git
  # 如果 GitHub 访问不了：pip install clip

  # 3. 克隆项目
  git clone https://github.com/yourusername/my_coop.git
  cd my_coop

  ---
  快速开始

  1. 准备数据集（ImageFolder 格式）

  数据集目录结构：

  data/数据集名/
  ├── 类别A/
  │   ├── 001.jpg
  │   └── 002.jpg
  ├── 类别B/
  │   ├── 001.jpg
  │   └── 002.jpg
  └── 类别C/
      └── 001.jpg

  子文件夹名就是类别名，会自动传给 PromptLearner。

  如果是 CIFAR-10，用项目自带的脚本生成：

  python scripts/prepare_cifar10_fewshot.py \
      --tar-path /path/to/cifar-10-python.tar.gz \
      --output-dir ./data/cifar10_fewshot \
      --shots 16

  2. 训练

  python tools/train.py \
      --data-root ./data/cifar10_fewshot \
      --backbone ViT-B/16 \
      --n-ctx 16 \
      --lr 0.002 \
      --epochs 50 \
      --batch-size 16

  CIFAR-10 16-shot 预期输出：
  Epoch   1/50  Loss: 2.4627  Acc:  5.62%
  Epoch   5/50  Loss: 1.4865  Acc: 43.75%
  Epoch  10/50  Loss: 0.5301  Acc: 94.38%
  Epoch  17/50  Loss: 0.1673  Acc: 100.0%
  ...
  Done.

  ---
  命令行参数
```bash
  ┌──────────────┬───────────┬──────────────────────────────────────────────────┐
  │     参数     │  默认值   │                       说明                       │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --data-root  │ 必填      │ 数据集根目录（ImageFolder 格式）                 │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --backbone   │ ViT-B/16  │ CLIP 骨干网络（ViT-B/16, ViT-B/32, RN50, RN101） │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --n-ctx      │ 16        │ 可学习 context token 的数量 M                    │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --lr         │ 0.002     │ 学习率                                           │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --epochs     │ 50        │ 训练轮数                                         │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --batch-size │ 32        │ 批次大小                                         │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --device     │ cuda      │ 设备（cuda 或 cpu）                              │
  ├──────────────┼───────────┼──────────────────────────────────────────────────┤
  │ --output-dir │ ./outputs │ checkpoint 输出目录                              │
  └──────────────┴───────────┴──────────────────────────────────────────────────┘
```
---
  原理

  背景

  CLIP 通过计算图像特征和文本特征的余弦相似度来分类。文本特征是把手工 prompt 如 "a photo of a [类别]" 送入 text
  encoder 得到的。

  CoOp 做法

  用 M 个可学习的连续向量 替代手工设计的 prompt 中的上下文词：

  手工 prompt:   [SOS] a photo of a [类别] [EOS]
  CoOp prompt:   [SOS] [v₁] [v₂] ... [vₘ] [类别] [EOS]
                       ↑ 可学习的 context vectors

  - 每个 vᵢ 是一个 d 维向量（d = CLIP embedding 维度，ViT-B/16 为 512）
  - 所有 M 个 context vector 在全部类别间共享
  - 训练时只更新这 M × d 个参数
  - CLIP 的 image encoder、text encoder、token embedding 全部冻结

  核心实现

  前向传播:
    image → image_encoder(冻结) → image_features         (B, d)
    类别token + ctx_vectors → text_encoder(冻结) → text_features  (n_cls, d)
    logits = image_features @ text_featuresᵀ × exp(logit_scale)   (B, n_cls)
    loss = CrossEntropyLoss(logits, labels)

  反向传播:
    只有 PromptLearner.ctx（M×d 的 Parameter）接收梯度

  ---
  设计选择

  - 无 CSC（Class-Specific Context） — 所有类别共享一组 context vector，实现更简洁
  - Context 在前，类别 token 在末尾 — 遵循原论文的默认设置，效果最好
  - 无 Meta-Net — 本项目仅实现基础版 CoOp，不含 CoCoOp
  - 不依赖 Dassl — 原版依赖 Dassl.pytorch，本项目完全自包含
  - ImageFolder 格式 — 直接用 PyTorch 内置的 ImageFolder，无需额外数据集类

  ---
  参考资料

  - Learning to Prompt for Vision-Language Models (https://arxiv.org/abs/2109.01134) — Zhou et al., IJCV 2022
  - OpenAI CLIP (https://github.com/openai/CLIP)
  - CoOp 原版实现 (https://github.com/KaiyangZhou/CoOp)

  ---
  License

  MIT