# Learning Transferable Visual Models From Natural Language Supervision

## Abstract
在原论文中，作者提到了CLIP的zero-shot能力--“After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks.”

疑问1：为什么具有zero-shot transfer的能力？（这将在读论文的过程中给重点关注，这里我给出gpt的答案）

gpt-5：CLIP 之所以具有 zero-shot 能力，是因为它在大规模图文配对数据上通过对比学习训练，学会了将图像和文本映射到同一个语义空间中，使匹配的图文特征更接近、不匹配的更远离。因此，在测试时它不需要为某个新任务单独训练分类器，而是可以直接把候选类别写成文本描述，如 “a photo of a cat” 或 “a photo of a dog”，再计算这些文本与输入图像的相似度，选择最相近的文本作为预测结果。本质上，CLIP 将传统的固定类别分类问题转化为了图像与语言之间的语义匹配问题，这正是它能够进行 zero-shot 推理的关键。

## Approach

"""Recent work in contrastive representation learning for images has found that contrastive objectives can learn better representations than their equivalent predictive objective"""

预测目标 (Predictive Objective)：
在CLIP论文的上下文中，这指的是类似图像描述生成的任务。模型看到一张图片，需要预测（生成）出与之配对的完整文字描述。这是一个非常困难的任务，因为文字描述的方式无穷无尽，模型需要精确地生成每一个词。

对比目标 (Contrastive Objective)：
这是CLIP采用的方法。模型不再去费力生成整个句子，而是学习一个共享的嵌入空间，只需判断哪张图配哪段文字。具体做法是：在一个包含N对（图, 文）的批次中，让模型最大化N对正确配对的相似度，同时最小化其他 N² - N 对错误配对的相似度。这是一个更简单的“找不同”或“连连看”任务。

"""Noting these findings, we explored training a system to solve the potentially easier proxy task of predicting only which text as a whole is paired with which image and not the exact words of that text. Starting with the same bag-of-words encoding baseline, we swapped the predictive objective for a contrastive objective in Figure 2 and observed a further 4x efficiency improvement in the rate of zero-shot transfer to ImageNet.
"""
也就是说CLIP是将整段文字与图像进行配对，而非文本中的某个词。

![伪代码](Figure\Numpy-like-pseudocode.png)

### Encoder的选择
作者在论文中选择了ResNet-50和Vision Transformer作为Image Encoder，同时均对其做出了一些修改。

选择Transformer作为Text Encoder。