# *用 transformers 实现 CoOp 的失败原因总结*
## 背景
我尝试用 Hugging Face 的 transformers 库加载 CLIP，然后自己写一个 CoOp 训练脚本。之前看 CLIP 源码是用 OpenAI 官方库学的，但这次想换 transformers 试试，结果花了很多时间在修 bug 上，最后还是放弃了，准备换回官方库。

主要问题
1. 两套 CLIP 的接口完全不一样
OpenAI 官方库的 CLIP 和 transformers 的 CLIP 虽然加载的是同一个模型权重，但代码结构差别很大。我之前学习的时候用的是官方库，记住了 transformer、positional_embedding、text_projection 这些属性名，但在 transformers 里这些东西要么名字不同，要么藏在别的地方。

比如官方库可以直接用 model.transformer，但 transformers 里是 model.text_model.encoder。官方库有 text_projection，transformers 里这个属性在 text_model 下面没有，得从更上层拿。

2. transformers 的文本编码器不支持直接传入嵌入向量
CoOp 的核心就是把 prompt 词向量替换成可学习的向量，所以需要把嵌入向量直接喂给文本编码器。但 transformers 的 CLIPTextTransformer 的 forward 只接受 input_ids，不接受 inputs_embeds 参数。

我尝试绕过这个问题，直接调用内部的 encoder 模块，手动加位置编码、手动取 EOS 特征、手动做投影，但每一步都有小问题。attention_mask 的维度格式不匹配，projection 属性找不到，位置编码长度对不上，这些问题一个一个冒出来。

3. 每个小问题都要花时间去翻源码
因为 transformers 的文档没有讲这么底层的东西，每次报错都要去翻 transformers 的源码，看 CLIPTextTransformer 到底有哪些属性，encoder 的 forward 接受什么参数，position_embedding 存在哪里。太花时间了，而且感觉在做重复劳动。

4. 本质上是在重新实现 transformer 的前向传播
绕了一圈，我其实是在手动实现 CLIP 文本编码器的 forward 逻辑，包括加位置编码、调用 encoder、取 EOS、投影。这本身就是 CLIP 内部已经封装好的东西，我为了绕过 inputs_embeds 不支持的限制，把这些代码重新写了一遍。但重写的过程很难保证和原始实现完全一致，出问题也很难排查。

经验教训
如果想做 CoOp 这种需要自定义输入的任务，OpenAI 官方库比 transformers 更合适，因为它暴露了更多底层接口，更容易插入自定义逻辑。

transformers 适合标准任务，比如加载模型做推理、做微调，但不适合需要修改模型内部数据流的场景。

如果非要在一个库里把一件事情做到底，就不要在两个库之间切换。我之前用官方库学的 CLIP，切换到 transformers 后认知成本很高。

下次遇到类似情况，应该先评估这个库是不是适合这个任务，而不是上来就写代码。如果官方论文用的就是官方库，直接沿用可以减少很多不必要的麻烦。

# *后续计划*
改用 OpenAI 官方 clip 库重新实现，CoOp 官方代码也是这样做的，沿着他们的路子走会更顺。

# *后续尝试使用agent抢救也失败了。更改后的代码之间的接口不了解，难以debug*