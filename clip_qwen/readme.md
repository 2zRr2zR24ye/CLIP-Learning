# 基于 CLIP + Qwen 的视觉语言模型，使用 Projector 连接视觉和语言。

## 项目结构
- `clip_qwen/` - 模型核心代码
- `data/` - 数据加载
- `scripts/` - 辅助脚本
- `outputs/` - 训练产出
- `train.py` - 训练入口
- `inference.py` - 推理入口

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python train.py

# 推理
python inference.py
```
# 注意事项
## 1.需要注意的是，本项目是基于*LLaVA-CC3M-Pretrain-595K*数据集训练，原生的数据集json结构为：
```bash
  {
    "id": "GCC_train_002582585",
    "image": "GCC_train_002582585.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Provide a brief description of the given image.\n<image>"
      },
      {
        "from": "gpt",
        "value": "olive oil is a healthy ingredient used liberally ."
      }
    ]
  }
```
这与本项目需要的结构不符合，因此，请运行./scripts/convert_llava_2_jsonl.py将结构转换至本项目所需的jsonl：
```bash
{"image": "GCC_train_002582585.jpg", "caption": "olive oil is a healthy ingredient used liberally ."}
...
...
```

## 2."<image>"的生命周期
- 在数据集中，利用./scripts/convert_llava_2_jsonl.py去除了原始数据集中的"<image>",在./data/datasets.py中加载数据集时：
```python
        conversation = [
            {"role": "user", "content": f"{self.image_token}\n请描述这张图片。"},
            {"role": "assistant", "content": caption}
        ]
```
可以看到，"<image>"作为一个占位符被硬解码添加到了输入数据的开头。

- 分词阶段。代码将"<image>"变成了数字ID，在train.py中：
```python
if '<|image|>' not in llm_tokenizer.get_vocab():
    num_added = llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|image|>']})
    print(f"add {num_added} special token: <|image|>")
    model.llm.resize_token_embeddings(len(llm_tokenizer))
```
手动将"<image>"加入qwen的词汇表，分配了唯一的数字ID。这样一来，"<image>"在分词过程中不会被分割开，而是作为一个独立的数字（不再是一串字符）

- 模型前向传播阶段。./clip_qwen/model.py:
```python
            pos = (input_ids[i] == image_token_ids).nonzero(as_tuple = True)[0]
            inputs_embeds[i, pos:pos + projed_features.shape[1]] = projed_features[i]
```
inputs_embeds[pos:pos+50] 全部变成了 CLIP 提取的图像特征（经过 Projector 翻译）。替换前：模型看到的是“一个占位符单词”。替换后：模型看到的是“一张图片的 50 个语义块”。

- 训练推理阶段。经过上面的替换，模型看到的输入序列不再是："[你是一个助手] [用户] <|image|> 请描述...", 而是："[你是一个助手] [用户] [图像块1] [图像块2] ... [图像块50] 请描述..."。qwen 根本不知道“此处曾有 <|image|> 这个单词”，它只看到 50 个连续的数字向量，就像看到了 50 个“新造的视觉单词”。于是它就像处理普通文本一样处理这些向量，生成描述。

*这就是 "<|image|>" 的作用：它是一个hook，一个方便在嵌入层层面“偷梁换柱”的标记*

- *缺点*：这样设计会有一定的缺点：覆盖文本：用 50 个向量覆盖了 pos 位置后的 49 个文本 token，这些被覆盖的文字永远丢失了。
优化的方案为：占位符 + 动态插入（torch.cat），但是，代码复杂，需要动态调整 attention_mask 和 batch 内序列长度。

# 测试结果
用项目下的cat.jpg输入模型3次，可以得到如下的结果：
```bash
1.the isthe cat with a small body and white fur, with black stripes on the legs and face. 
2.The Image is a feline, with a medium-sized body covered in short, soft fur. The cat has an almond-shaped face with large round eyes that are wide apart and expressive. Its nose is slightly upturned, giving it a playful appearance. It has two small ears set high on its head, one on each side of its skull. The tail is long and thin, ending in a point, and moves gracefully as the cat walks. The paws are small but sturdy, with five toes on each foot, which helps them navigate through their environment easily. The overall silhouette is sleek and elegant, making this feline a perfect representation
3.The photographed image is of a female cat with a soft and fluffy coat, having long whiskers and expressive eyes. The body of the cat is in shape, with prominent ribs and a well-defined tail. She appears to be relaxed and contented, with her head slightly tilted and her ears forward. 
```


