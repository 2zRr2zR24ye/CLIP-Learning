# Code Reading


## example:
```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```



## clip.load()函数
用来加载模型，写的很复杂但实质上后面的大部分代码是对JIT的处理。返回：“model, _transform(model.visual.input_resolution)”模型和预处理。该函数的关键点在于build_model函数。

## clip.tokenize()函数
Returns the tokenized representation of given input string(s)
输入为："text"或["text1", "text2"]

无论输入文本有多长，最终输出的长度均为77，超出的地方被截断（同时最后一位成为eot），不足的情况则补零（因为一开始建立了一个新的全零矩阵）。

返回为：result，一个二位整数tensor[number_of_input_strings, context_length]
倘若我现在传了3条文本那么，输出tensor为[3, 77]


