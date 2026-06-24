import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor

class VLMDataset(Dataset):
    def __init__(self, data_dir, annotation_file, clip_model_name, llm_tokenizer, max_length=256):
        self.data_dir = data_dir
        self.annotations = self.load_annotations(annotation_file)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length
        self.image_token = "<|image|>"

    def load_annotations(self, file_path):
        #  JSONL 格式，每行是一个字典 {"image": "1.jpg", "caption": "一只猫"}
        import json
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                annotations.append(json.loads(line))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.data_dir, ann['image'])
        caption = ann['caption']

        # 1. 处理图像
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.clip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # 2. 处理文本 (构建对话格式)
        # 注意：这里使用了 <|image|> 作为视觉特征的占位符
        conversation = [
            {"role": "user", "content": f"{self.image_token}\n请描述这张图片。"},
            {"role": "assistant", "content": caption}
        ]
        # 应用 Qwen 的聊天模板
        prompt = self.llm_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        # 将文本转为 token IDs
        text_inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption
        }
