import json
from tqdm import tqdm

def clean_caption(text):
    # 去除 <image> 标记（大小写不敏感）
    text = text.replace('<image>', '').replace('<Image>', '').replace('<IMAGE>', '')
    # 去除前后空格，并将多个连续空格合并为一个
    return ' '.join(text.split())

def convert_llava_chat_to_jsonl(chat_json_path, output_jsonl_path):
    """
    将 LLaVA-CC3M-Pretrain-595K 的 chat.json 转换为 JSONL 格式。
    自动清除 caption 中的 <image> 占位符。
    """
    
    with open(chat_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"loading {len(data)}")
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        success_count = 0
        skip_count = 0
        
        for item in tqdm(data, desc="progress"):
            image_file = item.get('image')
            if not image_file:
                skip_count += 1
                continue
            
            # 提取 gpt 的回答
            caption = None
            for conv in item.get('conversations', []):
                if conv.get('from') == 'gpt':
                    caption = conv.get('value', '')
                    break
            
            if not caption:
                skip_count += 1
                continue
            
            # 清理 caption：移除 <image> 等标记
            caption = clean_caption(caption)
            if not caption:  # 如果清理后为空，跳过
                skip_count += 1
                continue
            
            # 写入 JSONL
            json.dump({"image": image_file, "caption": caption}, f_out, ensure_ascii=False)
            f_out.write('\n')
            success_count += 1
    
    print(f"writing {success_count},skip {skip_count}")
    print(f"output: {output_jsonl_path}")

if __name__ == "__main__":
    # 请根据你的实际路径修改
    chat_json_path = "/root/autodl-tmp/my_vlm/dataset/chat.json"
    output_jsonl_path = "/root/autodl-tmp/my_vlm/dataset/annotations.jsonl"
    
    convert_llava_chat_to_jsonl(chat_json_path, output_jsonl_path)