import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModelForCausalLM
from proj import SimpleProj

class VLMModel(nn.Module):
    def __init__(self, clip_model_name: str, llm_model_name: str, device: str = "cuda"):
        super().__init__()
        self.device = device

        #加载clip的vision编码器并冻结
        self.vision_encoder = CLIPModel.from_pretrained(clip_model_name).to(device)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()
        
        #加载llm模型并冻结
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype = torch.bfloat16,
            device_map = "auto"
        )
        for param in self.llm.parameters():
            param.requires_grad = False

        #初始化proj
        clip_dim = self.vision_encoder.config.vision_config.hidden_size
        llm_dim = self.llm.config.hidden_size
        self.projector = SimpleProj(clip_dim, llm_dim).to(device)

    def forward(self, pixel_values, input_ids, attn_mask, image_token_ids):
        #获取视觉features
        with torch.no_grad():
            vision_outputs = self.vision_encoder.vision_model(pixel_values)
            image_features = vision_outputs.last_hidden_state

        #投影
        projed_features = self.projector(image_features)

        #获取文本embed
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        

        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            print("input_ids shape:", input_ids.shape)
            print("input_ids[i] shape:", input_ids[i].shape)
            print("image_token_ids:", image_token_ids, type(image_token_ids))
            pos = (input_ids[i] == image_token_ids).nonzero(as_tuple = True)[0]
            inputs_embeds[i, pos:pos + projed_features.shape[1]] = projed_features[i]

        outputs = self.llm(
            inputs_embeds = inputs_embeds,
            attention_mask = attn_mask,
            labels = input_ids,
        )

        return outputs

        