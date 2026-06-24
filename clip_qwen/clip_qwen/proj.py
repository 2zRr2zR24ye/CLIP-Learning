import torch.nn as nn

class SimpleProj(nn.Module):
    def __init__(self, input_dim=768, output_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),  # 升维
            nn.GELU(),                   # 非线性激活
            nn.Linear(2048, output_dim)  # 降维到目标维度
        )
        # 更好的初始化：使用 xavier 或 kaiming
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.mlp(x)