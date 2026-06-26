import os
import random
import yaml
import torch
import numpy as np


def set_seed(seed):
    """设置随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(state, filename):
    """保存检查点"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, map_location=None):
    """加载检查点"""
    return torch.load(filename, map_location=map_location)


def get_lr(optimizer):
    """获取当前学习率"""
    return optimizer.param_groups[0]['lr']