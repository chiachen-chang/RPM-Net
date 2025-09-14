"""
Configuration class for RPM-Net experiments
"""

import os
import torch


class Config:
    """实验配置类 - 把所有重要参数放在这里，方便调整"""
    
    def __init__(self, data_dir="./data", result_dir="./results"):
        # 文件路径配置
        self.DATA_DIR = data_dir
        self.RESULT_DIR = result_dir
        
        # 模型参数
        self.HIDDEN_DIM = 256       # 隐藏层维度
        self.EMBEDDING_DIM = 64     # 嵌入向量维度
        self.LEARNING_RATE = 0.001  # 学习率
        self.BATCH_SIZE = 256       # 批大小
        self.EPOCHS = 50            # 训练轮数
        
        # RPM特定参数
        self.LAMBDA = 1             # 对抗边际损失权重
        self.GAMMA = 1.0            # 距离-概率转换参数
        
        # Fisher Loss特定参数
        self.FISHER_LAMBDA = 1      # Fisher损失权重
        
        # 设备配置
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确保结果目录存在
        os.makedirs(self.RESULT_DIR, exist_ok=True)
        print(f"📁 结果将保存到: {self.RESULT_DIR}")
        print(f"💻 使用设备: {self.DEVICE}")
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def to_dict(self):
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
