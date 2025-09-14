"""
RPM-Net Model Implementation

Reciprocal Point MLP Network for Open Set Recognition in Network Security Threat Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPMModel(nn.Module):
    """
    RPM模型实现
    
    核心思想:
    1. 特征提取器: 将输入特征映射到嵌入空间
    2. 倒数点(Reciprocal Points): 每个已知类对应一个倒数点，代表"非该类"的空间
    3. 分类器: 基于样本与倒数点距离进行分类
    4. 对抗边际约束: 限制开放空间，防止未知样本被错误分类
    """
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_classes, gamma=1.0):
        super(RPMModel, self).__init__()
        
        self.num_classes = num_classes
        self.gamma = gamma
        
        # 特征提取器 - 简单的全连接网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 倒数点 - 每个已知类对应一个可学习的倒数点
        self.reciprocal_points = nn.Parameter(torch.randn(num_classes, embedding_dim))
        
        # 边际参数 - 用于对抗边际约束
        self.margins = nn.Parameter(torch.ones(num_classes))
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # 倒数点初始化为小的随机值
        nn.init.normal_(self.reciprocal_points, 0, 0.1)
        nn.init.constant_(self.margins, 1.0)
    
    def forward(self, x):
        """前向传播"""
        # 提取特征嵌入
        embeddings = self.feature_extractor(x)  # [batch_size, embedding_dim]
        
        # 计算到倒数点的距离
        distances = self.compute_distances(embeddings)  # [batch_size, num_classes]
        
        # 基于距离计算分类logits (距离越大，属于该类的概率越大)
        logits = self.gamma * distances  # [batch_size, num_classes]
        
        return embeddings, distances, logits
    
    def compute_distances(self, embeddings):
        """计算嵌入向量到倒数点的距离 - 使用欧几里得距离 + 余弦相似度的组合"""
        batch_size = embeddings.size(0)
        
        # 计算欧几里得距离
        embeddings_expanded = embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        reciprocal_expanded = self.reciprocal_points.unsqueeze(0)  # [1, num_classes, embedding_dim]
        
        euclidean_distances = torch.sum((embeddings_expanded - reciprocal_expanded) ** 2, dim=2)
        euclidean_distances = euclidean_distances / embeddings.size(1)  # 归一化
        
        # 计算余弦相似度
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        reciprocal_norm = F.normalize(self.reciprocal_points, p=2, dim=1)
        
        cosine_similarity = torch.mm(embeddings_norm, reciprocal_norm.t())
        
        # 组合距离 (欧几里得距离 - 余弦相似度)
        distances = euclidean_distances - cosine_similarity
        
        return distances
    
    def predict(self, x, threshold=None):
        """
        预测函数，支持已知类分类和未知类检测
        
        Args:
            x: 输入特征
            threshold: 未知类检测阈值，如果为None则返回分类结果
        
        Returns:
            predictions: 分类结果
            max_distances: 最大距离分数
            is_unknown: 是否为未知类（当threshold不为None时）
        """
        self.eval()
        with torch.no_grad():
            embeddings, distances, logits = self.forward(x)
            
            # 获取预测类别
            _, predicted = torch.max(logits, 1)
            
            # 计算最大距离（用于未知类检测）
            max_distances = torch.max(distances, 1)[0]
            
            if threshold is not None:
                # 检测未知类
                is_unknown = max_distances < threshold
                return predicted, max_distances, is_unknown
            else:
                return predicted, max_distances
