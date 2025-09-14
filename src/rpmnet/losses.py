"""
Loss functions for RPM-Net training

Includes classification loss, margin loss, and Fisher discriminant regularization
"""

import torch
import torch.nn.functional as F


def compute_fisher_loss_optimized(embeddings, targets, num_classes):
    """优化的Fisher Loss计算 - 使用向量化操作提升效率
    
    Fisher Loss的核心思想:
    1. 类内紧凑性: 减少同类样本在嵌入空间中的散度
    2. 类间分离性: 增大不同类别样本之间的距离
    3. Fisher比率: between_scatter / within_scatter，比率越大越好
    4. 损失设计: fisher_loss = 1.0 / (1.0 + fisher_ratio)，最大化Fisher比率
    """
    device = embeddings.device
    embedding_dim = embeddings.size(1)
    batch_size = embeddings.size(0)
    
    # 获取批次中的唯一类别
    unique_targets, inverse_indices, counts = torch.unique(targets, return_inverse=True, return_counts=True)
    
    if len(unique_targets) < 2:
        # 如果批次中少于2个类别，无法计算类间散度
        return torch.tensor(0.0, device=device)
    
    # 向量化计算类别均值
    one_hot = F.one_hot(inverse_indices, num_classes=len(unique_targets)).float()  # [batch_size, num_unique_classes]
    class_sums = torch.mm(one_hot.t(), embeddings)  # [num_unique_classes, embedding_dim]
    class_means = class_sums / counts.unsqueeze(1).float()  # [num_unique_classes, embedding_dim]
    
    # 计算类内散度 (向量化)
    expanded_embeddings = embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
    expanded_class_means = class_means[inverse_indices].unsqueeze(1)  # [batch_size, 1, embedding_dim]
    within_class_diff = expanded_embeddings - expanded_class_means  # [batch_size, 1, embedding_dim]
    within_scatter = torch.sum(within_class_diff ** 2)
    
    # 计算类间散度 (向量化)
    global_mean = torch.mean(embeddings, dim=0)  # [embedding_dim]
    between_class_diff = class_means - global_mean.unsqueeze(0)  # [num_unique_classes, embedding_dim]
    weighted_between_scatter = counts.unsqueeze(1).float() * (between_class_diff ** 2)  # [num_unique_classes, embedding_dim]
    between_scatter = torch.sum(weighted_between_scatter)
    
    # 防止除零
    within_scatter = torch.clamp(within_scatter, min=1e-6)
    
    # 计算Fisher比率和损失
    fisher_ratio = between_scatter / within_scatter
    fisher_loss = 1.0 / (1.0 + fisher_ratio)
    
    return fisher_loss


def compute_rpm_loss(logits, targets, embeddings, reciprocal_points, margins, 
                    lambda_weight=0.1, fisher_lambda=0.01, class_weights=None):
    """计算RPM+Fisher组合损失函数 = 分类损失 + λ × 对抗边际损失 + fisher_lambda × Fisher损失 (向量化实现)"""
    
    # 1. 分类损失 - 标准交叉熵损失
    if class_weights is not None:
        # 使用类别权重处理不平衡问题
        # 将字典转换为张量数组，每个类别一个权重
        weight_list = []
        for i in range(logits.size(1)):  # 遍历所有类别
            if i in class_weights:
                weight_list.append(class_weights[i])
            else:
                weight_list.append(1.0)  # 默认权重为1.0
        
        weight = torch.tensor(weight_list, dtype=torch.float32).to(logits.device)
        classification_loss = F.cross_entropy(logits, targets, weight=weight)
    else:
        classification_loss = F.cross_entropy(logits, targets)
    
    # 2. 对抗边际损失 - 限制已知类在边际范围内（向量化实现）
    batch_size = embeddings.size(0)
    embedding_dim = embeddings.size(1)
    
    # 获取每个样本对应的目标类别的倒数点和边际
    target_indices = targets.view(-1, 1)  # [batch_size, 1]
    
    # 获取每个样本对应的目标类别的倒数点
    # [batch_size, 1, embedding_dim]
    target_reciprocal_points = torch.gather(
        reciprocal_points.unsqueeze(0).expand(batch_size, -1, -1),
        1,
        target_indices.unsqueeze(-1).expand(batch_size, 1, embedding_dim)
    ).squeeze(1)  # [batch_size, embedding_dim]
    
    # 获取每个样本对应的目标类别的边际
    # [batch_size, 1]
    target_margins = torch.gather(
        margins.unsqueeze(0).expand(batch_size, -1),
        1,
        target_indices
    ).squeeze(1)  # [batch_size]
    
    # 计算样本到目标类别倒数点的欧几里得距离
    euclidean_distances = torch.sum((embeddings - target_reciprocal_points) ** 2, dim=1) / embedding_dim
    
    # 应用边际约束: max(distance - margin, 0)
    margin_constraints = torch.clamp(euclidean_distances - target_margins, min=0)
    
    # 计算平均边际损失
    margin_loss = margin_constraints.mean()
    
    # 3. Fisher Loss - 促进类内紧凑和类间分离
    fisher_loss = compute_fisher_loss_optimized(embeddings, targets, logits.size(1))
    
    # 总损失 = 分类损失 + λ × 对抗边际损失 + fisher_lambda × Fisher损失
    total_loss = classification_loss + lambda_weight * margin_loss + fisher_lambda * fisher_loss
    
    return total_loss, classification_loss, margin_loss, fisher_loss
