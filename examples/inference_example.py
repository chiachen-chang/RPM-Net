#!/usr/bin/env python3
"""
RPM-Net推理示例

这个脚本展示了如何加载训练好的RPM-Net模型进行推理和未知类检测
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from rpmnet import RPMModel, Config


def load_trained_model(model_path, device='cpu'):
    """加载训练好的模型"""
    print(f"🔄 正在加载模型: {model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型实例
    model = RPMModel(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        embedding_dim=checkpoint['embedding_dim'],
        num_classes=checkpoint['num_classes'],
        gamma=checkpoint['gamma']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ 模型加载成功!")
    return model, checkpoint


def predict_single_sample(model, sample, idx_to_label, threshold=None):
    """对单个样本进行预测"""
    # 确保输入是torch张量
    if not isinstance(sample, torch.Tensor):
        sample = torch.FloatTensor(sample)
    
    # 如果输入是单个样本，添加batch维度
    if len(sample.shape) == 1:
        sample = sample.unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        predicted, max_distances = model.predict(sample, threshold=threshold)
        
        # 转换预测索引为类别标签
        predicted_labels = [idx_to_label[idx.item()] for idx in predicted]
        
        if threshold is not None:
            is_unknown = max_distances < threshold
            return predicted_labels[0], max_distances[0].item(), is_unknown[0].item()
        else:
            return predicted_labels[0], max_distances[0].item()


def detect_unknown_class(known_scores, unknown_score, threshold_percentile=5):
    """
    检测样本是否为未知类
    
    参数:
    - known_scores: 已知类样本的得分分布
    - unknown_score: 待检测样本的得分
    - threshold_percentile: 阈值百分位数，默认使用5%分位数
    
    返回:
    - is_unknown: 布尔值，True表示样本可能是未知类
    - threshold: 使用的阈值
    """
    threshold = np.percentile(known_scores, threshold_percentile)
    is_unknown = unknown_score < threshold
    
    return is_unknown, threshold


def main():
    """主函数 - 推理示例"""
    print("🔍 RPM-Net推理示例")
    print("=" * 40)
    
    # 1. 加载训练好的模型
    model_path = "./results/rpm_fisher_model_final.pth"  # 请根据实际路径修改
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model, checkpoint = load_trained_model(model_path, device)
        idx_to_label = checkpoint['idx_to_label']
        label_to_idx = checkpoint['label_to_idx']
        
        print(f"📋 已知类别: {list(idx_to_label.values())}")
        print(f"💻 使用设备: {device}")
        
    except FileNotFoundError:
        print("❌ 模型文件未找到，请先训练模型或检查路径")
        print("💡 提示: 运行 python examples/train_unsw.py 来训练模型")
        return
    
    # 2. 创建示例数据（实际使用时应该加载真实数据）
    print("\n📊 创建示例数据...")
    input_dim = checkpoint['input_dim']
    
    # 创建一些随机样本作为示例
    np.random.seed(42)
    sample_data = np.random.randn(5, input_dim)  # 5个样本
    
    print(f"   样本形状: {sample_data.shape}")
    
    # 3. 进行预测
    print("\n🔮 进行预测...")
    
    for i, sample in enumerate(sample_data):
        print(f"\n样本 {i+1}:")
        
        # 分类预测
        predicted_label, max_distance = predict_single_sample(
            model, sample, idx_to_label
        )
        
        print(f"  预测类别: {predicted_label}")
        print(f"  最大距离: {max_distance:.4f}")
        
        # 未知类检测（需要已知类得分分布来计算阈值）
        # 这里我们使用一个简化的方法
        if max_distance < 0.5:  # 简化的阈值
            print(f"  检测结果: 可能是未知类")
        else:
            print(f"  检测结果: 已知类")
    
    # 4. 批量预测示例
    print("\n📦 批量预测示例...")
    
    # 将样本转换为张量
    sample_tensor = torch.FloatTensor(sample_data)
    
    with torch.no_grad():
        embeddings, distances, logits = model(sample_tensor)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits, 1)
        
        print(f"   预测结果: {[idx_to_label[idx.item()] for idx in predicted]}")
        print(f"   最大距离: {torch.max(distances, dim=1)[0].numpy()}")
        print(f"   预测概率: {torch.max(probabilities, dim=1)[0].numpy()}")
    
    # 5. 模型信息
    print("\n📋 模型信息:")
    print(f"   输入维度: {checkpoint['input_dim']}")
    print(f"   隐藏层维度: {checkpoint['hidden_dim']}")
    print(f"   嵌入维度: {checkpoint['embedding_dim']}")
    print(f"   类别数量: {checkpoint['num_classes']}")
    print(f"   Gamma参数: {checkpoint['gamma']}")
    
    print("\n✅ 推理示例完成!")


if __name__ == "__main__":
    main()
