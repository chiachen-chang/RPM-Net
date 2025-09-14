#!/usr/bin/env python3
"""
RPM-Net训练示例 - CICIDS2017数据集

这个脚本展示了如何使用RPM-Net在CICIDS2017数据集上进行训练和评估
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from rpmnet import (
    RPMModel, Config, load_data, prepare_data_loaders, 
    train_rpm_model, evaluate_known_classification, 
    evaluate_all_unknown_classes
)

# 设置随机种子，确保实验可重复
np.random.seed(42)
torch.manual_seed(42)


def main():
    """主函数 - 完整的RPM+Fisher实验流程"""
    print("🎯 开始RPM+Fisher在CICIDS2017数据集上的完整实验")
    print("=" * 60)
    
    # 1. 初始化配置
    config = Config(
        data_dir="./data",  # 数据目录
        result_dir="./results"  # 结果保存目录
    )
    
    # 2. 加载数据
    data, class_info = load_data(config.DATA_DIR, dataset_name="cicids2017")
    
    # 3. 准备数据加载器
    train_loader, test_known_loader, validation_unknown_loader, test_unknown_loaders, label_to_idx, idx_to_label = prepare_data_loaders(
        data, class_info, config
    )
    
    # 4. 创建模型
    input_dim = data['train'][0].shape[1]
    num_classes = len(class_info['known_classes'])
    
    print(f"🏗️ 创建RPM+Fisher模型...")
    print(f"   输入维度: {input_dim}")
    print(f"   已知类别数: {num_classes}")
    print(f"   嵌入维度: {config.EMBEDDING_DIM}")
    print(f"   Fisher损失权重: {config.FISHER_LAMBDA}")
    
    model = RPMModel(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=num_classes,
        gamma=config.GAMMA
    )
    
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练模型
    model, training_history = train_rpm_model(
        model, train_loader, test_known_loader, 
        validation_unknown_loader, test_unknown_loaders, 
        idx_to_label, config
    )
    
    # 6. 评估已知类分类性能
    known_results = evaluate_known_classification(model, test_known_loader, idx_to_label, config)
    
    # 7. 评估未知类检测性能
    unknown_results = evaluate_all_unknown_classes(model, test_known_loader, test_unknown_loaders, config)
    
    # 8. 保存最终模型
    model_file = os.path.join(config.RESULT_DIR, 'rpm_fisher_model_cicids2017_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'input_dim': input_dim,
        'hidden_dim': config.HIDDEN_DIM,
        'embedding_dim': config.EMBEDDING_DIM,
        'num_classes': num_classes,
        'gamma': config.GAMMA
    }, model_file)
    
    print(f"💾 最终模型已保存到: {model_file}")
    
    # 9. 打印最终结果摘要
    print("\n📊 最终实验结果摘要:")
    print("-" * 40)
    print(f"已知类分类准确率: {known_results['accuracy']:.4f}")
    print(f"已知类Precision (macro): {known_results['macro_precision']:.4f}")
    print(f"已知类Recall (macro): {known_results['macro_recall']:.4f}")
    print(f"已知类F1-score (macro): {known_results['macro_f1']:.4f}")
    
    if unknown_results:
        avg_auroc = np.mean([metrics['auroc'] for metrics in unknown_results.values()])
        avg_aupr_out = np.mean([metrics['aupr_out'] for metrics in unknown_results.values()])
        print(f"未知类平均AUROC: {avg_auroc:.4f}")
        print(f"未知类平均AUPR-OUT: {avg_aupr_out:.4f}")
    
    print("\n🎉 实验完成！")
    print(f"📁 所有结果已保存到: {config.RESULT_DIR}")
    
    return model, known_results, unknown_results


if __name__ == "__main__":
    # 运行完整实验
    model, known_results, unknown_results = main()
