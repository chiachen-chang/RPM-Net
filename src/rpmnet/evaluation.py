"""
Evaluation utilities for RPM-Net

Includes functions for evaluating known class classification and unknown class detection
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, 
    roc_auc_score, average_precision_score, roc_curve
)


def evaluate_known_classification(model, test_known_loader, idx_to_label, config, verbose=True):
    """评估已知类分类性能"""
    if verbose:
        print("📊 评估已知类分类性能...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in test_known_loader:
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
            
            embeddings, distances, logits = model(data)
            probabilities = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 转换索引为标签
    pred_labels = [idx_to_label[idx] for idx in all_predictions]
    true_labels = [idx_to_label[idx] for idx in all_targets]
    
    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    # 提取macro指标
    macro_metrics = report['macro avg']
    macro_precision = macro_metrics['precision']
    macro_recall = macro_metrics['recall']
    macro_f1 = macro_metrics['f1-score']
    
    if verbose:
        print(f"✅ 已知类分类准确率: {accuracy:.4f}")
        print(f"✅ 已知类分类Macro指标:")
        print(f"   Precision (macro): {macro_precision:.4f}")
        print(f"   Recall (macro): {macro_recall:.4f}")
        print(f"   F1-score (macro): {macro_f1:.4f}")
        
        print("\n📋 各类别详细性能:")
        for label in idx_to_label.values():
            if label in report:
                metrics = report[label]
                print(f"  {label}:")
                print(f"    精确率: {metrics['precision']:.4f}")
                print(f"    召回率: {metrics['recall']:.4f}")
                print(f"    F1分数: {metrics['f1-score']:.4f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': pred_labels,
        'true_labels': true_labels,
        'probabilities': all_probabilities,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def evaluate_unknown_detection(model, test_known_loader, unknown_loader, config, unknown_type="validation"):
    """评估未知类检测性能 - 对每个未知类分别评估: AUROC, AUPR-IN, AUPR-OUT"""
    model.eval()
    
    # 首先获取已知类的置信度分数
    known_scores = []
    
    with torch.no_grad():
        for data, targets in test_known_loader:
            data = data.to(config.DEVICE)
            embeddings, distances, logits = model(data)
            
            # 使用最大距离作为置信度分数 (距离越大，越可能是已知类)
            max_distances = torch.max(distances, dim=1)[0]
            known_scores.extend(max_distances.cpu().numpy())
    
    known_scores = np.array(known_scores)
    
    # 获取未知类的置信度分数
    unknown_scores = []
    
    with torch.no_grad():
        for data, in unknown_loader:
            data = data.to(config.DEVICE)
            embeddings, distances, logits = model(data)
            
            # 使用最大距离作为置信度分数
            max_distances = torch.max(distances, dim=1)[0]
            unknown_scores.extend(max_distances.cpu().numpy())
    
    unknown_scores = np.array(unknown_scores)
    
    # 计算检测指标
    detection_results = compute_detection_metrics(known_scores, unknown_scores)
    
    return detection_results


def evaluate_all_unknown_classes(model, test_known_loader, test_unknown_loaders, config):
    """评估所有未知类检测性能 - 对每个未知类分别评估"""
    print("📊 评估未知类检测性能...")
    
    results = {}
    
    for unknown_class, unknown_loader in test_unknown_loaders.items():
        print(f"  评估未知类: {unknown_class}")
        
        detection_results = evaluate_unknown_detection(
            model, test_known_loader, unknown_loader, config, unknown_type="test"
        )
        
        results[unknown_class] = detection_results
        
        print(f"    AUROC: {detection_results['auroc']:.4f}")
        print(f"    AUPR-IN: {detection_results['aupr_in']:.4f}")
        print(f"    AUPR-OUT: {detection_results['aupr_out']:.4f}")
        print(f"    FPR@95%TPR: {detection_results['fpr_95']:.4f}")
    
    return results


def compute_detection_metrics(known_scores, unknown_scores):
    """计算OOD检测指标"""
    # 构造标签 (0=已知, 1=未知)
    y_true = np.concatenate([
        np.zeros(len(known_scores)),  # 已知类标记为0
        np.ones(len(unknown_scores))  # 未知类标记为1
    ])
    
    # 构造分数 (分数越高越可能是未知类)
    # 由于我们的模型输出的是"越大越像已知类"的分数，所以要取负号
    y_scores = np.concatenate([-known_scores, -unknown_scores])
    
    # 计算AUROC
    auroc = roc_auc_score(y_true, y_scores)
    
    # 计算AUPR-IN (已知类为正例)
    y_true_in = 1 - y_true  # 已知类为正例 (1=已知, 0=未知)
    aupr_in = average_precision_score(y_true_in, -y_scores)
    
    # 计算AUPR-OUT (未知类为正例)
    aupr_out = average_precision_score(y_true, y_scores)
    
    # 计算FPR@95%TPR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # 找到TPR >= 0.95的最小FPR
    valid_indices = tpr >= 0.95
    if np.any(valid_indices):
        fpr_95 = np.min(fpr[valid_indices])
    else:
        fpr_95 = 1.0  # 如果无法达到95% TPR，则FPR设为1
    
    return {
        'auroc': auroc,
        'aupr_in': aupr_in,
        'aupr_out': aupr_out,
        'fpr_95': fpr_95
    }
