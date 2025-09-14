"""
Data utilities for RPM-Net

Includes data loading and preprocessing functions
"""

import numpy as np
import pickle
import os
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir, dataset_name="unsw"):
    """加载处理好的数据集"""
    print(f"🔄 正在加载{dataset_name.upper()}数据集...")
    
    try:
        # 加载训练数据
        X_train = np.load(os.path.join(data_dir, f'X_train_{dataset_name}.npy'))
        y_train = np.load(os.path.join(data_dir, f'y_train_{dataset_name}.npy'))
        
        # 加载已知类测试数据
        X_test_known = np.load(os.path.join(data_dir, f'X_test_known_{dataset_name}.npy'))
        y_test_known = np.load(os.path.join(data_dir, f'y_test_known_{dataset_name}.npy'))
        
        # 加载验证未知类数据
        X_validation_unknown = np.load(os.path.join(data_dir, f'X_validation_unknown_{dataset_name}.npy'))
        y_validation_unknown = np.load(os.path.join(data_dir, f'y_validation_unknown_{dataset_name}.npy'))
        
        # 加载测试未知类数据
        X_test_unknown = np.load(os.path.join(data_dir, f'X_test_unknown_{dataset_name}.npy'))
        y_test_unknown = np.load(os.path.join(data_dir, f'y_test_unknown_{dataset_name}.npy'))
        
        # 加载类别信息
        with open(os.path.join(data_dir, f'class_info_{dataset_name}.pkl'), 'rb') as f:
            class_info = pickle.load(f)
        
        print(f"✅ 数据加载成功!")
        print(f"   训练集: {X_train.shape}")
        print(f"   已知类测试集: {X_test_known.shape}")
        print(f"   验证未知类: {X_validation_unknown.shape}")
        print(f"   测试未知类: {X_test_unknown.shape}")
        print(f"   特征数量: {X_train.shape[1]}")
        print(f"   已知类别: {class_info['known_classes']}")
        print(f"   验证未知类别: {class_info.get('validation_unknown_classes', [])}")
        print(f"   测试未知类别: {class_info['test_unknown_classes']}")
        
        return {
            'train': (X_train, y_train),
            'test_known': (X_test_known, y_test_known),
            'validation_unknown': (X_validation_unknown, y_validation_unknown),
            'test_unknown': (X_test_unknown, y_test_unknown)
        }, class_info
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise e


def prepare_data_loaders(data, class_info, config):
    """准备PyTorch数据加载器"""
    print("🔄 准备数据加载器...")
    
    # 创建标签到索引的映射
    known_classes = class_info['known_classes']
    label_to_idx = {label: idx for idx, label in enumerate(known_classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"📋 标签映射: {label_to_idx}")
    
    # 准备训练数据
    X_train, y_train = data['train']
    y_train_idx = np.array([label_to_idx[label] for label in y_train])
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train_idx)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 准备已知类测试数据
    X_test_known, y_test_known = data['test_known']
    y_test_known_idx = np.array([label_to_idx[label] for label in y_test_known])
    
    test_known_dataset = TensorDataset(
        torch.FloatTensor(X_test_known),
        torch.LongTensor(y_test_known_idx)
    )
    test_known_loader = DataLoader(test_known_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 准备验证集未知类数据
    X_validation_unknown, y_validation_unknown = data['validation_unknown']
    validation_unknown_dataset = TensorDataset(torch.FloatTensor(X_validation_unknown))
    validation_unknown_loader = DataLoader(validation_unknown_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 为每个测试未知类创建单独的加载器
    print("  创建测试未知类数据加载器...")
    X_test_unknown, y_test_unknown = data['test_unknown']
    
    test_unknown_loaders = {}
    
    # 获取所有测试未知类别
    test_unknown_classes = class_info['test_unknown_classes']
    
    for unknown_class in test_unknown_classes:
        # 找到该未知类的样本
        mask = y_test_unknown == unknown_class
        X_unknown_class = X_test_unknown[mask]
        
        if len(X_unknown_class) > 0:
            unknown_dataset = TensorDataset(torch.FloatTensor(X_unknown_class))
            unknown_loader = DataLoader(unknown_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            test_unknown_loaders[unknown_class] = unknown_loader
            print(f"   类别 {unknown_class}: {len(X_unknown_class)} 样本")
    
    print(f"✅ 数据加载器准备完成!")
    print(f"   训练集大小: {len(train_loader.dataset)}")
    print(f"   已知类测试集大小: {len(test_known_loader.dataset)}")
    print(f"   验证集未知类大小: {len(validation_unknown_loader.dataset)}")
    print(f"   测试未知类种类数: {len(test_unknown_loaders)}")
    
    return train_loader, test_known_loader, validation_unknown_loader, test_unknown_loaders, label_to_idx, idx_to_label


def compute_class_weights(train_loader, idx_to_label):
    """计算类别权重处理不平衡问题"""
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    y_train = np.array(all_labels)
    
    # 获取训练集中每个类别的样本数量
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    
    # 计算类别权重 - 样本数量越少，权重越大
    weights = np.max(class_counts) / class_counts
    
    # 归一化权重
    weights = weights / np.sum(weights) * len(unique_classes)
    
    # 创建类别索引到权重的映射
    class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    
    print("📊 类别权重:")
    for cls, weight in class_weights.items():
        print(f"   类别 {cls} ({idx_to_label[cls]}): {weight:.4f}")
    
    return class_weights
