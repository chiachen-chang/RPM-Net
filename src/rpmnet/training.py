"""
Training utilities for RPM-Net

Includes training loop and related functions
"""

import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from .losses import compute_rpm_loss
from .evaluation import evaluate_known_classification, evaluate_unknown_detection
from .data_utils import compute_class_weights


def train_rpm_model(model, train_loader, test_known_loader, validation_unknown_loader, 
                   test_unknown_loaders, idx_to_label, config):
    """训练RPM+Fisher模型，并在每个epoch评估验证集和测试集的性能"""
    print("🚀 开始训练RPM+Fisher模型...")
    
    # 从idx_to_label创建label_to_idx映射
    label_to_idx = {label: idx for idx, label in idx_to_label.items()}
    
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # 计算类别权重处理不平衡问题
    class_weights = compute_class_weights(train_loader, idx_to_label)
    
    training_history = {
        'total_loss': [],
        'classification_loss': [],
        'margin_loss': [],
        'fisher_loss': [],
        'accuracy': [],
        'validation_aupr_out': [],
        'test_aupr_out': [],
        'test_accuracy': [],
        'test_macro_f1': []
    }
    
    model.train()
    
    for epoch in range(config.EPOCHS):
        epoch_total_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_margin_loss = 0.0
        epoch_fisher_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
            
            # 前向传播
            embeddings, distances, logits = model(data)
            
            # 计算损失
            total_loss, classification_loss, margin_loss, fisher_loss = compute_rpm_loss(
                logits, targets, embeddings, model.reciprocal_points, model.margins, 
                config.LAMBDA, config.FISHER_LAMBDA, class_weights
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 统计
            epoch_total_loss += total_loss.item()
            epoch_classification_loss += classification_loss.item()
            epoch_margin_loss += margin_loss.item()
            epoch_fisher_loss += fisher_loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
        
        # 更新学习率
        scheduler.step()
        
        # 记录每轮的平均指标
        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_classification_loss = epoch_classification_loss / len(train_loader)
        avg_margin_loss = epoch_margin_loss / len(train_loader)
        avg_fisher_loss = epoch_fisher_loss / len(train_loader)
        accuracy = 100.0 * epoch_correct / epoch_total
        
        training_history['total_loss'].append(avg_total_loss)
        training_history['classification_loss'].append(avg_classification_loss)
        training_history['margin_loss'].append(avg_margin_loss)
        training_history['fisher_loss'].append(avg_fisher_loss)
        training_history['accuracy'].append(accuracy)
        
        # 在每个epoch评估模型性能
        model.eval()
        
        # 评估已知类分类性能
        known_results = evaluate_known_classification(model, test_known_loader, idx_to_label, config, verbose=False)
        training_history['test_accuracy'].append(known_results['accuracy'])
        training_history['test_macro_f1'].append(known_results['macro_f1'])
        
        # 评估验证集的AUPR-OUT
        validation_metrics = evaluate_unknown_detection(
            model, test_known_loader, validation_unknown_loader, config, unknown_type="validation"
        )
        training_history['validation_aupr_out'].append(validation_metrics['aupr_out'])
        
        # 评估测试集平均AUPR-OUT
        test_aupr_outs = []
        unknown_results = {}
        for unknown_class, unknown_loader in test_unknown_loaders.items():
            test_metrics = evaluate_unknown_detection(
                model, test_known_loader, unknown_loader, config, unknown_type="test"
            )
            test_aupr_outs.append(test_metrics['aupr_out'])
            unknown_results[unknown_class] = test_metrics
        
        avg_test_aupr_out = np.mean(test_aupr_outs) if test_aupr_outs else 0.0
        training_history['test_aupr_out'].append(avg_test_aupr_out)
        
        model.train()
        
        # 打印训练进度
        print(f"轮次 [{epoch+1}/{config.EPOCHS}]:")
        print(f"  总损失: {avg_total_loss:.4f}")
        print(f"  分类损失: {avg_classification_loss:.4f}")
        print(f"  边际损失: {avg_margin_loss:.4f}")
        print(f"  Fisher损失: {avg_fisher_loss:.4f}")
        print(f"  训练准确率: {accuracy:.2f}%")
        print(f"  测试准确率: {known_results['accuracy']*100:.2f}%")
        print(f"  测试宏平均F1: {known_results['macro_f1']:.4f}")
        print(f"  验证集AUPR-OUT: {validation_metrics['aupr_out']:.4f}")
        print(f"  测试集平均AUPR-OUT: {avg_test_aupr_out:.4f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 从第10个epoch开始，每个epoch都保存结果
        if epoch >= 9:
            # 创建带有epoch编号的文件夹
            epoch_dir = os.path.join(config.RESULT_DIR, f"fisher_epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            print(f"📁 创建第{epoch+1}轮结果目录: {epoch_dir}")
            
            # 保存当前完整模型
            complete_model_file = os.path.join(epoch_dir, f'rpm_fisher_model_epoch_{epoch+1}.pth')
            torch.save(model, complete_model_file)
            print(f"💾 完整模型已保存到: {complete_model_file}")
            
            # 保存模型状态字典
            model_file = os.path.join(epoch_dir, f'rpm_fisher_model_state_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_total_loss,
                'config': config.to_dict(),
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'input_dim': model.feature_extractor[0].in_features,
                'hidden_dim': config.HIDDEN_DIM,
                'embedding_dim': config.EMBEDDING_DIM,
                'num_classes': model.num_classes,
                'gamma': model.gamma
            }, model_file)
            
            # 保存当前训练历史
            history_file = os.path.join(epoch_dir, f'training_history_epoch_{epoch+1}.pkl')
            with open(history_file, 'wb') as f:
                pickle.dump(training_history, f)
            
            # 绘制当前训练曲线
            plot_training_curves(training_history, config, save_dir=epoch_dir, epoch_num=epoch+1)
            
            print(f"✅ 第{epoch+1}轮结果已保存到: {epoch_dir}")
    
    print("✅ 模型训练完成!")
    return model, training_history


def plot_training_curves(training_history, config, save_dir=None, epoch_num=None):
    """绘制训练曲线"""
    if save_dir is None:
        save_dir = config.RESULT_DIR
    
    print("📊 绘制训练曲线...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('RPM+Fisher训练过程', fontsize=16)
    
    # 总损失
    axes[0, 0].plot(training_history['total_loss'])
    axes[0, 0].set_title('总损失')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].grid(True)
    
    # 分类损失
    axes[0, 1].plot(training_history['classification_loss'], color='orange')
    axes[0, 1].set_title('分类损失')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].grid(True)
    
    # 边际损失
    axes[1, 0].plot(training_history['margin_loss'], color='green')
    axes[1, 0].set_title('对抗边际损失')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('损失值')
    axes[1, 0].grid(True)
    
    # Fisher损失
    axes[1, 1].plot(training_history['fisher_loss'], color='purple')
    axes[1, 1].set_title('Fisher损失')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('损失值')
    axes[1, 1].grid(True)
    
    # 准确率
    axes[2, 0].plot(training_history['accuracy'], color='red', label='训练')
    if 'test_accuracy' in training_history:
        axes[2, 0].plot(np.array(training_history['test_accuracy']) * 100, color='blue', label='测试')
    axes[2, 0].set_title('准确率')
    axes[2, 0].set_xlabel('轮次')
    axes[2, 0].set_ylabel('准确率 (%)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # 验证集和测试集AUPR-OUT
    axes[2, 1].plot(training_history['validation_aupr_out'], color='darkgreen', label='验证集')
    if 'test_aupr_out' in training_history:
        axes[2, 1].plot(training_history['test_aupr_out'], color='brown', label='测试集')
    axes[2, 1].set_title('AUPR-OUT')
    axes[2, 1].set_xlabel('轮次')
    axes[2, 1].set_ylabel('AUPR-OUT')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    if epoch_num is not None:
        plot_file = os.path.join(save_dir, f'training_curves_epoch_{epoch_num}.png')
    else:
        plot_file = os.path.join(save_dir, 'training_curves.png')
    
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 训练曲线已保存到: {plot_file}")
