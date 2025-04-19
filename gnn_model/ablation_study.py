import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gnn import Model, FeatureImportanceAnalyzer
from utils.data_loader import *
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='GNN假新闻检测模型消融实验')
    parser.add_argument('--seed', type=int, default=777, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='指定CUDA设备')
    parser.add_argument('--dataset', type=str, default='politifact', help='数据集 [politifact, gossipcop]')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--nhid', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='丢弃率')
    parser.add_argument('--feature', type=str, default='bert', help='特征类型 [profile, spacy, bert, content]')
    parser.add_argument('--model', type=str, default='sage', help='模型类型 [gcn, gat, sage]')
    parser.add_argument('--concat', type=bool, default=True, help='是否连接新闻嵌入和图嵌入')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='多GPU模式')
    parser.add_argument('--output_dir', type=str, default='results', help='结果输出目录')
    return parser.parse_args()

def prepare_data(args):
    """准备数据集并加载模型"""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 加载数据集
    dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())
    
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    
    # 分割数据集
    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    
    # 创建数据加载器
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    return test_loader, args

def run_feature_importance_analysis(args, model_path=None):
    """运行特征重要性分析实验"""
    test_loader, args = prepare_data(args)
    
    # 加载预训练模型
    model = Model(args, concat=args.concat)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model = model.to(args.device)
    model.eval()
    
    # 进行特征重要性分析
    print(f"\n开始在{args.dataset}数据集上进行节点特征与图结构重要性分析...")
    analyzer = FeatureImportanceAnalyzer(args, model, test_loader)
    importance_results = analyzer.evaluate_importance()
    
    # 可视化结果
    visualize_importance_results(importance_results, args)
    
    return importance_results

def visualize_importance_results(results, args):
    """可视化特征重要性分析结果"""
    # 创建结果目录
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制准确率比较图
    plt.figure(figsize=(10, 6))
    methods = ['原始模型', '仅节点特征', '仅图结构']
    accs = [results['original_acc'], results['feature_only_acc'], results['structure_only_acc']]
    
    plt.bar(methods, accs, color=['blue', 'orange', 'green'])
    plt.ylim(0, 1.0)
    plt.ylabel('准确率')
    plt.title(f'{args.dataset}数据集上不同模型组件的性能比较')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.savefig(f'{args.output_dir}/{args.dataset}_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    
    # 绘制相对重要性饼图
    plt.figure(figsize=(8, 8))
    importance = [results['feature_importance'], results['structure_importance']]
    labels = ['节点特征', '图结构']
    explode = (0.1, 0.1)
    
    plt.pie(importance, explode=explode, labels=labels, autopct='%1.1f%%', 
            shadow=True, startangle=90, colors=['orange', 'green'])
    plt.axis('equal')
    plt.title(f'{args.dataset}数据集上节点特征与图结构的相对重要性')
    
    plt.savefig(f'{args.output_dir}/{args.dataset}_importance_pie.png', dpi=300, bbox_inches='tight')
    
    # 将结果保存为CSV
    df = pd.DataFrame({
        'Dataset': [args.dataset],
        'Model': [args.model],
        'Feature': [args.feature],
        'Original_Acc': [results['original_acc']],
        'Feature_Only_Acc': [results['feature_only_acc']],
        'Structure_Only_Acc': [results['structure_only_acc']],
        'Feature_Importance': [results['feature_importance']],
        'Structure_Importance': [results['structure_importance']]
    })
    
    df.to_csv(f'{args.output_dir}/{args.dataset}_importance_results.csv', index=False)
    
    print(f"可视化结果已保存至 {args.output_dir} 目录")

def compare_models(args):
    """比较不同GNN模型的性能和特征重要性"""
    models = ['gcn', 'sage', 'gat']
    results = []
    
    for model_type in models:
        print(f"\n------ 分析 {model_type.upper()} 模型 ------")
        args.model = model_type
        model_results = run_feature_importance_analysis(args)
        results.append({
            'model': model_type,
            'results': model_results
        })
    
    # 比较不同模型的结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 准确率比较
    x = np.arange(len(models))
    width = 0.25
    
    original_accs = [r['results']['original_acc'] for r in results]
    feature_accs = [r['results']['feature_only_acc'] for r in results]
    structure_accs = [r['results']['structure_only_acc'] for r in results]
    
    ax1.bar(x - width, original_accs, width, label='原始模型')
    ax1.bar(x, feature_accs, width, label='仅节点特征')
    ax1.bar(x + width, structure_accs, width, label='仅图结构')
    
    ax1.set_ylabel('准确率')
    ax1.set_title('不同GNN模型的性能比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    
    # 特征重要性比较
    feature_importance = [r['results']['feature_importance'] for r in results]
    structure_importance = [r['results']['structure_importance'] for r in results]
    
    ax2.bar(x - width/2, feature_importance, width, label='节点特征重要性')
    ax2.bar(x + width/2, structure_importance, width, label='图结构重要性')
    
    ax2.set_ylabel('相对重要性 (%)')
    ax2.set_title('不同GNN模型的特征重要性比较')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    
    # 保存比较结果
    df = pd.DataFrame({
        'Model': models,
        'Original_Acc': original_accs,
        'Feature_Only_Acc': feature_accs,
        'Structure_Only_Acc': structure_accs,
        'Feature_Importance': feature_importance,
        'Structure_Importance': structure_importance
    })
    
    df.to_csv(f'{args.output_dir}/model_comparison.csv', index=False)
    
    return results

if __name__ == '__main__':
    args = parse_args()
    print("开始进行GNN假新闻检测模型消融实验...")
    
    # 单模型分析
    importance_results = run_feature_importance_analysis(args)
    
    # 多模型比较分析
    if args.model == 'all':
        compare_results = compare_models(args)
        
    print("\n消融实验完成! 结果已保存到", args.output_dir) 