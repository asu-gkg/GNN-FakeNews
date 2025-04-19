import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 实验配置
datasets = ['politifact', 'gossipcop']
features = ['profile', 'bert']
models = ['gcn', 'sage']
seeds = [777, 888, 999]  # 多个随机种子以获得统计显著性

# 结果存储
results = {
    'dataset': [],
    'feature': [],
    'model': [],
    'use_topo': [],
    'accuracy': [],
    'f1_macro': [],
    'recall': [],
    'auc': []
}

# 运行实验
for dataset in datasets:
    for feature in features:
        for model in models:
            # 使用原始GNN模型
            for seed in seeds:
                cmd = f"PYTHONPATH=. python gnn_model/gnn.py --dataset {dataset} --model {model} --feature {feature} --seed {seed} --batch_size 64"
                print(f"Running: {cmd}")
                output = subprocess.check_output(cmd, shell=True).decode('utf-8')
                
                # 解析结果
                for line in output.split('\n'):
                    if line.startswith('Test set results:'):
                        metrics = line.replace('Test set results:', '').strip().split(',')
                        acc = float(metrics[0].split(':')[1].strip())
                        f1_macro = float(metrics[1].split(':')[1].strip())
                        recall = float(metrics[4].split(':')[1].strip())
                        auc = float(metrics[5].split(':')[1].strip())
                        
                        results['dataset'].append(dataset)
                        results['feature'].append(feature)
                        results['model'].append(model)
                        results['use_topo'].append(False)
                        results['accuracy'].append(acc)
                        results['f1_macro'].append(f1_macro)
                        results['recall'].append(recall)
                        results['auc'].append(auc)
            
            # 使用拓扑增强GNN模型
            for seed in seeds:
                cmd = f"PYTHONPATH=. python gnn_model/gnn_topo.py --dataset {dataset} --model {model} --feature {feature} --seed {seed} --topo_features True --batch_size 64"
                print(f"Running: {cmd}")
                output = subprocess.check_output(cmd, shell=True).decode('utf-8')
                
                # 解析结果
                for line in output.split('\n'):
                    if line.startswith('Test set results:'):
                        metrics = line.replace('Test set results:', '').strip().split(',')
                        acc = float(metrics[0].split(':')[1].strip())
                        f1_macro = float(metrics[1].split(':')[1].strip())
                        recall = float(metrics[4].split(':')[1].strip())
                        auc = float(metrics[5].split(':')[1].strip())
                        
                        results['dataset'].append(dataset)
                        results['feature'].append(feature)
                        results['model'].append(model)
                        results['use_topo'].append(True)
                        results['accuracy'].append(acc)
                        results['f1_macro'].append(f1_macro)
                        results['recall'].append(recall)
                        results['auc'].append(auc)

# 转换为DataFrame并保存结果
df = pd.DataFrame(results)
df.to_csv('experiment_results.csv', index=False)

# 计算平均性能并绘制比较图
plt.figure(figsize=(12, 8))

# 按数据集和特征类型分组计算平均值，排除非数值列
numeric_cols = ['accuracy', 'f1_macro', 'recall', 'auc']
grouped = df.groupby(['dataset', 'feature', 'use_topo'])[numeric_cols].mean().reset_index()

# 创建4个子图，每个对应一个指标
metrics = ['accuracy', 'f1_macro', 'recall', 'auc']
titles = ['Accuracy', 'F1 Macro', 'Recall', 'AUC']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    plt.subplot(2, 2, i+1)
    
    # 为每个数据集和特征类型绘制柱状图
    x = np.arange(len(grouped['dataset'].unique()) * len(grouped['feature'].unique()))
    width = 0.35
    
    # 不使用拓扑特征的结果
    no_topo = grouped[grouped['use_topo'] == False]
    plt.bar(x - width/2, no_topo[metric], width, label='Original GNN')
    
    # 使用拓扑特征的结果
    with_topo = grouped[grouped['use_topo'] == True]
    plt.bar(x + width/2, with_topo[metric], width, label='Topo-Enhanced GNN')
    
    # 设置图表标签
    plt.title(title)
    plt.xticks(x, [f"{d}-{f}" for d, f in zip(no_topo['dataset'], no_topo['feature'])])
    plt.ylabel(title)
    if i == 1:
        plt.legend()

plt.tight_layout()
plt.savefig('experiment_results.png')
plt.show()

# 打印结果摘要
print("\n=== 实验结果摘要 ===")
summary = grouped.groupby(['use_topo'])[numeric_cols].mean()
print("\n平均性能提升:")
for metric in metrics:
    improvement = (summary.loc[True, metric] - summary.loc[False, metric]) / summary.loc[False, metric] * 100
    print(f"{metric}: {improvement:.2f}%") 