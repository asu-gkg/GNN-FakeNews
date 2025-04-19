 #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较better_gnn和gnn模型性能的脚本
使用相同的数据集和参数设置运行两个模型，并比较它们的性能指标
"""
import argparse
import time
import os
import numpy as np
import torch
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

# 设置命令行参数
parser = argparse.ArgumentParser(description='比较gnn和better_gnn的性能')
parser.add_argument('--dataset', type=str, default='politifact', help='数据集 [politifact, gossipcop]')
parser.add_argument('--feature', type=str, default='bert', help='特征类型 [profile, spacy, bert, content]')
parser.add_argument('--device', type=str, default='cuda:0', help='GPU设备')
parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
parser.add_argument('--runs', type=int, default=3, help='每个模型运行次数')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--output_dir', type=str, default='comparison_results', help='输出目录')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 设置两个模型的参数
gnn_params = [
    'python', 'gnn_model/gnn.py',
    '--dataset', args.dataset,
    '--feature', args.feature,
    '--device', args.device,
    '--batch_size', str(args.batch_size),
    '--epochs', str(args.epochs),
    '--lr', '0.01',
    '--weight_decay', '0.01',
    '--model', 'sage',
    '--seed'
]

better_gnn_params = [
    'python', 'gnn_model/better_gnn.py',
    '--dataset', args.dataset,
    '--feature', args.feature,
    '--device', args.device,
    '--batch_size', str(args.batch_size),
    '--epochs', str(args.epochs),
    '--lr', '1e-3',
    '--weight_decay', '5e-4',
    '--seed'
]

# 结果存储
results = {
    'model': [],
    'run': [],
    'time': [],
    'accuracy': [],
    'f1_macro': [],
    'auc': [],
    'ap': []
}

# 运行模型并收集结果
def run_model(model_name, params, run_id, seed):
    print(f"\n{'='*80}")
    print(f"运行 {model_name} (运行 {run_id+1}/{args.runs})")
    print(f"{'='*80}")
    
    # 设置随机种子
    current_seed = seed + run_id
    current_params = params + [str(current_seed)]
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行模型并捕获输出
    process = subprocess.Popen(
        current_params,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    stdout, stderr = process.communicate()
    
    # 记录结束时间
    elapsed_time = time.time() - start_time
    
    # 保存输出到文件
    output_file = os.path.join(args.output_dir, f"{model_name}_run{run_id+1}.log")
    with open(output_file, 'w') as f:
        f.write(stdout)
        if stderr:
            f.write("\nERRORS:\n")
            f.write(stderr)
    
    # 从输出中提取测试结果
    results['model'].append(model_name)
    results['run'].append(run_id+1)
    results['time'].append(elapsed_time)
    
    # 尝试从输出中解析结果
    try:
        # 针对GNN模型的输出格式
        if model_name == 'gnn':
            test_result_line = [line for line in stdout.split('\n') if 'Test set results:' in line][0]
            metrics = test_result_line.split('Test set results:')[1].strip()
            accuracy = float(metrics.split('acc:')[1].split(',')[0].strip())
            f1_macro = float(metrics.split('f1_macro:')[1].split(',')[0].strip())
            auc = float(metrics.split('auc:')[1].split(',')[0].strip())
            ap = float(metrics.split('ap:')[1].strip())
        # 针对Better GNN模型的输出格式
        elif model_name == 'better_gnn':
            test_result_line = [line for line in stdout.split('\n') if '>>> Test' in line][0]
            accuracy = float(test_result_line.split('Acc')[1].split('F1')[0].strip())
            f1_macro = float(test_result_line.split('F1')[1].split('AUC')[0].strip())
            auc = float(test_result_line.split('AUC')[1].strip())
            ap = 0.0  # Better GNN可能没有报告AP值，设为0
            
        results['accuracy'].append(accuracy)
        results['f1_macro'].append(f1_macro)
        results['auc'].append(auc)
        results['ap'].append(ap)
        
        print(f"完成 {model_name} 运行 {run_id+1}:")
        print(f"  准确率: {accuracy:.4f}, F1: {f1_macro:.4f}, AUC: {auc:.4f}, 用时: {elapsed_time:.2f}秒")
        
    except Exception as e:
        print(f"解析 {model_name} 运行 {run_id+1} 的结果时出错: {str(e)}")
        print(f"请查看输出文件: {output_file}")
        results['accuracy'].append(0.0)
        results['f1_macro'].append(0.0)
        results['auc'].append(0.0)
        results['ap'].append(0.0)

# 运行两个模型多次
for run in range(args.runs):
    run_model('gnn', gnn_params, run, args.seed)
    run_model('better_gnn', better_gnn_params, run, args.seed)

# 创建DataFrame
df = pd.DataFrame(results)

# 保存原始结果
df.to_csv(os.path.join(args.output_dir, 'all_results.csv'), index=False)

# 计算每个模型的平均性能
summary = df.groupby('model').agg({
    'time': ['mean', 'std'],
    'accuracy': ['mean', 'std'],
    'f1_macro': ['mean', 'std'],
    'auc': ['mean', 'std'],
    'ap': ['mean', 'std']
})

# 保存摘要结果
summary.to_csv(os.path.join(args.output_dir, 'summary.csv'))

# 在控制台打印美观的表格
print("\n\n性能比较摘要:")
table_data = []
for model in ['gnn', 'better_gnn']:
    if model in summary.index:
        row = [
            model,
            f"{summary.loc[model, ('accuracy', 'mean')]:.4f} ± {summary.loc[model, ('accuracy', 'std')]:.4f}",
            f"{summary.loc[model, ('f1_macro', 'mean')]:.4f} ± {summary.loc[model, ('f1_macro', 'std')]:.4f}",
            f"{summary.loc[model, ('auc', 'mean')]:.4f} ± {summary.loc[model, ('auc', 'std')]:.4f}",
            f"{summary.loc[model, ('time', 'mean')]:.2f}s ± {summary.loc[model, ('time', 'std')]:.2f}"
        ]
        table_data.append(row)

print(tabulate(table_data, headers=['模型', '准确率', 'F1分数', 'AUC', '运行时间'], tablefmt='pretty'))

# 绘制性能比较图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 准确率比较
axs[0].bar(['GNN', 'Better GNN'], 
          [summary.loc['gnn', ('accuracy', 'mean')], summary.loc['better_gnn', ('accuracy', 'mean')]],
          yerr=[summary.loc['gnn', ('accuracy', 'std')], summary.loc['better_gnn', ('accuracy', 'std')]])
axs[0].set_title('准确率比较')
axs[0].set_ylim(0.5, 1.0)  # 设置y轴范围从0.5到1.0，这样差异更明显

# F1分数比较
axs[1].bar(['GNN', 'Better GNN'], 
          [summary.loc['gnn', ('f1_macro', 'mean')], summary.loc['better_gnn', ('f1_macro', 'mean')]],
          yerr=[summary.loc['gnn', ('f1_macro', 'std')], summary.loc['better_gnn', ('f1_macro', 'std')]])
axs[1].set_title('F1分数比较')
axs[1].set_ylim(0.5, 1.0)

# AUC比较
axs[2].bar(['GNN', 'Better GNN'], 
          [summary.loc['gnn', ('auc', 'mean')], summary.loc['better_gnn', ('auc', 'mean')]],
          yerr=[summary.loc['gnn', ('auc', 'std')], summary.loc['better_gnn', ('auc', 'std')]])
axs[2].set_title('AUC比较')
axs[2].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'performance_comparison.png'), dpi=300)

print(f"\n结果已保存到 {args.output_dir} 目录")
print(f"- CSV文件: all_results.csv, summary.csv")
print(f"- 性能对比图: performance_comparison.png")
print(f"- 各运行日志: gnn_run*.log, better_gnn_run*.log")