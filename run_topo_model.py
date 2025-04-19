# ✅ 使用真实 GossipCop 数据集，绘制拓扑结构分析图（使用 FNNDataset 和 extract_graph_features）
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx

# 真实模块导入路径（以下模块需保证在项目中存在）
from utils.data_loader import FNNDataset, ToUndirected
from gnn_model.gnn_topo import extract_graph_features

# 设置参数
dataset_name = 'politifact'
feature_type = 'profile'

# 加载真实数据集
dataset = FNNDataset(root='data', feature=feature_type, empty=False, name=dataset_name, transform=ToUndirected())

# 提取图拓扑特征与标签
topo_features = []
labels = []
graph_sizes = []

for i in range(len(dataset)):
    data = dataset[i]
    topo_feat = extract_graph_features(data)
    topo_features.append(topo_feat.numpy())
    labels.append(data.y.item())
    graph_sizes.append(data.num_nodes)

topo_features = np.array(topo_features)
labels = np.array(labels)
graph_sizes = np.array(graph_sizes)

# 特征名
feature_names = ['Average Degree', 'Degree Centrality', 'Clustering Coefficient', 'Graph Density', 'Node Count']

# 构造 DataFrame
df = pd.DataFrame(topo_features, columns=feature_names)
df['label'] = labels
df['graph_size'] = graph_sizes
df['Label'] = df['label'].map({0: 'Real News', 1: 'Fake News'})

# 使用统一的 subplot 绘图结构
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.set(style="whitegrid")

# 1. Boxplot
df_melted = pd.melt(df, id_vars=['Label'], value_vars=feature_names, var_name='Feature', value_name='Value')
sns.boxplot(x='Feature', y='Value', hue='Label', data=df_melted, ax=axs[0, 0])
axs[0, 0].set_title(f'{dataset_name.capitalize()} Dataset: Topological Feature Distribution')
axs[0, 0].tick_params(axis='x', rotation=45)

# 2. Scatter plot
x_feature, y_feature = 'Average Degree', 'Clustering Coefficient'
sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Label', style='Label', palette='Set1', alpha=0.7, ax=axs[0, 1])
axs[0, 1].set_title(f'{x_feature} vs {y_feature}')

# 3. Histogram
sns.histplot(data=df, x='Node Count', hue='Label', element='step', stat='density', bins=30, palette='Set2', ax=axs[1, 0])
axs[1, 0].set_title('Propagation Graph Size Distribution')

# 4. Heatmap
corr = df[feature_names].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", ax=axs[1, 1])
axs[1, 1].set_title('Topological Feature Correlation')

# 保存图像
fig.tight_layout(pad=2.5)
output_path = f"./data/{dataset_name}_{feature_type}_topo_analysis_real.png"
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

output_path
