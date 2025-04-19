import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx
from utils.data_loader import FNNDataset, ToUndirected
from gnn_model.gnn_topo import extract_graph_features

# Set parameters
dataset_name = 'politifact'  # or 'gossipcop'
feature_type = 'profile'     # or 'bert', 'spacy', 'content'

# Load dataset
dataset = FNNDataset(root='data', feature=feature_type, empty=False, name=dataset_name, transform=ToUndirected())

# Extract graph topological features and labels
topo_features = []
labels = []
graph_sizes = []

for i in range(len(dataset)):
    data = dataset[i]
    # Extract topological features
    topo_feat = extract_graph_features(data)
    topo_features.append(topo_feat.numpy())
    # Get labels (0=real news, 1=fake news)
    labels.append(data.y.item())
    # Get graph size
    graph_sizes.append(data.num_nodes)

# Convert to NumPy arrays
topo_features = np.array(topo_features)
labels = np.array(labels)
graph_sizes = np.array(graph_sizes)

# Create feature name list
feature_names = ['Average Degree', 'Degree Centrality', 'Clustering Coefficient', 'Graph Density', 'Node Count']

# Create dataframe for visualization
df = pd.DataFrame({
    'label': labels,
    'graph_size': graph_sizes
})

for i, name in enumerate(feature_names):
    df[name] = topo_features[:, i]

# Set visualization style
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))

# 1. Box plot comparing real/fake news distribution across topological features
plt.subplot(2, 2, 1)
df_melted = pd.melt(df, id_vars=['label'], value_vars=feature_names, 
                    var_name='Feature', value_name='Value')
df_melted['Label'] = df_melted['label'].map({0: 'Real News', 1: 'Fake News'})
sns.boxplot(x='Feature', y='Value', hue='Label', data=df_melted)
plt.title(f'{dataset_name} Dataset: Topological Feature Distribution Comparison')
plt.xticks(rotation=45)
plt.tight_layout()

# 2. Scatter plot showing two most discriminative features
plt.subplot(2, 2, 2)
if dataset_name == 'politifact':
    x_feature, y_feature = 'Clustering Coefficient', 'Graph Density'  # Example features, can be adjusted based on actual discriminative power
else:
    x_feature, y_feature = 'Average Degree', 'Clustering Coefficient'
    
sns.scatterplot(x=x_feature, y=y_feature, hue='label', 
                palette={0: 'blue', 1: 'red'}, 
                data=df, alpha=0.7,
                hue_norm=[0, 1], 
                size='graph_size', sizes=(20, 200))
plt.title(f'{x_feature} vs {y_feature}')
plt.legend(title='Label', labels=['Real News', 'Fake News'])

# 3. Histogram comparing graph size distribution for real/fake news
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='graph_size', hue='label', 
             palette={0: 'blue', 1: 'red'}, 
             element="step", common_norm=False,
             stat="density", bins=30)
plt.title('Propagation Graph Size Distribution')
plt.xlabel('Node Count')
plt.ylabel('Density')
plt.legend(title='Label', labels=['Real News', 'Fake News'])

# 4. Heatmap showing feature correlation
plt.subplot(2, 2, 4)
corr = df[feature_names].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f")
plt.title('Topological Feature Correlation')

# Save the figure
plt.tight_layout()
plt.savefig(f'{dataset_name}_{feature_type}_topo_analysis.png', dpi=300)
plt.show()

# Calculate feature discrimination between real and fake news
print(f"\n=== {dataset_name} Dataset Topological Feature Analysis ===")
fake_indices = labels == 1
real_indices = labels == 0

print("\nFeature Mean Comparison between Real/Fake News:")
for i, name in enumerate(feature_names):
    fake_mean = topo_features[fake_indices, i].mean()
    real_mean = topo_features[real_indices, i].mean()
    diff_percent = (fake_mean - real_mean) / real_mean * 100 if real_mean != 0 else float('inf')
    print(f"{name}: Fake News Mean={fake_mean:.4f}, Real News Mean={real_mean:.4f}, Difference={diff_percent:.2f}%") 