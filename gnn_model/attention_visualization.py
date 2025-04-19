import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
import os
import argparse
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from utils.data_loader import FNNDataset, ToUndirected

class GATForVisualization(torch.nn.Module):
    """
    用于注意力可视化的GAT模型
    """
    def __init__(self, args):
        super(GATForVisualization, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.concat = args.concat
        
        # 使用GAT层，并保存注意力权重
        self.gat = GATConv(self.num_features, self.nhid, heads=1, dropout=self.dropout_ratio, return_attention_weights=True)
        
        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 获取注意力权重
        (x, attention_weights) = self.gat(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        # 全局池化
        pooled = torch.zeros(data.num_graphs, self.nhid, device=x.device)
        for i in range(data.num_graphs):
            mask = (batch == i)
            pooled[i] = x[mask].max(dim=0)[0]
        
        # 特征拼接
        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            pooled = torch.cat([pooled, news], dim=1)
            pooled = F.relu(self.lin1(pooled))
        
        # 输出层
        out = F.log_softmax(self.lin2(pooled), dim=-1)
        
        return out, attention_weights

def visualize_attention(model, data_loader, args, num_samples=5):
    """
    可视化GAT模型的注意力权重
    
    参数:
        model: GATForVisualization模型
        data_loader: 数据加载器
        args: 参数对象
        num_samples: 要可视化的样本数量
    """
    model.eval()
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'attention_viz')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取一批数据
    with torch.no_grad():
        sample_count = 0
        for data in data_loader:
            data = data.to(args.device)
            
            # 只处理指定数量的样本
            for graph_idx in range(min(data.num_graphs, num_samples - sample_count)):
                # 提取单个图
                sub_nodes = (data.batch == graph_idx).nonzero().squeeze().tolist()
                if not isinstance(sub_nodes, list):
                    sub_nodes = [sub_nodes]  # 处理只有一个节点的情况
                
                sub_data = data.clone()
                sub_data.x = data.x[sub_nodes]
                sub_data.y = data.y[graph_idx].unsqueeze(0)
                sub_data.batch = torch.zeros(len(sub_nodes), dtype=torch.long, device=args.device)
                
                # 提取子图的边
                edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=args.device)
                for i in range(data.edge_index.size(1)):
                    if data.edge_index[0, i] in sub_nodes and data.edge_index[1, i] in sub_nodes:
                        edge_mask[i] = True
                
                # 重新映射节点索引
                node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sub_nodes)}
                new_edge_index = data.edge_index[:, edge_mask].clone()
                for i in range(new_edge_index.size(1)):
                    new_edge_index[0, i] = node_map[new_edge_index[0, i].item()]
                    new_edge_index[1, i] = node_map[new_edge_index[1, i].item()]
                
                sub_data.edge_index = new_edge_index
                
                # 获取模型预测和注意力权重
                out, (_, attention_weights) = model(sub_data)
                predicted_class = out.argmax(dim=1).item()
                true_class = sub_data.y.item()
                
                # 转换为NetworkX图进行可视化
                if len(sub_nodes) > 1:  # 确保有足够的节点
                    G = to_networkx(sub_data, to_undirected=True)
                    
                    # 提取注意力权重
                    edge_weights = attention_weights.cpu().numpy()
                    max_weight = edge_weights.max()
                    normalized_weights = edge_weights / max_weight
                    
                    # 设置节点颜色 (蓝色为真实新闻，红色为假新闻)
                    node_colors = ['lightblue' if true_class == 0 else 'lightcoral']
                    node_colors.extend(['lightgray'] * (len(G.nodes) - 1))
                    
                    # 设置边的宽度根据注意力权重
                    edge_widths = [3 * w for w in normalized_weights]
                    
                    # 绘制图
                    plt.figure(figsize=(10, 10))
                    pos = nx.spring_layout(G, seed=42)
                    
                    # 绘制节点
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
                    
                    # 绘制边，颜色根据注意力权重
                    edges = list(G.edges())
                    cmap = plt.cm.Reds
                    nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, 
                                          edge_color=normalized_weights, edge_cmap=cmap, alpha=0.7)
                    
                    # 添加颜色条
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_weight))
                    sm.set_array([])
                    plt.colorbar(sm, label='注意力权重')
                    
                    # 添加节点标签
                    labels = {0: 'Root'}
                    labels.update({i: str(i) for i in range(1, len(G.nodes))})
                    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
                    
                    # 设置标题
                    status = "正确" if predicted_class == true_class else "错误"
                    plt.title(f"样本 #{sample_count+1}: {'真实新闻' if true_class == 0 else '假新闻'} (预测: {'真实新闻' if predicted_class == 0 else '假新闻'}) - {status}")
                    
                    # 保存图像
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'sample_{sample_count+1}_attention.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # 输出注意力分析
                    print(f"样本 #{sample_count+1} - {'真实新闻' if true_class == 0 else '假新闻'} (预测: {'真实新闻' if predicted_class == 0 else '假新闻'}):")
                    
                    # 分析哪些节点接收了最高的注意力
                    source_nodes, target_nodes = sub_data.edge_index.cpu().numpy()
                    for i, (s, t, w) in enumerate(zip(source_nodes, target_nodes, edge_weights)):
                        print(f"  边 {s} -> {t}: 注意力权重 = {w:.4f}")
                    
                    # 计算每个节点的平均接收注意力
                    node_attention = {}
                    for i, (s, t, w) in enumerate(zip(source_nodes, target_nodes, edge_weights)):
                        if t not in node_attention:
                            node_attention[t] = []
                        node_attention[t].append(w)
                    
                    # 输出平均注意力
                    avg_attention = {node: np.mean(weights) for node, weights in node_attention.items()}
                    sorted_nodes = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)
                    print("  节点接收的平均注意力:")
                    for node, att in sorted_nodes[:5]:  # 只显示前5个
                        print(f"    节点 {node}: {att:.4f}")
                    
                    print()
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
            
            if sample_count >= num_samples:
                break
    
    print(f"注意力可视化已保存到 {output_dir} 目录")

def analyze_attention_patterns(model, data_loader, args):
    """
    分析模型的注意力模式与新闻真假性的关系
    
    参数:
        model: GATForVisualization模型
        data_loader: 数据加载器
        args: 参数对象
    """
    model.eval()
    
    # 收集真假新闻样本的注意力统计信息
    real_attention_stats = {
        'max': [],
        'mean': [],
        'std': []
    }
    
    fake_attention_stats = {
        'max': [],
        'mean': [],
        'std': []
    }
    
    # 分析每个样本
    with torch.no_grad():
        for data in data_loader:
            data = data.to(args.device)
            
            for graph_idx in range(data.num_graphs):
                # 提取单个图
                sub_nodes = (data.batch == graph_idx).nonzero().squeeze().tolist()
                if not isinstance(sub_nodes, list):
                    sub_nodes = [sub_nodes]
                
                if len(sub_nodes) <= 1:  # 跳过只有一个节点的图
                    continue
                
                sub_data = data.clone()
                sub_data.x = data.x[sub_nodes]
                sub_data.y = data.y[graph_idx].unsqueeze(0)
                sub_data.batch = torch.zeros(len(sub_nodes), dtype=torch.long, device=args.device)
                
                # 提取子图的边
                edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool, device=args.device)
                for i in range(data.edge_index.size(1)):
                    if data.edge_index[0, i] in sub_nodes and data.edge_index[1, i] in sub_nodes:
                        edge_mask[i] = True
                
                # 重新映射节点索引
                node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sub_nodes)}
                new_edge_index = data.edge_index[:, edge_mask].clone()
                for i in range(new_edge_index.size(1)):
                    new_edge_index[0, i] = node_map[new_edge_index[0, i].item()]
                    new_edge_index[1, i] = node_map[new_edge_index[1, i].item()]
                
                sub_data.edge_index = new_edge_index
                
                # 获取注意力权重
                _, (_, attention_weights) = model(sub_data)
                true_class = sub_data.y.item()
                
                # 计算统计信息
                weights = attention_weights.cpu().numpy()
                
                if true_class == 0:  # 真实新闻
                    real_attention_stats['max'].append(weights.max())
                    real_attention_stats['mean'].append(weights.mean())
                    real_attention_stats['std'].append(weights.std())
                else:  # 假新闻
                    fake_attention_stats['max'].append(weights.max())
                    fake_attention_stats['mean'].append(weights.mean())
                    fake_attention_stats['std'].append(weights.std())
    
    # 可视化注意力统计特征
    plt.figure(figsize=(15, 5))
    
    # 最大注意力
    plt.subplot(1, 3, 1)
    plt.hist(real_attention_stats['max'], alpha=0.5, bins=20, label='真实新闻')
    plt.hist(fake_attention_stats['max'], alpha=0.5, bins=20, label='假新闻')
    plt.xlabel('最大注意力权重')
    plt.ylabel('频率')
    plt.legend()
    plt.title('最大注意力权重分布')
    
    # 平均注意力
    plt.subplot(1, 3, 2)
    plt.hist(real_attention_stats['mean'], alpha=0.5, bins=20, label='真实新闻')
    plt.hist(fake_attention_stats['mean'], alpha=0.5, bins=20, label='假新闻')
    plt.xlabel('平均注意力权重')
    plt.ylabel('频率')
    plt.legend()
    plt.title('平均注意力权重分布')
    
    # 注意力标准差
    plt.subplot(1, 3, 3)
    plt.hist(real_attention_stats['std'], alpha=0.5, bins=20, label='真实新闻')
    plt.hist(fake_attention_stats['std'], alpha=0.5, bins=20, label='假新闻')
    plt.xlabel('注意力权重标准差')
    plt.ylabel('频率')
    plt.legend()
    plt.title('注意力权重标准差分布')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'attention_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算统计显著性
    real_mean = np.mean(real_attention_stats['mean'])
    fake_mean = np.mean(fake_attention_stats['mean'])
    real_std = np.mean(real_attention_stats['std'])
    fake_std = np.mean(fake_attention_stats['std'])
    
    print("\n注意力模式分析:")
    print(f"真实新闻平均注意力: {real_mean:.4f} vs 假新闻: {fake_mean:.4f}")
    print(f"真实新闻注意力标准差: {real_std:.4f} vs 假新闻: {fake_std:.4f}")
    
    if real_mean > fake_mean:
        print("真实新闻倾向于产生更高的平均注意力权重")
    else:
        print("假新闻倾向于产生更高的平均注意力权重")
    
    if real_std > fake_std:
        print("真实新闻的注意力分布更分散（更多样化的注意力模式）")
    else:
        print("假新闻的注意力分布更分散（更多样化的注意力模式）")
    
    return {
        'real': real_attention_stats,
        'fake': fake_attention_stats
    }

def train_model(args):
    """
    训练GAT模型并保存
    
    参数:
        args: 参数对象
    
    返回:
        训练好的模型
    """
    # 加载数据
    dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())
    
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    
    # 分割数据集
    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    
    # 创建数据加载器
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = GATForVisualization(args)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练模型
    best_val_acc = 0
    for epoch in range(args.epochs):
        model.train()
        loss_train = 0.0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            out, _ = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        
        train_acc = correct / total
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for data in val_loader:
                data = data.to(args.device)
                out, _ = model(data)
                pred = out.argmax(dim=1)
                val_correct += (pred == data.y).sum().item()
                val_total += data.y.size(0)
            
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_gat_model.pth'))
                print(f"Epoch {epoch+1}/{args.epochs}: 保存最佳模型")
        
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={loss_train:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_gat_model.pth')))
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for data in test_loader:
            data = data.to(args.device)
            out, _ = model(data)
            pred = out.argmax(dim=1)
            test_correct += (pred == data.y).sum().item()
            test_total += data.y.size(0)
        
        test_acc = test_correct / test_total
    
    print(f"测试准确率: {test_acc:.4f}")
    
    return model, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN注意力可视化')
    parser.add_argument('--seed', type=int, default=777, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--dataset', type=str, default='politifact', help='数据集')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='权重衰减')
    parser.add_argument('--nhid', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--dropout_ratio', type=float, default=0.2, help='丢弃率')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--concat', type=bool, default=True, help='是否连接特征')
    parser.add_argument('--feature', type=str, default='bert', help='特征类型')
    parser.add_argument('--output_dir', type=str, default='results/attention', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='要可视化的样本数量')
    parser.add_argument('--load_model', type=str, default=None, help='加载已训练的模型路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练或加载模型
    if args.load_model is not None and os.path.exists(args.load_model):
        print(f"加载预训练模型: {args.load_model}")
        
        # 加载数据
        dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())
        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features
        
        # 创建模型
        model = GATForVisualization(args)
        model.load_state_dict(torch.load(args.load_model))
        model = model.to(args.device)
        
        # 加载测试集
        _, _, test_set = random_split(dataset, 
                                     [int(len(dataset) * 0.2), 
                                      int(len(dataset) * 0.1), 
                                      len(dataset) - int(len(dataset) * 0.2) - int(len(dataset) * 0.1)])
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    else:
        print("训练新模型...")
        model, test_loader = train_model(args)
    
    # 可视化注意力
    print("\n开始可视化注意力...")
    visualize_attention(model, test_loader, args, num_samples=args.num_samples)
    
    # 分析注意力模式
    print("\n开始分析注意力模式...")
    attention_stats = analyze_attention_patterns(model, test_loader, args)
    
    print("\n总结:")
    print(f"1. 我们分析了GAT模型在{args.dataset}数据集上的注意力分布")
    print(f"2. 可视化了{args.num_samples}个样本的注意力权重")
    print("3. 比较了真实新闻和假新闻的注意力模式差异")
    print("4. 可视化结果已保存到指定目录") 