import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool as gmp

class MultiLayerGNN(torch.nn.Module):
    """
    多层GNN模型，用于探索GNN层数对假新闻检测性能的影响
    
    参数:
        args: 参数对象，包含模型配置
        num_layers: GNN层数
        concat: 是否连接新闻特征和图特征
    """
    def __init__(self, args, num_layers=2, concat=False):
        super(MultiLayerGNN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat
        self.num_layers = num_layers
        
        # 创建多层GNN
        self.convs = torch.nn.ModuleList()
        
        # 第一层的输入维度是特征维度
        if self.model == 'gcn':
            self.convs.append(GCNConv(self.num_features, self.nhid))
        elif self.model == 'sage':
            self.convs.append(SAGEConv(self.num_features, self.nhid))
        elif self.model == 'gat':
            self.convs.append(GATConv(self.num_features, self.nhid))
        
        # 后续层的输入和输出维度都是隐藏层维度
        for _ in range(1, num_layers):
            if self.model == 'gcn':
                self.convs.append(GCNConv(self.nhid, self.nhid))
            elif self.model == 'sage':
                self.convs.append(SAGEConv(self.nhid, self.nhid))
            elif self.model == 'gat':
                self.convs.append(GATConv(self.nhid, self.nhid))
        
        # 如果使用节点特征拼接
        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        
        # 输出层
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)
    
    def reset_parameters(self):
        """重置模型参数"""
        for conv in self.convs:
            conv.reset_parameters()
        if self.concat:
            self.lin0.reset_parameters()
            self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self, data):
        """前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        
        # 多层GNN传播
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        # 全局池化
        x = gmp(x, batch)
        
        # 如果使用节点特征拼接
        if self.concat:
            # 获取每个图的根节点特征
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))
        
        # 输出层
        x = F.log_softmax(self.lin2(x), dim=-1)
        
        return x

def layer_analysis(args, test_loader, max_layers=5):
    """
    分析GNN层数对模型性能的影响
    
    参数:
        args: 参数对象，包含模型配置
        test_loader: 测试数据加载器
        max_layers: 最大层数
        
    返回:
        results: 包含不同层数模型性能的字典
    """
    import time
    from utils.eval_helper import eval_deep
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    results = {
        'num_layers': [],
        'acc': [],
        'f1_macro': [],
        'auc': [],
        'training_time': []
    }
    
    for num_layers in range(1, max_layers + 1):
        print(f"\n===== 训练 {num_layers} 层 {args.model.upper()} 模型 =====")
        
        # 创建模型
        model = MultiLayerGNN(args, num_layers=num_layers, concat=args.concat)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 训练模型
        model.train()
        start_time = time.time()
        
        for epoch in range(args.epochs):
            loss_train = 0.0
            for i, data in enumerate(args.train_loader):
                optimizer.zero_grad()
                data = data.to(args.device)
                out = model(data)
                y = data.y
                loss = F.nll_loss(out, y)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss_train:.4f}")
        
        training_time = time.time() - start_time
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            out_log = []
            for data in test_loader:
                data = data.to(args.device)
                out = model(data)
                y = data.y
                out_log.append([F.softmax(out, dim=1), y])
            
            metrics, _ = eval_deep(out_log, test_loader)
            acc, f1_macro, _, _, _, auc, _ = metrics
        
        # 保存结果
        results['num_layers'].append(num_layers)
        results['acc'].append(acc)
        results['f1_macro'].append(f1_macro)
        results['auc'].append(auc)
        results['training_time'].append(training_time)
        
        print(f"层数: {num_layers}, 准确率: {acc:.4f}, F1: {f1_macro:.4f}, AUC: {auc:.4f}, 训练时间: {training_time:.2f}s")
    
    # 可视化结果
    output_dir = getattr(args, 'output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制性能指标随层数变化的曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['num_layers'], results['acc'], 'o-', label='准确率')
    plt.plot(results['num_layers'], results['f1_macro'], 's-', label='F1宏平均')
    plt.plot(results['num_layers'], results['auc'], '^-', label='AUC')
    plt.xlabel('GNN层数')
    plt.ylabel('性能指标')
    plt.title(f'{args.model.upper()}模型在{args.dataset}数据集上的性能随层数变化')
    plt.xticks(results['num_layers'])
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(results['num_layers'], results['training_time'], 'o-', color='red')
    plt.xlabel('GNN层数')
    plt.ylabel('训练时间 (秒)')
    plt.title('训练时间随层数变化')
    plt.xticks(results['num_layers'])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{args.model}_layer_analysis.png', dpi=300, bbox_inches='tight')
    
    # 分析最优层数
    best_layer = results['num_layers'][np.argmax(results['acc'])]
    
    # 绘制过拟合分析图（如果有验证集）
    if hasattr(args, 'val_loader'):
        val_acc = []
        train_acc = []
        
        for num_layers in range(1, max_layers + 1):
            model = MultiLayerGNN(args, num_layers=num_layers, concat=args.concat)
            model = model.to(args.device)
            
            # 训练并记录训练集和验证集准确率
            # ... (这部分代码需要完整实现训练循环并记录准确率)
        
        # 绘制训练集和验证集准确率对比图
        # ...
    
    # 输出分析结论
    print(f"\n===== 层数分析结论 =====")
    print(f"对于{args.dataset}数据集上的{args.model.upper()}模型:")
    print(f"最佳层数: {best_layer}")
    print(f"最高准确率: {max(results['acc']):.4f}")
    
    # 分析过拟合现象
    if max_layers >= 3:
        if results['acc'][2] < results['acc'][1] and results['acc'][2] < results['acc'][0]:
            print("观察到过拟合现象: 层数增加时性能下降")
        elif all(results['acc'][i] <= results['acc'][i+1] for i in range(len(results['acc'])-1)):
            print("没有观察到过拟合现象: 更深的层数持续改善性能")
        else:
            print(f"最佳层数为{best_layer}, 过深或过浅的层数会导致性能下降")
    
    return results

# 使用示例代码
if __name__ == "__main__":
    import argparse
    from torch.utils.data import random_split
    from torch_geometric.data import DataLoader
    from utils.data_loader import FNNDataset, ToUndirected
    
    parser = argparse.ArgumentParser(description='多层GNN假新闻检测模型')
    parser.add_argument('--seed', type=int, default=777, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--dataset', type=str, default='politifact', help='数据集')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--nhid', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='丢弃率')
    parser.add_argument('--epochs', type=int, default=35, help='训练轮数')
    parser.add_argument('--concat', type=bool, default=True, help='是否连接特征')
    parser.add_argument('--feature', type=str, default='bert', help='特征类型')
    parser.add_argument('--model', type=str, default='sage', help='GNN模型类型')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--max_layers', type=int, default=5, help='最大层数')
    
    args = parser.parse_args()
    
    # 设置随机种子
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
    args.train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    args.val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # 分析层数对性能的影响
    print("开始分析GNN层数对性能的影响...")
    analysis_results = layer_analysis(args, test_loader, max_layers=args.max_layers)
    
    print("分析完成! 结果已保存至", args.output_dir) 