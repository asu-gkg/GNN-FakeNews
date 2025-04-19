import argparse
import time
from tqdm import tqdm
import copy as cp
import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import to_networkx


from utils.data_loader import *
from utils.eval_helper import *


"""
Enhanced GNN model with graph topological features
"""


def extract_graph_features(data):
    """
    Extract topological features from the graph
    """
    G = to_networkx(data, to_undirected=True)
    
    # 计算图的主要拓扑特征
    num_nodes = G.number_of_nodes()
    if num_nodes <= 1:
        return torch.zeros(5, dtype=torch.float)
    
    # 计算度分布相关指标
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes
    max_degree = max(degrees) if degrees else 0
    degree_centrality = max_degree / (num_nodes - 1) if num_nodes > 1 else 0
    
    # 计算聚类系数和连通性
    clustering = nx.average_clustering(G) if num_nodes > 2 else 0
    
    # 计算图密度
    density = nx.density(G)
    
    # 返回5个关键拓扑特征
    return torch.tensor([
        avg_degree / 10.0,  # 归一化处理
        degree_centrality,
        clustering,
        density,
        float(num_nodes) / 100.0  # 归一化处理
    ], dtype=torch.float)


class TopoModel(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(TopoModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat
        self.topo_features = args.topo_features

        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)

        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

        # 新增: 拓扑特征融合层
        if self.topo_features:
            self.topo_lin = torch.nn.Linear(5, self.nhid)  # 5个拓扑特征
            self.final_lin = torch.nn.Linear(self.nhid * 2, self.nhid)
            
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        # GNN处理
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, batch)

        # 处理新闻节点特征
        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))

        # 新增: 拓扑特征融合
        if self.topo_features:
            # 为每个图提取拓扑特征
            topo_feats = []
            for i in range(data.num_graphs):
                # 创建子图
                sub_data = data.clone()
                mask = data.batch == i
                sub_data.x = data.x[mask]
                sub_data.edge_index = data.edge_index[:, (data.batch[data.edge_index[0]] == i) & (data.batch[data.edge_index[1]] == i)]
                
                # 获取拓扑特征
                topo_feat = extract_graph_features(sub_data)
                topo_feats.append(topo_feat)
            
            # 拼接所有图的拓扑特征
            topo_feats = torch.stack(topo_feats).to(x.device)
            
            # 处理拓扑特征
            topo_x = F.relu(self.topo_lin(topo_feats))
            
            # 融合GNN特征和拓扑特征
            x = torch.cat([x, topo_x], dim=1)
            x = F.relu(self.final_lin(x))

        x = F.log_softmax(self.lin2(x), dim=-1)
        return x


@torch.no_grad()
def compute_test(loader, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    for data in loader:
        if not args.multi_gpu:
            data = data.to(args.device)
        out = model(data)
        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
        else:
            y = data.y
        if verbose:
            print(F.softmax(out, dim=1).cpu().numpy())
        out_log.append([F.softmax(out, dim=1), y])
        loss_test += F.nll_loss(out, y).item()
    return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=35, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='sage', help='model type, [gcn, gat, sage]')
parser.add_argument('--topo_features', type=bool, default=True, help='whether to use topological features')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
    loader = DataListLoader
else:
    loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = TopoModel(args, concat=args.concat)
if args.multi_gpu:
    model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
    # Model training
    min_loss = 1e10
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if not args.multi_gpu:
                data = data.to(args.device)
            out = model(data)
            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
            else:
                y = data.y
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
              f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
              f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

    [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
          f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}') 