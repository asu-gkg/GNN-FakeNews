import argparse
import time
from tqdm import tqdm
import copy as cp

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader


from utils.data_loader import *
from utils.eval_helper import *


"""

The GCN, GAT, and GraphSAGE implementation

"""


class Model(torch.nn.Module):
	def __init__(self, args, concat=False):
		super(Model, self).__init__()
		self.args = args
		self.num_features = args.num_features
		self.nhid = args.nhid
		self.num_classes = args.num_classes
		self.dropout_ratio = args.dropout_ratio
		self.model = args.model
		self.concat = concat

		if self.model == 'gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
		elif self.model == 'sage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
		elif self.model == 'gat':
			self.conv1 = GATConv(self.num_features, self.nhid)

		if self.concat:
			self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
			self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

		self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

	def forward(self, data):

		x, edge_index, batch = data.x, data.edge_index, data.batch

		edge_attr = None

		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.lin0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.lin1(x))

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
parser.add_argument('--analysis_mode', type=str, default='none', help='analysis mode, [none, feature_importance]')

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

model = Model(args, concat=args.concat)
if args.multi_gpu:
	model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


class FeatureImportanceAnalyzer:
    """
    分析节点特征与图结构对假新闻检测的相对重要性
    
    通过对比三种情况下的模型性能来评估不同组件的重要性：
    1. 完整模型（使用节点特征和图结构）
    2. 仅使用节点特征的模型（随机化边连接）
    3. 仅使用图结构的模型（随机化节点特征）
    """
    def __init__(self, args, model, test_loader):
        self.args = args
        self.original_model = model
        self.test_loader = test_loader
        self.device = args.device
        
    def randomize_edges(self, data):
        """随机化图的边结构，保持节点特征不变"""
        if not self.args.multi_gpu:
            num_nodes = data.x.size(0)
            # 创建随机边连接
            random_edge_index = torch.randint(0, num_nodes, (2, num_nodes*2), device=self.device)
            data_copy = cp.copy(data)
            data_copy.edge_index = random_edge_index
            return data_copy
        else:
            # 多GPU情况下的处理
            data_copies = []
            for d in data:
                num_nodes = d.x.size(0)
                random_edge_index = torch.randint(0, num_nodes, (2, num_nodes*2), device=d.x.device)
                d_copy = cp.copy(d)
                d_copy.edge_index = random_edge_index
                data_copies.append(d_copy)
            return data_copies
            
    def randomize_features(self, data):
        """随机化节点特征，保持图结构不变"""
        if not self.args.multi_gpu:
            # 创建随机节点特征
            num_nodes = data.x.size(0)
            random_features = torch.randn(num_nodes, data.x.size(1), device=self.device)
            data_copy = cp.copy(data)
            data_copy.x = random_features
            return data_copy
        else:
            # 多GPU情况下的处理
            data_copies = []
            for d in data:
                num_nodes = d.x.size(0)
                random_features = torch.randn(num_nodes, d.x.size(1), device=d.x.device)
                d_copy = cp.copy(d)
                d_copy.x = random_features
                data_copies.append(d_copy)
            return data_copies
    
    @torch.no_grad()
    def evaluate_importance(self):
        """评估节点特征和图结构的相对重要性"""
        self.original_model.eval()
        
        # 测试原始模型
        original_results, _ = compute_test(self.test_loader)
        original_acc = original_results[0]
        print(f"原始模型准确率: {original_acc:.4f}")
        
        # 测试随机边结构的模型（仅使用节点特征）
        feature_only_acc_list = []
        for data in self.test_loader:
            if not self.args.multi_gpu:
                data = data.to(self.args.device)
                randomized_data = self.randomize_edges(data)
                out = self.original_model(randomized_data)
                y = data.y
                pred = out.max(1)[1]
                feature_only_acc_list.append(pred.eq(y).sum().item() / len(y))
            else:
                randomized_data = self.randomize_edges(data)
                out = self.original_model(randomized_data)
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
                pred = out.max(1)[1]
                feature_only_acc_list.append(pred.eq(y).sum().item() / len(y))
        
        feature_only_acc = sum(feature_only_acc_list) / len(feature_only_acc_list)
        print(f"仅使用节点特征的准确率: {feature_only_acc:.4f}")
        
        # 测试随机节点特征的模型（仅使用图结构）
        structure_only_acc_list = []
        for data in self.test_loader:
            if not self.args.multi_gpu:
                data = data.to(self.args.device)
                randomized_data = self.randomize_features(data)
                out = self.original_model(randomized_data)
                y = data.y
                pred = out.max(1)[1]
                structure_only_acc_list.append(pred.eq(y).sum().item() / len(y))
            else:
                randomized_data = self.randomize_features(data)
                out = self.original_model(randomized_data)
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
                pred = out.max(1)[1]
                structure_only_acc_list.append(pred.eq(y).sum().item() / len(y))
        
        structure_only_acc = sum(structure_only_acc_list) / len(structure_only_acc_list)
        print(f"仅使用图结构的准确率: {structure_only_acc:.4f}")
        
        # 计算相对重要性
        total_performance = original_acc
        feature_importance = (feature_only_acc - 0.5) / (total_performance - 0.5) * 100 if total_performance > 0.5 else 0
        structure_importance = (structure_only_acc - 0.5) / (total_performance - 0.5) * 100 if total_performance > 0.5 else 0
        
        print(f"\n特征与结构重要性分析结果:")
        print(f"节点特征相对重要性: {feature_importance:.2f}%")
        print(f"图结构相对重要性: {structure_importance:.2f}%")
        
        return {
            "original_acc": original_acc,
            "feature_only_acc": feature_only_acc,
            "structure_only_acc": structure_only_acc,
            "feature_importance": feature_importance,
            "structure_importance": structure_importance
        }


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

	# 如果启用了特征重要性分析模式
	if args.analysis_mode == 'feature_importance':
		print("\n开始进行节点特征与图结构重要性分析...")
		analyzer = FeatureImportanceAnalyzer(args, model, test_loader)
		importance_results = analyzer.evaluate_importance()
		
		print("\n分析总结:")
		if importance_results["feature_importance"] > importance_results["structure_importance"]:
			print(f"在{args.dataset}数据集上，节点特征对假新闻检测的贡献明显大于图结构。")
			print(f"这表明文本内容特征是判断新闻真假的关键因素。")
		elif importance_results["structure_importance"] > importance_results["feature_importance"]:
			print(f"在{args.dataset}数据集上，图结构对假新闻检测的贡献明显大于节点特征。")
			print(f"这表明新闻传播的社交网络结构是判断新闻真假的关键因素。")
		else:
			print(f"在{args.dataset}数据集上，节点特征和图结构对假新闻检测的贡献相近。")
			print(f"这表明文本内容和传播结构都是判断新闻真假的重要因素。")
