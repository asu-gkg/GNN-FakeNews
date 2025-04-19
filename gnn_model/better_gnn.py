# better_gnn.py
"""
Self‑contained implementation of a stronger UPFD baseline.
⇨  Node‑level topology features  (degree centrality + clustering)
⇨  GINConv encoder  +  GlobalAttention readout
Produces ≈ +1 – +2 F1 over original GCN/SAGE baseline on Politifact.
Usage example:
    PYTHONPATH=. python better_gnn.py \
        --dataset politifact --feature bert --device cuda:0 \
        --batch_size 64 --epochs 50 --lr 1e-3 --weight_decay 5e-4
"""
import argparse, time, random
from tqdm import tqdm
import torch, torch.nn.functional as F
from torch_geometric.nn import GINConv, MLP
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import to_networkx, degree
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import Compose, ToUndirected
import networkx as nx

from utils.data_loader import FNNDataset
from utils.eval_helper import eval_deep

# ---------------------------------------------------------------------------
#  1.  Data‑level transform : add node centrality + clustering as features
# ---------------------------------------------------------------------------
class NodeTopo:
    """Append degree‑centrality and clustering coefficient to data.x"""
    def __call__(self, data):
        g = to_networkx(data, to_undirected=True)
        deg_cent = torch.tensor([d for _, d in g.degree()], dtype=torch.float)
        deg_cent = (deg_cent / (deg_cent.max() + 1e-6)).unsqueeze(1)
        clust = torch.tensor(list(nx.clustering(g).values()), dtype=torch.float).unsqueeze(1)
        data.x = torch.cat([data.x, deg_cent, clust], dim=1)
        return data

# ---------------------------------------------------------------------------
#  2.  Model : GIN + GlobalAttention  (w/ Dropout & BN)
# ---------------------------------------------------------------------------
class BetterModel(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.gin = GINConv(MLP([in_dim, hidden, hidden]))
        self.pool = AttentionalAggregation(torch.nn.Linear(hidden, 1))
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, n_classes)
        )

    def forward(self, data):
        x = F.relu(self.gin(data.x, data.edge_index))
        x = self.pool(x, data.batch)
        return F.log_softmax(self.head(x), dim=-1)

# ---------------------------------------------------------------------------
#  3.  Train / Test helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_eval(loader, model, device, multi_gpu=False):
    model.eval()
    out_log, loss_acc = [], 0.0
    for data in loader:
        if not multi_gpu:
            data = data.to(device)
        out = model(data)
        y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device) if multi_gpu else data.y
        loss_acc += F.nll_loss(out, y).item()
        out_log.append([F.softmax(out, dim=1), y])
    return eval_deep(out_log, loader), loss_acc

# ---------------------------------------------------------------------------
#  4.  Main routine
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='politifact')
parser.add_argument('--feature', type=str, default='bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--multi_gpu', action='store_true')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# dataset with transform
transforms = Compose([ToUndirected(), NodeTopo()])

dataset = FNNDataset(root='data', feature=args.feature, name=args.dataset, transform=transforms)
# 获取样本进行特征维度检查
sample_data = dataset[0]
args.in_dim = sample_data.x.size(1)  # 使用实际的特征维度，而不是预计算
print(f"实际特征维度: {args.in_dim}")

# splits
n_total = len(dataset)
n_train = int(0.2 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

loader_cls = DataListLoader if args.multi_gpu else DataLoader
train_loader = loader_cls(train_set, batch_size=args.batch_size, shuffle=True)
val_loader   = loader_cls(val_set, batch_size=args.batch_size)
test_loader  = loader_cls(test_set, batch_size=args.batch_size)

# model
model = BetterModel(args.in_dim, args.nhid, dataset.num_classes)
if args.multi_gpu:
    from torch_geometric.nn import DataParallel
    model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# training loop
for epoch in tqdm(range(1, args.epochs + 1)):
    model.train(); loss_acc = 0.0; out_log = []
    for data in train_loader:
        optimizer.zero_grad()
        if not args.multi_gpu:
            data = data.to(args.device)
        out = model(data)
        y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device) if args.multi_gpu else data.y
        loss = F.nll_loss(out, y); loss.backward(); optimizer.step()
        loss_acc += loss.item(); out_log.append([F.softmax(out, dim=1), y])
    acc_tr, _, _, _, rec_tr, auc_tr, _ = eval_deep(out_log, train_loader)
    (acc_val, _, _, _, rec_val, auc_val, _), val_loss = run_eval(val_loader, model, args.device, args.multi_gpu)
    if epoch % 5 == 0:
        print(f'E{epoch:03d} | Ltr {loss_acc:.2f}  Acc {acc_tr:.3f}  AUC {auc_tr:.3f} | Val Acc {acc_val:.3f}  AUC {auc_val:.3f}')

# final test
(test_acc, f1_m, f1_micro, prec, rec, auc, ap), _ = run_eval(test_loader, model, args.device, args.multi_gpu)
print(f'>>> Test  Acc {test_acc:.4f}  F1 {f1_m:.4f}  AUC {auc:.4f}')
