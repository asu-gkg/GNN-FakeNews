import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--feature', type=str, default='profile', help='[profile, bert, spacy, content]')
parser.add_argument('--model', type=str, default='gcn', help='[gcn, sage, gat]')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--use_topo', type=bool, default=True, help='whether to use topological features')

args = parser.parse_args()

# 设置命令
if args.use_topo:
    cmd = f"PYTHONPATH=. python gnn_model/gnn_topo.py --dataset {args.dataset} --model {args.model} --feature {args.feature} --batch_size {args.batch_size} --topo_features True"
else:
    cmd = f"PYTHONPATH=. python gnn_model/gnn.py --dataset {args.dataset} --model {args.model} --feature {args.feature} --batch_size {args.batch_size}"

# 运行模型
print(f"执行命令: {cmd}")
os.system(cmd)

print("\n模型运行完成!")
print(f"数据集: {args.dataset}")
print(f"特征类型: {args.feature}")
print(f"模型类型: {args.model}")
print(f"拓扑特征: {'已使用' if args.use_topo else '未使用'}")
print("\n下一步建议尝试:")
print("1. 运行analyze_topo_features.py查看图拓扑特征分析")
print("2. 修改--use_topo参数比较有无拓扑特征的模型差异")
print("3. 尝试不同的数据集、特征类型或模型结构") 