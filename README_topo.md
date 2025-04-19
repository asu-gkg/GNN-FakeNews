# 假新闻检测 - 拓扑特征分析工具

这个工具用于分析假新闻检测数据集中的图结构拓扑特征，提取关键图论指标并可视化结果，帮助理解真假新闻在传播网络结构上的差异。

## 功能特点

- 从JSON格式的图数据中提取丰富的拓扑特征
- 对比真假新闻在拓扑特征上的差异
- 生成多种可视化图表（箱线图、散点图、直方图、热图等）
- 使用随机森林评估拓扑特征的分类效果
- 分析特征重要性，找出最具区分性的网络结构特征

## 依赖库

```
numpy
pandas
matplotlib
seaborn
networkx
scikit-learn
```

## 使用方法

1. 准备数据集：确保数据按照以下结构组织
   ```
   data/
   ├── politifact/
   │   ├── graphs/
   │   │   ├── news_id_1.json
   │   │   ├── news_id_2.json
   │   │   └── ...
   │   └── labels.json
   └── gossipcop/
       ├── graphs/
       │   ├── news_id_1.json
       │   ├── news_id_2.json
       │   └── ...
       └── labels.json
   ```

2. 图文件格式示例（JSON）：
   ```json
   {
     "nodes": [
       {"id": "node1"},
       {"id": "node2"},
       {"id": "node3"}
     ],
     "edges": [
       {"source": "node1", "target": "node2"},
       {"source": "node2", "target": "node3"}
     ]
   }
   ```

3. 标签文件格式示例（JSON）：
   ```json
   {
     "news_id_1": 0,
     "news_id_2": 1,
     "news_id_3": 0
   }
   ```
   其中 0 表示真新闻，1 表示假新闻。

4. 运行分析脚本：
   ```bash
   python topo_analysis.py
   ```

5. 检查输出结果：
   - 所有结果将保存在 `topo_analysis_results` 目录下
   - 统计数据：CSV格式的描述性统计和特征重要性
   - 可视化图表：多种格式的可视化结果

## 提取的拓扑特征

该工具会从传播图中提取以下拓扑特征：

- **Node Count**: 节点数量（参与传播的用户数）
- **Edge Count**: 边数量（传播关系数量）
- **Average Degree**: 平均度
- **Max Degree**: 最大度
- **Degree Std**: 度的标准差
- **Degree Centrality**: 度中心性平均值
- **Clustering Coefficient**: 聚类系数
- **Graph Density**: 图密度
- **Connected Components**: 连通分量数量
- **Largest CC Size**: 最大连通分量大小
- **Largest CC Ratio**: 最大连通分量所占比例
- **Diameter**: 图直径（最大连通分量中的最长最短路径）
- **Avg Shortest Path**: 平均最短路径长度

## 注意事项

- 如果无法找到真实数据，脚本会生成合成数据用于演示目的
- 可以通过修改代码中的 `load_dataset` 函数来适应不同的数据格式
- 对于大型数据集，某些计算密集型特征（如直径）可能需要较长时间

## 示例分析结果

完成分析后，您将获得：

1. 拓扑特征分布的箱线图
2. 区分性最强的特征对散点图
3. 传播图大小分布直方图
4. 拓扑特征之间的相关性热图
5. 特征重要性条形图（F分数和随机森林）
6. 分类性能报告

通过这些可视化和分析结果，您可以深入了解真假新闻在网络拓扑结构上的差异特点。 