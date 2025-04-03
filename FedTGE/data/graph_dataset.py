import torch
from torch_geometric.data import Data
import numpy as np


def convert_optgdba_to_fedtge(optgdba_graphs):
    """将OPTGDBA格式的图数据转换为FEDTGE格式"""
    fedtge_graphs = []

    for graph in optgdba_graphs:
        # 转换边索引
        edges = [list(pair) for pair in graph.g.edges()]
        edges.extend([[j, i] for i, j in edges])  # 添加反向边
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        # 转换节点特征
        node_features = graph.node_features

        # 转换图标签
        graph_label = torch.tensor([graph.label], dtype=torch.long)

        # 创建PyG数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=graph_label,
        )

        # 添加度信息作为节点特征
        if hasattr(graph, 'node_tags'):
            degree_features = torch.zeros((node_features.shape[0], 1))
            for i, deg in enumerate(graph.node_tags):
                degree_features[i] = deg
            data.degree = degree_features

        fedtge_graphs.append(data)

    return fedtge_graphs


def prepare_batch(graphs, batch_size):
    """准备批次数据"""
    n = len(graphs)
    indices = torch.randperm(n)

    for i in range(0, n, batch_size):
        batch_indices = indices[i:min(i + batch_size, n)]
        batch_graphs = [graphs[idx] for idx in batch_indices]

        # 组装批次数据
        batch_x = []
        batch_edge_index = []
        batch_y = []
        batch_ptr = [0]

        node_offset = 0
        for graph in batch_graphs:
            num_nodes = graph.x.size(0)

            batch_x.append(graph.x)
            batch_edge_index.append(graph.edge_index + node_offset)
            batch_y.append(graph.y)

            node_offset += num_nodes
            batch_ptr.append(node_offset)

        # 将列表转换为张量
        batch_x = torch.cat(batch_x, dim=0)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        batch_y = torch.cat(batch_y)
        batch_ptr = torch.tensor(batch_ptr)

        yield Data(x=batch_x,
                   edge_index=batch_edge_index,
                   y=batch_y,
                   batch=batch_ptr)


def split_graph_data(graphs, train_ratio=0.8, val_ratio=0.1, seed=42):
    """划分数据集为训练集、验证集和测试集"""
    np.random.seed(seed)

    n = len(graphs)
    indices = np.random.permutation(n)

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]

    return train_graphs, val_graphs, test_graphs