import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import copy
import random
import numpy as np
from models.graph_gcn import GraphGCN
from data.graph_dataset import convert_optgdba_to_fedtge, prepare_batch, split_graph_data


def client_update(args, client_graphs, global_model):
    """客户端本地训练"""
    local_model = copy.deepcopy(global_model)
    local_model.train()

    optimizer = optim.Adam(local_model.parameters(), lr=args.lr)

    for epoch in range(args.local_epochs):
        for batch in prepare_batch(client_graphs, args.batch_size):
            batch = batch.to(args.device)

            optimizer.zero_grad()
            output = local_model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(output, batch.y)

            loss.backward()
            optimizer.step()

    return local_model.state_dict()


def average_weights(w):
    """联邦平均"""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def evaluate(model, test_graphs, device, batch_size=32):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in prepare_batch(test_graphs, batch_size):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            pred = output.max(1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total


def train_federated(args, graphs):
    """联邦训练主函数"""
    # 1. 数据预处理
    fedtge_graphs = convert_optgdba_to_fedtge(graphs)
    train_graphs, val_graphs, test_graphs = split_graph_data(fedtge_graphs)

    # 2. 划分客户端数据
    num_clients = args.num_clients
    client_graphs = [[] for _ in range(num_clients)]
    for i, graph in enumerate(train_graphs):
        client_idx = i % num_clients
        client_graphs[client_idx].append(graph)

    # 3. 初始化全局模型
    input_dim = train_graphs[0].x.shape[1]
    global_model = GraphGCN(
        nfeat=input_dim,
        nhid=args.hidden_dim,
        nclass=args.num_classes,
        dropout=args.dropout,
        graph_pooling=args.graph_pooling,
        device=args.device
    ).to(args.device)

    # 4. 联邦训练
    best_acc = 0
    for round in range(args.num_rounds):
        # 选择客户端
        num_selected = max(1, int(args.frac * num_clients))
        selected_clients = random.sample(range(num_clients), num_selected)

        # 客户端本地训练
        local_weights = []
        for client_id in selected_clients:
            weights = client_update(args, client_graphs[client_id], global_model)
            local_weights.append(weights)

        # 聚合模型
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # 评估
        val_acc = evaluate(global_model, val_graphs, args.device)
        test_acc = evaluate(global_model, test_graphs, args.device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(global_model.state_dict())

        print(f'Round {round + 1}/{args.num_rounds}:')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')

    # 加载最佳模型
    global_model.load_state_dict(best_model_weights)
    final_test_acc = evaluate(global_model, test_graphs, args.device)
    print(f'Final Test Accuracy: {final_test_acc:.4f}')

    return global_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--graph_pooling', type=str, default='add')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载OPTGDBA数据
    from util import load_data

    graphs, num_classes, tag2index = load_data("MUTAG", degree_as_tag=True)

    # 开始训练
    model = train_federated(args, graphs)