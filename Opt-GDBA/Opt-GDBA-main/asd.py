import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import OptGDBA as OptGDBA
from mask import gen_mask
from input import gen_input
from util import *
from graphcnn import Discriminator
import pickle
import copy
from division import hash_func_map, features_func_map, division_func_map, get_node_id
criterion = nn.CrossEntropyLoss()   #交叉熵损失函数

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])    #创建一个新的权重字典，作为最终的结果字典。选择 w[0] 是因为所有权重字典结构相同，只需复制一个初始化即可。
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def bkd_cdd_test(graphs, target_label):  # 筛选出标签不等于 target_label 的图的索引。

    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label:
            backdoor_graphs_indexes.append(graph_idx)

    return backdoor_graphs_indexes


def bkd_cdd(num_backdoor_train_graphs, graphs, target_label,
            dataset):  # 从给定的图数据集中选择一些“后门图”（backdoor graphs），这些图的标签不是目标标签 target_label，并将它们返回
    if dataset == 'MUTAG':
        num_backdoor_train_graphs = 1

    temp_n = 0
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label and temp_n < num_backdoor_train_graphs:
            backdoor_graphs_indexes.append(graph_idx)
            temp_n += 1

    return backdoor_graphs_indexes


def init_trigger(args, x, bkd_gids: list, bkd_nid_groups: list, init_feat: float):
    if init_feat == None:
        init_feat = - 1
        print('init feat == None, transferred into -1')

    graphs = copy.deepcopy(x)
    for idx in bkd_gids:

        edges = [list(pair) for pair in graphs[idx].g.edges()]
        edges.extend([[i, j] for j, i in edges])

        for i in bkd_nid_groups[idx]:
            for j in bkd_nid_groups[idx]:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i, j) in graphs[idx].g.edges():
                    graphs[idx].g.remove_edge(i, j)
        edge_mat_temp = torch.zeros(len(graphs[idx].g), len(graphs[idx].g))
        for [x_i, y_i] in edges:
            edge_mat_temp[x_i, y_i] = 1
        graphs[idx].edge_mat = edge_mat_temp
        # change graph labels
        assert args.target is not None
        graphs[idx].label = args.target
        graphs[idx].node_tags = list(dict(graphs[idx].g.degree).values())

        # change features in-place
        featdim = graphs[idx].node_features.shape[1]
        a = np.array(graphs[idx].node_features)
        a[bkd_nid_groups[idx]] = np.ones((len(bkd_nid_groups[idx]), featdim)) * init_feat
        graphs[idx].node_features = torch.Tensor(a.tolist())

    return graphs


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--port', type=str, default="acm4",
                        help='name of sever')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--num_agents', type=int, default=20,
                        help="number of agents:n")
    parser.add_argument('--num_corrupt', type=int, default=4,
                        help="number of corrupt agents")
    parser.add_argument('--frac_epoch', type=float, default=0.5,
                        help='fraction of users are chosen')
    parser.add_argument('--is_Customized', type=int, default=0,
                        help='is_Customized')
    parser.add_argument('--is_test', type=int, default=0,
                        help='is_test')
    parser.add_argument('--is_defense', type=int, default=0,
                        help='is_defense')
    parser.add_argument('--triggersize', type=int, default=4,
                        help='number of nodes in a clique (trigger size)')
    parser.add_argument('--target', type=int, default=0,
                        help='targe class (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=1,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n_epoch', type=int, default=1,
                        help='Ratio of training rounds')
    parser.add_argument('--num_backdoor_train_graphs', type=int, default=1,
                        help='Ratio of malicious training data -> number')
    parser.add_argument('--n_train_D', type=int, default=1,
                        help='training rounds')
    parser.add_argument('--n_train_G', type=int, default=1,
                        help='training rounds')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='coefficient')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=False,
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true", default=True,
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--topo_thrd', type=float, default=0.5,
                        help="threshold for topology generator")
    parser.add_argument('--gtn_layernum', type=int, default=3,
                        help="layer number of GraphTrojanNet")
    parser.add_argument('--topo_activation', type=str, default='sigmoid',
                        help="activation function for topology generator")
    parser.add_argument('--feat_activation', type=str, default='relu',
                        help="activation function for feature generator")
    parser.add_argument('--feat_thrd', type=float, default=0,
                        help="threshold for feature generator (only useful for binary feature)")
    parser.add_argument('--filename', type=str, default="output",
                        help='output file')
    parser.add_argument('--filenamebd', type=str, default="output_bd",
                        help='output backdoor file')
    parser.add_argument("--num_group", type=int, default=6,
                        help="Number of groups to divide the graph into")
    args = parser.parse_args()
    args.features_func = features_func_map.get('id', get_node_id)
    args.hash_func = hash_func_map.get('md5', hash)
    args.division_func = division_func_map.get('node', None)
    # args.degree_as_tag = bool(args.degree_as_tag)

    cpu = torch.device('cpu')
    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #########数据加载###############
    graphs, num_classes, tag2index = load_data(args.dataset, args.degree_as_tag)
    train_graphs, test_graphs, test_idx = separate_data(graphs, args.seed, args.fold_idx)
    print('#train_graphs:', len(train_graphs), '#test_graphs:', len(test_graphs))

    print('input dim:', train_graphs[0].node_features.shape[1])

    train_data_size = len(train_graphs)
    client_data_size = int(train_data_size / args.num_agents)
    split_data_size = [client_data_size for _ in range(args.num_agents - 1)]
    split_data_size.append(train_data_size - client_data_size * (args.num_agents - 1))

    # ✅ 先划分数据集给各个客户端
    train_graphs = torch.utils.data.random_split(train_graphs, split_data_size)

    # ✅ 随机选择部分客户端作为恶意客户端
    malicious_clients = np.random.choice(range(args.num_agents), args.num_corrupt, replace=False)
    print(f"恶意客户端列表: {malicious_clients}")

    # ✅ 初始化全局模型（善意客户端）
    global_model = Discriminator(args.num_layers, args.num_mlp_layers, train_graphs[0][0].node_features.shape[1],
                                 args.hidden_dim, num_classes, args.final_dropout, args.learn_eps,
                                 args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer_D = optim.Adam(global_model.parameters(), lr=args.lr)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)

    # ✅ 复制测试集（用于后门测试）
    test_graphs_trigger = copy.deepcopy(test_graphs)
    test_backdoor = bkd_cdd_test(test_graphs_trigger, args.target)

    # ✅ 计算图的节点数
    train_nodenums = [len(train_graphs[idx][0].g.adj) for idx in range(len(train_graphs))]
    test_nodenums = [len(test_graphs[idx].g.adj) for idx in range(len(test_graphs))]
    nodemax = max(max(train_nodenums), max(test_nodenums))  # 取最大值
    featdim = train_graphs[0][0].node_features.shape[1]

    # ✅ 初始化后门攻击生成器
    generator = {}
    optimizer_G = {}
    scheduler_G = {}
    for g_i in range(args.num_corrupt):
        generator[g_i] = OptGDBA.Generator(nodemax, featdim, args.gtn_layernum, args.triggersize)
        optimizer_G[g_i] = torch.optim.Adam(generator[g_i].parameters(), lr=args.lr)
        scheduler_G[g_i] = torch.optim.lr_scheduler.StepLR(optimizer_G[g_i], step_size=50, gamma=0.1)



    # ✅ 初始化测试数据
    Ainput_test, Xinput_test = gen_input(test_graphs_trigger, test_backdoor, nodemax)

    with open(args.filenamebd, 'w+') as f:
        f.write("acc_train acc_clean acc_backdoor\n")

        bkd_gids_train = {}
        Ainput_train = {}
        Xinput_train = {}
        nodenums_id = {}
        train_graphs_trigger = {}


        # ✅ 生成后门数据并替换恶意客户端的数据
        # ✅ 遍历客户端，恶意客户端执行后门攻击，善意客户端保持原始数据
        # ✅ 生成后门数据并替换恶意客户端的数据
        for id in range(args.num_agents):
            print(f"🔍 处理客户端 {id}")

            train_graphs_trigger[id] = copy.deepcopy(train_graphs[id])
            nodenums_id[id] = torch.tensor(
                [len(train_graphs_trigger[id][idx].g.adj) for idx in range(len(train_graphs_trigger[id]))]
            ).to(device)

            if id in malicious_clients:
                print(f"⚠️ 客户端 {id} 是恶意客户端，执行后门攻击")

                # ✅ 选择后门样本
                bkd_gids_train[id] = bkd_cdd(args.num_backdoor_train_graphs, train_graphs_trigger[id], args.target,
                                             args.dataset)
                print(f"🔍 Debug: bkd_gids_train[{id}] = {bkd_gids_train[id]}")

                if len(bkd_gids_train[id]) == 0:
                    print(f"⚠️ Warning: bkd_gids_train[{id}] 为空，可能未选择到后门样本")
                    continue

                    # ✅ 生成后门输入数据
                Ainput_train[id], Xinput_train[id] = gen_input(train_graphs_trigger[id], bkd_gids_train[id], nodemax)

                # ✅ 确保 `Ainput_train[id]` 是 `tensor` 而不是 `dict`
                if isinstance(Ainput_train[id], dict) and len(Ainput_train[id]) > 0:
                    print(f"📌 Debug: Ainput_train[{id}] keys = {list(Ainput_train[id].keys())}")
                else:
                    print(f"⚠️ Warning: Ainput_train[{id}] 为空，使用默认值填充")
                    Ainput_train[id] = {gid: torch.zeros((1, nodemax)).to(device) for gid in bkd_gids_train[id]}

                # ✅ 处理 `Xinput_train[id]`
                if isinstance(Xinput_train[id], dict) and len(Xinput_train[id]) > 0:
                    print(f"📌 Debug: Xinput_train[{id}] keys = {list(Xinput_train[id].keys())}")
                else:
                    print(f"⚠️ Warning: Xinput_train[{id}] 为空，使用默认值填充")
                    Xinput_train[id] = {gid: torch.zeros((1, nodemax)).to(device) for gid in bkd_gids_train[id]}

                # ✅ 生成后门数据
                print(f"🔍 Debug: 生成后门数据，使用 generator[{id % args.num_corrupt}]")
                for gid in bkd_gids_train[id]:
                    if gid not in Ainput_train[id]:
                        print(f"⚠️ Warning: gid={gid} 不在 Ainput_train[{id}] 中，跳过")
                        continue  # 跳过不存在的数据

                    train_graphs_trigger[id], _, _, _, _, _ = generator[id % args.num_corrupt](
                        args, id, train_graphs_trigger[id], bkd_gids_train[id],
                        Ainput_train[id], Xinput_train[id], nodenums_id[id], nodemax,
                        args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False
                    )


            else:
                print(f"✅ 客户端 {id} 是善意客户端，保留原始数据")
                bkd_gids_train[id] = []
                Ainput_train[id] = None
                Xinput_train[id] = None

        print("✅ 数据处理完成，准备训练")

        # ✅ 保存全局模型参数
        global_weights = global_model.state_dict()


        for epoch in tqdm(range(1, args.epochs + 1)):
            local_weights, local_losses = [], []
            m = max(int(args.frac_epoch * args.num_agents), 1)
            idxs_users = np.random.choice(range(args.num_agents), m, replace=False)
            print(f"=== 选中 {m} 个客户端进行训练: {idxs_users} ===")

            for id in idxs_users:
                print(f"=== 客户端 {id} 训练开始 | Epoch {epoch}/{args.epochs} ===")

                global_model.load_state_dict(copy.deepcopy(global_weights))
                global_model.train()
                loss_accum = 0

                for _ in range(10):
                    # ✅ 选取本轮训练数据
                    selected_idx = np.random.permutation(len(train_graphs_trigger[id]))[:args.batch_size]
                    batch_graph = [train_graphs_trigger[id][idx] for idx in selected_idx]

                    # ✅ 计算损失
                    output = global_model(batch_graph, is_Customized=args.is_Customized, is_test=args.is_test, trigger_size=args.triggersize)

                    labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
                    criterion.reduction = "none"
                    raw_loss = criterion(output, labels)

                    # ✅ 执行 ASD 筛选
                    loss_values = raw_loss.cpu().detach().numpy()
                    if epoch < args.epochs * 0.33:
                        loss_indices = np.argsort(loss_values)[:int(len(loss_values) * 0.5)]  # 选取 50% 低损失数据
                    elif epoch < args.epochs * 0.66:
                        loss_indices = np.argsort(loss_values)[:int(len(loss_values) * 0.4)]  # 选取 40% 低损失数据
                    else:
                        loss_indices = np.argsort(loss_values)[:int(len(loss_values) * 0.3)]  # 选取 30% 低损失数据

                    batch_graph = [batch_graph[i] for i in loss_indices]

                    # ✅ 计算最终损失
                    final_output = global_model(batch_graph)
                    final_labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
                    loss = criterion(final_output, final_labels)

                    optimizer_D.zero_grad()
                    loss.backward()
                    optimizer_D.step()

                    loss_accum += loss.item()

                avg_loss = loss_accum / args.iters_per_epoch
                local_losses.append(avg_loss)
                local_weights.append(copy.deepcopy(global_model.state_dict()))

                print(f"客户端 {id} | Epoch {epoch}: ASD 筛选 {len(batch_graph)} 个低损失样本，平均损失: {avg_loss:.4f}")

            # ✅ 联邦平均 (FedAvg)
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            print(f"=== Epoch {epoch} 训练完成，模型已聚合 ===")

        # ✅ 评估阶段
        print("\n=== 评估 ASD 机制 ===")

        # 计算干净测试集上的准确率
        output_clean = global_model(test_graphs)
        predicted_labels_clean = torch.argmax(output_clean, dim=1)
        acc_clean = (predicted_labels_clean.cpu().numpy() == np.array([g.label for g in test_graphs])).mean() * 100

        # 计算后门攻击成功率 (ASR)
        output_backdoor = global_model(test_graphs_trigger, is_Customized=args.is_Customized, is_test=True,
                                       trigger_size=args.triggersize)

        predicted_labels = torch.argmax(output_backdoor, dim=1)
        asr = (predicted_labels.cpu().numpy() == args.target).mean() * 100

        acc_backdoor = 100 - asr  # 后门攻击成功率越低，说明 ASD 防御越成功

        print(f"✅ 最终模型准确率: {acc_clean:.2f}%")
        print(f"✅ ASD 防御后门成功率: {acc_backdoor:.2f}%")
        print(f"=== 后门攻击成功率 (ASR): {asr:.2f}% ===")


if __name__ == '__main__':
    main()



