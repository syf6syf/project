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

#联邦平均公式，用于计算权重的平均值
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

def train(args, model, device, train_graphs, optimizer, epoch, tag2index):
    model.train()

    total_iters = args.iters_per_epoch    #训练迭代轮数
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0   #总损失累计
    #for pos in pbar:
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size] #  从训练数据集中随机选择一个批次的数据索引

        batch_graph = [train_graphs[idx] for idx in selected_idx]  # 选择出相应的图数据，形成 一个批次的训练数据

        batch_graph = [g for graph in batch_graph for g in args.division_func(graph, args)] # 干净训练图划分

        #output = model(batch_graph)
        output = pass_data_iteratively_(model, batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss 计算损失
        loss = criterion(output, labels)

        # backprop   
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        #pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    #print("loss training: %f" % (average_loss))

    return average_loss   #返回平均损失

def train_G(args, model, generator, id, device, train_graphs_trigger, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    model.eval()#train()     将模型调为评估模式，让model进行评估该生成器所产生的触发器是否能达到理想结果（标签分类结果）
    generator.train()     #将生成器模型调为训练模式
    total_iters = 1 #args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    loss_poison_total = 0
    #for pos in pbar:
    #local_feat = torch.zeros(nodemax,nodemax)
    for pos in range(total_iters):
        selected_idx = bkd_gids_train #np.random.permutation(len(train_graphs))[:args.batch_size]
        sub_loss = nn.MSELoss() 
        batch_graph = [train_graphs_trigger[idx] for idx in selected_idx]

        output_graph, trigger_group, edges_len, nodes_len, trigger_id, trigger_l = generator(args, id, train_graphs_trigger, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
        output = model(output_graph)
        output_graph_poison = torch.stack([output[idx] for idx in selected_idx])

        labels_poison = torch.LongTensor([args.target for idx in selected_idx]).to(device)

        loss_poison = criterion(output_graph_poison, labels_poison)

        loss = sub_loss(trigger_id, trigger_l.detach()) #Intermediate Supervision
    average_loss = 0
    return loss, loss_poison, edges_len, nodes_len

def train_D(args, model, generator, id, device, train_graphs_trigger, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    model.train()    #将模型调为训练模式
    generator.eval()#train()   将触发器生成器调为评估模式
    total_iters = args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    loss_poison_total = 0
    #for pos in pbar:
    for pos in range(total_iters):
        selected_idx = bkd_gids_train 

        batch_graph = [train_graphs_trigger[idx] for idx in selected_idx]

        output_graph, _, _, _, _, _ = generator(args, id, train_graphs_trigger, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
        # print(' id:',id,' #train_graphs:', len(output_graph))
        # output_graph = sum([args.division_func(graph, args) for graph in tqdm(output_graph)], start=[])
        # print(' id:',id,' #train_graphs_cut:', len(output_graph))

        output = model(output_graph)
        labels = torch.LongTensor([graph.label for graph in output_graph]).to(device)

        # compute loss
        loss = criterion(output, labels) 
        loss_accum += loss

    average_loss = loss_accum / total_iters

    return average_loss
def optimize_D(loss, global_model, optimizer_D):
    global_model.zero_grad() 
    optimizer_D.zero_grad()
    loss.backward()
    optimizer_D.step()
    return
def optimize_G(alpha, loss1, loss2, model, optimizer_G):
    model.zero_grad()
    optimizer_G.zero_grad()
    loss =  alpha * loss1 + loss2 
    loss.backward()
    optimizer_G.step()
    return
###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=1):
    model.eval()
    output = []
    # idx = np.arange(len(graphs))
    # **确保 `graphs` 是列表**
    if isinstance(graphs, S2VGraph):
        graphs = [graphs]  # 转换为列表

    idx = np.arange(len(graphs))
    for i in  tqdm(range(0, len(graphs), minibatch_size)):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def pass_data_iteratively_(model, graphs, minibatch_size=1):
    model.eval()
    output = []
    # idx = np.arange(len(graphs))
    # **确保 `graphs` 是列表**
    if isinstance(graphs, S2VGraph):
        graphs = [graphs]  # 转换为列表

    idx = np.arange(len(graphs))
    for i in  tqdm(range(0, len(graphs), minibatch_size)):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]))
    return torch.cat(output, 0)

def test_clean_no(args, model, device, test_graphs, tag2index): #用于划分前 干净图 测试
    model.eval()

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    #print("pred:",pred)

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # print(labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy test（干净图分割前）: %f" % acc_test)

    return acc_test

def test_clean(args, model, device, test_graph_groups, tag2index): #用于划分后 干净图 测试
    """
    按原始图的子图分组进行预测，并返回最终的多数投票预测结果
    :param args: 传入的参数
    :param model: 训练好的 GNN 模型
    :param device: 运行设备
    :param test_graph_groups: 二级列表，每个元素是一个原始图的所有子图
    :param tag2index: 标签索引映射
    :return: 原始图的最终预测精度
    """
    model.eval()
    final_predictions = []  # 存储每个原始图的最终预测

    for idx, subgraphs in enumerate(test_graph_groups):  # 遍历每个原始图的子图组
        if not subgraphs:  # 防止空子图组
            continue

        # for i, subgraph in enumerate(subgraphs):
        #     print(f"子图 {i}: 节点数={len(subgraph.g.nodes)}, 边数={len(subgraph.g.edges)}")

        # **对子图进行模型预测**
        output = pass_data_iteratively(model, subgraphs)  # 预测所有子图
        preds = output.max(1, keepdim=True)[1]  # 获取预测标签
        preds = preds.view(-1).tolist()  # 转换为列表

        # **多数投票**
        most_common_pred = Counter(preds).most_common(1)[0][0]
        final_predictions.append(most_common_pred)  # 存储最终预测

        # **打印该原始图的子图预测结果**
        print(f"原始图 {idx + 1} 的干净子图预测结果: {preds} -> 多数投票: {most_common_pred}")

    # **获取原始图的真实标签**
    labels = torch.LongTensor([subgraphs[0].label for subgraphs in test_graph_groups if isinstance(subgraphs, list) and len(subgraphs) > 0]).to(device)


    # **检查长度匹配**
    assert len(final_predictions) == len(labels), f"预测数量 {len(final_predictions)} 与真实标签数量 {len(labels)} 不匹配"

    # **计算最终精度**
    correct = sum([1 for i in range(len(final_predictions)) if final_predictions[i] == labels[i].item()])
    acc_test = correct / float(len(final_predictions))

    print("clean accuracy (干净图分割后 越大越好): %f" % acc_test)
    return acc_test

def test_back_no(args, model, device, test_graphs, tag2index):  #用于划分前 后门图 测试
    model.eval()

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    #print("pred:",pred)

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # print(labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy test（后门无分割）: %f" % acc_test)

    return acc_test

from collections import Counter



def test_back(args, model, device, test_graph_groups, tag2index): #用于划分后 后门图 测试
    """
    按原始图的子图分组进行预测，并返回最终的多数投票预测结果
    :param args: 传入的参数
    :param model: 训练好的 GNN 模型
    :param device: 运行设备
    :param test_graph_groups: 二级列表，每个元素是一个原始图的所有子图
    :param tag2index: 标签索引映射
    :return: 原始图的最终预测精度
    """
    model.eval()
    final_predictions = []  # 存储每个原始图的最终预测

    for idx, subgraphs in enumerate(test_graph_groups):  # 遍历每个原始图的子图组
        if not subgraphs:  # 防止空子图组
            continue

        # for i, subgraph in enumerate(subgraphs):
        #     print(f"子图 {i}: 节点数={len(subgraph.g.nodes)}, 边数={len(subgraph.g.edges)}")

        # **对子图进行模型预测**
        output = pass_data_iteratively(model, subgraphs)  # 预测所有子图
        preds = output.max(1, keepdim=True)[1]  # 获取预测标签
        preds = preds.view(-1).tolist()  # 转换为列表

        # **多数投票**
        most_common_pred = Counter(preds).most_common(1)[0][0]
        final_predictions.append(most_common_pred)  # 存储最终预测

        # **打印该原始图的子图预测结果**
        print(f"原始图 {idx + 1} 的后门子图预测结果: {preds} -> 多数投票: {most_common_pred}")

    # **获取原始图的真实标签**
    labels = torch.LongTensor([subgraphs[0].label for subgraphs in test_graph_groups if isinstance(subgraphs, list) and len(subgraphs) > 0]).to(device)


    # **检查长度匹配**
    assert len(final_predictions) == len(labels), f"预测数量 {len(final_predictions)} 与真实标签数量 {len(labels)} 不匹配"

    # **计算最终精度**
    correct = sum([1 for i in range(len(final_predictions)) if final_predictions[i] == labels[i].item()])
    acc_test = correct / float(len(final_predictions))

    print("Backdoor accuracy (越小说明防御住): %f" % acc_test)
    return acc_test



def bkd_cdd_test(graphs, target_label):  #筛选出标签不等于 target_label 的图的索引。
    
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label: 
            backdoor_graphs_indexes.append(graph_idx)
        
    return backdoor_graphs_indexes
def bkd_cdd(num_backdoor_train_graphs, graphs, target_label, dataset):  #从给定的图数据集中选择一些“后门图”（backdoor graphs），这些图的标签不是目标标签 target_label，并将它们返回
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
        edge_mat_temp = torch.zeros(len(graphs[idx].g),len(graphs[idx].g))
        for [x_i,y_i] in edges:
            edge_mat_temp[x_i,y_i] = 1
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
    #args.degree_as_tag = bool(args.degree_as_tag)

    cpu = torch.device('cpu')
    # set up seeds and gpu device
    torch.manual_seed(0) 
    np.random.seed(0) 
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    graphs, num_classes, tag2index = load_data(args.dataset, args.degree_as_tag)

    train_graphs, test_graphs, test_idx = separate_data(graphs, args.seed, args.fold_idx)
    print('#train_graphs:', len(train_graphs), '#test_graphs:', len(test_graphs))
    print('input dim:', train_graphs[0].node_features.shape[1])
    #划分数据
    
    
    train_data_size = len(train_graphs)
    client_data_size=int(train_data_size/(args.num_agents))
    split_data_size = [client_data_size for i in range(args.num_agents-1)]
    split_data_size.append(train_data_size-client_data_size*(args.num_agents-1))
    train_graphs = torch.utils.data.random_split(train_graphs,split_data_size)
    
    global_model = Discriminator(args.num_layers, args.num_mlp_layers, train_graphs[0][0].node_features.shape[1],
                        args.hidden_dim, \
                        num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                        args.neighbor_pooling_type, device).to(device)
    
    optimizer_D = optim.Adam(global_model.parameters(), lr=args.lr)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)

    test_graphs_trigger = copy.deepcopy(test_graphs)
    test_backdoor = bkd_cdd_test(test_graphs_trigger, args.target)

    #nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
    nodenums = [len(test_graphs[idx].g.adj) for idx in range(len(test_graphs))]
    nodemax = max(nodenums) 
    #featdim = np.array(self.benign_dr.data['features'][0]).shape[1] 
    featdim = train_graphs[0][0].node_features.shape[1]
    
    generator = {}
    optimizer_G = {}
    scheduler_G = {}
    for g_i in range(args.num_corrupt):
        generator[g_i] = OptGDBA.Generator(nodemax, featdim, args.gtn_layernum, args.triggersize)
        optimizer_G[g_i] = optim.Adam(generator[g_i].parameters(), lr=args.lr)
        scheduler_G[g_i] = optim.lr_scheduler.StepLR(optimizer_G[g_i], step_size=50, gamma=0.1)
    
    # init test data
    # NOTE: for data that can only add perturbation on features, only init the topo value
   
    Ainput_test, Xinput_test = gen_input(test_graphs_trigger, test_backdoor, nodemax) 
    
    with open(args.filenamebd, 'w+') as f:
        f.write("acc_train acc_clean acc_backdoor\n")
        bkd_gids_train = {}
        Ainput_train = {}
        Xinput_train = {}
        nodenums_id = {}
        train_graphs_trigger = {}
        
        for id in range(args.num_corrupt):          
            train_graphs_trigger[id] = copy.deepcopy(train_graphs[id])
            nodenums_id[id] = [len(train_graphs_trigger[id][idx].g.adj) for idx in range(len(train_graphs_trigger[id]))]
            bkd_gids_train[id] = bkd_cdd(args.num_backdoor_train_graphs, train_graphs_trigger[id], args.target, args.dataset)
            Ainput_train[id], Xinput_train[id] = gen_input(train_graphs_trigger[id], bkd_gids_train[id], nodemax)
        
        global_weights = global_model.state_dict()

        # 创建日志文件（在循环外）
        log_file = open("training_log_div6.txt", "w")
        log_file.write("Epoch, Loss,Clean_Accuracy_no,Clean_Accuracy,Backdoor_Accuracy_no,Backdoor_Accuracy\n")


        for epoch in tqdm(range(1, args.epochs + 1)):
            local_weights, local_losses = [], []
            m = max(int(args.frac_epoch * args.num_agents), 1)
            idxs_users = np.random.choice(range(args.num_agents), m, replace=False)
            print("idxs_users:", idxs_users)

            for id in idxs_users: 
                global_model.load_state_dict(copy.deepcopy(global_weights)) 
                if id < args.num_corrupt: 
                    train_graphs_trigger[id] = copy.deepcopy(train_graphs[id])
                    for kk in range(args.n_train_D):
                        loss = train_D(args, global_model, generator[id], id, device, train_graphs_trigger[id],
                                        epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                        Xinput_train[id], nodenums_id[id], nodemax, 
                                        binaryfeat=False)
                        optimize_D(loss, global_model, optimizer_D)
                    if epoch % args.n_epoch == 0:
                        for kk in range(args.n_train_G):
                            loss, loss_poison, edges_len, nodes_len = train_G(args, global_model, generator[id], id, device, train_graphs_trigger[id], 
                                            epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                            Xinput_train[id], nodenums_id[id], nodemax, 
                                            binaryfeat=False)
                            optimize_G(args.alpha, loss, loss_poison, generator[id], optimizer_G[id])
                else:
                    #train_graphs[id] = [subgraph for graph in tqdm(train_graphs[id]) for subgraph in args.division_func(graph, args)]
                    loss = train(args, global_model, device, train_graphs[id], optimizer_D, epoch, tag2index)

                l_weights = global_model.state_dict()
                local_weights.append(l_weights)
                local_losses.append(loss)

            scheduler_D.step()     
            global_weights = average_weights(local_weights)   
            global_model.load_state_dict(global_weights)
             
            loss_avg = sum(local_losses) / len(local_losses)    
            
            #----------------- Evaluation -----------------#
            if epoch%5 ==0:
                id = 0
                args.is_test = 1
                test_backdoor0 = copy.deepcopy(test_backdoor)
                generator[id].eval()
                nodenums_test = [len(test_graphs[idx].g.adj) for idx in range(len(test_graphs))]
                #对测试图后门攻击图生成器的注入
                bkd_dr_test, bkd_nid_groups_test, _, _, _, _= generator[id](args, id, test_graphs_trigger, test_backdoor0, Ainput_test, Xinput_test, nodenums_test, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
                for gid in test_backdoor: 
                    for i in bkd_nid_groups_test[gid]:
                        for j in bkd_nid_groups_test[gid]:
                            if i != j:
                                bkd_dr_test[gid].edge_mat[i][j] = 1
                                if (i,j) not in bkd_dr_test[gid].g.edges():
                                    bkd_dr_test[gid].g.add_edge(i, j)
                                                        
                    bkd_dr_test[gid].node_tags = list(dict(bkd_dr_test[gid].g.degree).values())
                args.is_test = 0


                acc_test_clean_no = test_clean_no(args, global_model, device, test_graphs, tag2index) #干净图无划分测试
                test_graphs_division= [args.division_func(graph, args) for graph in tqdm(test_graphs)] #干净图划分
                acc_test_clean = test_clean(args, global_model, device, test_graphs_division, tag2index) #干净图划分后测试

                bkd_dr_ = [bkd_dr_test[idx] for idx in test_backdoor] #提取测试图中的后门图
                print(' #test_back_graphs:', len(bkd_dr_))
                # **添加子图划分**

                acc_test_backdoor_no = test_back_no(args, global_model, device,bkd_dr_, tag2index) #后门图无划分测试

                # 每个原始图分割子图，放一起。不同图之间的子图 分开
                # bkd_dr_ = sum([args.division_func(graph, args) for graph in tqdm(bkd_dr_)], start=[])
                bkd_dr_ = [args.division_func(graph, args) for graph in tqdm(bkd_dr_)] #后门图划分

                subgraph_counts = [len(subgraphs) for subgraphs in bkd_dr_] #依次输出 每个原始图 划分出的 子图数量
                print("每个原始后门图划分后的子图数量:", subgraph_counts)
                print(' #总共对几个原始后门图进行分割:',len(bkd_dr_))

                acc_test_backdoor = test_back(args, global_model, device,bkd_dr_, tag2index) #后门图有划分测试

                # **记录结果（不要关闭文件）**
                log_file.write(f"{epoch}, {loss_avg:.6f}, {acc_test_clean_no:.6f},{acc_test_clean:.6f},{acc_test_backdoor_no:.6f}, {acc_test_backdoor:.6f}\n")
                log_file.flush()  # 确保数据立即写入文件

                f.flush()
            # **在循环外关闭日志文件**
        log_file.close()

            #scheduler.step() 
    # f = open('./saved_model/' + str(args.dataset) + '_triggersize_' + str(args.triggersize), 'wb')

    # pickle.dump(global_model, f)
    # f.close()


if __name__ == '__main__':
    main()