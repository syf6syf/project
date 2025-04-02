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

criterion = nn.CrossEntropyLoss() #  交叉熵损失函数 用于 分类任务 的损失函数，尤其适用于 多类分类问题。其主要作用是衡量模型预测的类别分布（概率分布）与真实类别之间的差异

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys(): # 每一层key的权重平均
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# 用于训练干净图 （包含优化处理）
def train(args, model, device, train_graphs, optimizer, epoch, tag2index):
    model.train()

    total_iters = args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    #for pos in pbar:
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size] #  从训练数据集中随机选择一个批次的数据索引

        batch_graph = [train_graphs[idx] for idx in selected_idx] # 选择出相应的图数据，形成 一个批次的训练数据
        
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device) # 提取图中标签

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None: # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy() #  转换为 NumPy 格式
        loss_accum += loss

        # report
        #pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    #print("loss training: %f" % (average_loss))

    return average_loss # 经过 total_iters 次训练后，每次迭代的平均损失

# 训练生成器，同时在评估模式下。生成器通过使用后门攻击样本生成带有触发的图数据，然后与目标模型的预测结果计算损失。
def train_G(args, model, generator, id, device, train_graphs_trigger, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    model.eval()#train()
    generator.train() # 将生成器设置为训练模式
    total_iters = 1 #args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    loss_poison_total = 0
    #for pos in pbar:
    #local_feat = torch.zeros(nodemax,nodemax)
    for pos in range(total_iters):
        selected_idx = bkd_gids_train # 后门攻击的图数据索引
        #np.random.permutation(len(train_graphs))[:args.batch_size]
        sub_loss = nn.MSELoss() # 均方误差损失函数
        batch_graph = [train_graphs_trigger[idx] for idx in selected_idx]

        # 生成器生成图数据，生成具有特定属性（例如嵌入了触发的图）的图数据
        output_graph, trigger_group, edges_len, nodes_len, trigger_id, trigger_l = generator(args, id, train_graphs_trigger, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
        output = model(output_graph)
        output_graph_poison = torch.stack([output[idx] for idx in selected_idx]) # 提取所有毒化样本的预测输出 的张量

        labels_poison = torch.LongTensor([args.target for idx in selected_idx]).to(device) # 每个被选中的毒化样本都被分配同样的目标标签 args.target

        #  交叉熵损失用于指导模型学习，将毒化图的预测结果误导到攻击者指定的目标标签上。对毒化效果的直接评估。如果毒化成功，模型会在输入毒化图时产生错误预测
        loss_poison = criterion(output_graph_poison, labels_poison) # nn.CrossEntropyLoss() 交叉熵损失函数 对毒化效果计算损失，参数一预测的标签输出，参数二目标标签

        # 是对触发器本身的优化，而非最终分类结果。通过最小化均方误差损失，可以生成更有效的触发器，使后门攻击更加成功
        loss = sub_loss(trigger_id, trigger_l.detach()) # 中间监督，对生成器的中间结果进行约束，指导生成器生成更符合预期的中间特征，
        # trigger_id 是生成器在生成过程中的一个中间张量，中间监督损失是一种 逐步引导 生成器的方法
    average_loss = 0
    return loss, loss_poison, edges_len, nodes_len


# 训练判别器（Discriminator）
def train_D(args, model, generator, id, device, train_graphs_trigger, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    model.train()
    generator.eval()#train()
    total_iters = args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    loss_poison_total = 0
    #for pos in pbar:
    for pos in range(total_iters):
        selected_idx = bkd_gids_train 

        batch_graph = [train_graphs_trigger[idx] for idx in selected_idx]

        # 生成毒化样本
        output_graph, _, _, _, _, _ = generator(args, id, train_graphs_trigger, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
        # train_graphs = sum([args.division_func(graph, args) for graph in tqdm(train_graphs)], start=[])
        # train_graphs = sum([args.division_func(graph, args) for graph in tqdm(output_graph)], start=[])


        output = model(output_graph)
        labels = torch.LongTensor([graph.label for graph in output_graph]).to(device)

        # compute loss
        loss = criterion(output, labels) 
        loss_accum += loss

    average_loss = loss_accum / total_iters

    return average_loss
def optimize_D(loss, global_model, optimizer_D): #优化判别器
    global_model.zero_grad() 
    optimizer_D.zero_grad()
    loss.backward()
    optimizer_D.step()
    return
def optimize_G(alpha, loss1, loss2, model, optimizer_G):# 优化生成器 确保恶意客户端能够同时达到两个目标
    model.zero_grad()
    optimizer_G.zero_grad()
    loss =  alpha * loss1 + loss2 
    loss.backward()
    optimizer_G.step()
    return
###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=1): # 以小批次方式传递图数据给模型
    model.eval()
    output = [] # 用于存储每个小批次的模型输出
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, test_graphs, tag2index):
    model.eval()

    output = pass_data_iteratively(model, test_graphs) # 所有测试图的预测输出张量
    pred = output.max(1, keepdim=True)[1] # 每个样本的预测类别
    #print("pred:",pred)

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device) # 获取每个测试图的真实标签
    # print(labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item() # 计算正确预测数量
    acc_test = correct / float(len(test_graphs))

    print("accuracy test: %f" % acc_test)

    return acc_test

# 筛选不属于指定标签的数据样本，并返回这些样本的索引列表
def bkd_cdd_test(graphs, target_label):
    
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label: 
            backdoor_graphs_indexes.append(graph_idx)
        
    return backdoor_graphs_indexes
# 选择一部分不属于目标标签的数据样本的索引列表，并可以限制返回的数量
def bkd_cdd(num_backdoor_train_graphs, graphs, target_label, dataset):
    if dataset == 'MUTAG':
        num_backdoor_train_graphs = 1
    
    temp_n = 0
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label and temp_n < num_backdoor_train_graphs:
            backdoor_graphs_indexes.append(graph_idx)
            temp_n += 1
    
    return backdoor_graphs_indexes

# 初始化后门图中的触发器，通过修改图的邻接矩阵、标签和节点特征来实现后门攻击的目标，这些图已经被注入了后门触发器（即节点特征和结构的改变）。
# bkd_gids 指定了哪些图会参与后门攻击;
# bkd_nid_groups 是一个 列表的列表，其中的每个子列表包含了一个图中需要毒化的 节点索引
def init_trigger(args, x, bkd_gids: list, bkd_nid_groups: list, init_feat: float):
    if init_feat == None:
        init_feat = - 1
        print('init feat == None, transferred into -1')

    graphs = copy.deepcopy(x)   
    for idx in bkd_gids: # 遍历所有需要插入后门的图样本
        
        edges = [list(pair) for pair in graphs[idx].g.edges()] # 获取图中所有的边，并将它们转换为列表形式
        edges.extend([[i, j] for j, i in edges]) # 添加边的反向边（即双向边），确保无向图中的边能够被完全表示
        
        for i in bkd_nid_groups[idx]: # 检查并移除它们之间的边
            for j in bkd_nid_groups[idx]: # 从边列表中移除该边
                if [i, j] in edges: # 在 边列表 中移除
                    edges.remove([i, j]) # 循环自然会把双向边删除
                if (i, j) in graphs[idx].g.edges(): # 从 图的结构中 删除该边
                    graphs[idx].g.remove_edge(i, j)
        edge_mat_temp = torch.zeros(len(graphs[idx].g),len(graphs[idx].g)) # 用于表示图的邻接矩阵
        for [x_i,y_i] in edges:
            edge_mat_temp[x_i,y_i] = 1 # 将相应位置设置为 1，表示两个节点之间存在连接。感觉就是 邻接矩阵（包含双向）
        graphs[idx].edge_mat = edge_mat_temp
        # change graph labels
        assert args.target is not None # 确保目标标签存在，这表示攻击后门的目标标签是已定义的。
        graphs[idx].label = args.target # 以便毒化图在训练时能够被模型误分类为目标标签。
        graphs[idx].node_tags = list(dict(graphs[idx].g.degree).values())  # 图中节点的 度 信息
    
        # change features in-place
        featdim = graphs[idx].node_features.shape[1] # 获取节点特征的维度（node_features的列数）
        a = np.array(graphs[idx].node_features) # 节点特征转换为 NumPy 数组
        a[bkd_nid_groups[idx]] = np.ones((len(bkd_nid_groups[idx]), featdim)) * init_feat
        # bkd_nid_groups[idx]在 a 中对应的位置 变成 1*init_feat 。对于后门节点组中的每个节点，将其特征设置为全为 init_feat 的值。
        graphs[idx].node_features = torch.Tensor(a.tolist()) # 修改后的节点特征转换为 PyTorch 张量
            
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
    parser.add_argument('--epochs', type=int, default=200,
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
    args = parser.parse_args()

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

    #######数据集划分#######
    train_data_size = len(train_graphs)
    client_data_size=int(train_data_size/(args.num_agents))
    split_data_size = [client_data_size for i in range(args.num_agents-1)] # 该列表的前 num_agents-1 个元素都为 client_data_size。这意味着除最后一个客户端外，所有客户端都会分配相同数量的数据
    split_data_size.append(train_data_size-client_data_size*(args.num_agents-1))
    train_graphs = torch.utils.data.random_split(train_graphs,split_data_size) #（将数据集划分成多个子集）


    ########全局变量初始化#########
    # 基于图神经网络（GNN）的 判别器模型（Discriminator）,处理图数据并完成图分类或节点分类任务
    global_model = Discriminator(args.num_layers, args.num_mlp_layers, train_graphs[0][0].node_features.shape[1],
                        args.hidden_dim, \
                        num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                        args.neighbor_pooling_type, device).to(device)
    
    optimizer_D = optim.Adam(global_model.parameters(), lr=args.lr) # 创建了一个 Adam 优化器 (optim.Adam) 来优化 global_model 的参数
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1) # 创建学习率调度器，对学习率进行调整，通常用于训练过程中逐步减小学习率，以便模型在接近最优解时能更精细地调整参数。


    ####### 测试阶段 后门图数据准备 ######
    test_graphs_trigger = copy.deepcopy(test_graphs) # 创建一个副本 test_graphs_trigger 来保存经过后门攻击修改的图数据
    test_backdoor = bkd_cdd_test(test_graphs_trigger, args.target)
    #  根据目标标签筛选出后门测试图 （选不属于指定标签的数据样本，并返回这些样本的索引列表）

    #nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
    nodenums = [len(graphs[idx].g.adj) for idx in range(len(graphs))] # 各图节点
    nodemax = max(nodenums) 
    #featdim = np.array(self.benign_dr.data['features'][0]).shape[1] 
    featdim = train_graphs[0][0].node_features.shape[1] # 表示每个节点特征的维度（特征数）



    ######生成器初始化######
    generator = {}
    optimizer_G = {}
    scheduler_G = {}
    for g_i in range(args.num_corrupt): # 为每个生成器创建独立的优化器和学习率调度器，并根据传入的参数初始化它们
        generator[g_i] = OptGDBA.Generator(nodemax, featdim, args.gtn_layernum, args.triggersize)
        optimizer_G[g_i] = optim.Adam(generator[g_i].parameters(), lr=args.lr)
        scheduler_G[g_i] = optim.lr_scheduler.StepLR(optimizer_G[g_i], step_size=50, gamma=0.1) # 创建学习率调度器
    
    # init test data
    # NOTE: for data that can only add perturbation on features, only init the topo value
   
    Ainput_test, Xinput_test = gen_input(test_graphs_trigger, test_backdoor, nodemax) # 据输入图数据生成图神经网络所需的邻接矩阵（Ainputs）和 节点特征矩阵（Xinputs）
    
    with open(args.filenamebd, 'w+') as f:
        f.write("acc_train acc_clean acc_backdoor\n") # 说明接下来写入的是三个指标：acc_train（训练集准确率）、acc_clean（干净数据集的准确率）和 acc_backdoor（后门数据集的准确率）
        bkd_gids_train = {} # ：存储后门图的 ID（即包含后门攻击的训练图的索引）
        Ainput_train = {}
        Xinput_train = {}
        nodenums_id = {}
        train_graphs_trigger = {} # 用于存储 经过后门攻击处理的图数据

        ######生成带有后门攻击的训练数据#########
        for id in range(args.num_corrupt):  # args.num_corrupt：表示生成后门攻击数据的图的数量
            train_graphs_trigger[id] = copy.deepcopy(train_graphs[id])
            nodenums_id[id] = [len(train_graphs_trigger[id][idx].g.adj) for idx in range(len(train_graphs_trigger[id]))] #  计算每个图的节点数
            bkd_gids_train[id] = bkd_cdd(args.num_backdoor_train_graphs, train_graphs_trigger[id], args.target, args.dataset) # 生成后门攻击图的索引
            Ainput_train[id], Xinput_train[id] = gen_input(train_graphs_trigger[id], bkd_gids_train[id], nodemax) # 生成训练图中攻击图的邻接矩阵和节点特征矩阵


        #####训练过程########
        global_weights = global_model.state_dict()  # 获取 global_model 模型的所有参数（即权重和偏置），并将其存储在一个 Python 字典 global_weights 中。

        for epoch in tqdm(range(1, args.epochs + 1)): # 每个训练周期
            local_weights, local_losses = [], []
            m = max(int(args.frac_epoch * args.num_agents), 1) # 计算需要参与的客户端数量
            idxs_users = np.random.choice(range(args.num_agents), m, replace=False) # 随机选择参与训练的客户端
            print("idxs_users:", idxs_users)

            for id in idxs_users: 
                global_model.load_state_dict(copy.deepcopy(global_weights)) # 每次在进行本地训练之前，都加载当前的全局模型权重，把全局模型的参数分配给客户端
                if id < args.num_corrupt: # 说明当前客户端是用于后门攻击的客户端，需要进行特定的训练流程
                    train_graphs_trigger[id] = copy.deepcopy(train_graphs[id]) # 深拷贝当前训练图数据（train_graphs[id]），以防止修改原始图数据。
                    for kk in range(args.n_train_D):
                        loss = train_D(args, global_model, generator[id], id, device, train_graphs_trigger[id], 
                                        epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                        Xinput_train[id], nodenums_id[id], nodemax, 
                                        binaryfeat=False) # 训练判别器（Discriminator）
                        optimize_D(loss, global_model, optimizer_D) # 优化，更新权重
                    if epoch % args.n_epoch == 0: # 每隔 args.n_epoch 训练周期，开始训练生成器 train_G。训练生成器的目的是生成具有后门攻击的图样本，并通过优化使其更好地欺骗模型。
                        for kk in range(args.n_train_G): # 使用后门图和干净图，训练恶意客户端
                            loss, loss_poison, edges_len, nodes_len = train_G(args, global_model, generator[id], id, device, train_graphs_trigger[id], 
                                            epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                            Xinput_train[id], nodenums_id[id], nodemax, 
                                            binaryfeat=False)
                            optimize_G(args.alpha, loss, loss_poison, generator[id], optimizer_G[id])
                else: # 非后门攻击客户端的普通训练
                    loss = train(args, global_model, device, train_graphs[id], optimizer_D, epoch, tag2index) # train中包含反向传播

                l_weights = global_model.state_dict()
                local_weights.append(l_weights)
                local_losses.append(loss)

            scheduler_D.step() #  学习率调度器
            global_weights = average_weights(local_weights)   
            global_model.load_state_dict(global_weights) # 更新全局模型
             
            loss_avg = sum(local_losses) / len(local_losses)    
            
            #----------------- Evaluation -----------------#
            if epoch%5 ==0: # 每经过 5 个训练周期（epoch），执行接下来的测试操作（周期性评估）
                id = 0
                args.is_test = 1 # 表示当前是测试模式
                test_backdoor0 = copy.deepcopy(test_backdoor)
                generator[id].eval() # 将 生成器（Generator） 设置为评估模式
                nodenums_test = [len(test_graphs[idx].g.adj) for idx in range(len(test_graphs))] # 计算测试数据中每个图的节点数
                # 调用 生成器 生成后门攻击图，bkd_dr_test 存储生成的图数据，bkd_nid_groups_test 存储与后门攻击相关的节点分组
                bkd_dr_test, bkd_nid_groups_test, _, _, _, _= generator[id](args, id, test_graphs_trigger, test_backdoor0, Ainput_test, Xinput_test, nodenums_test, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
                for gid in test_backdoor: 
                    for i in bkd_nid_groups_test[gid]: # 遍历所有与后门攻击相关的节点对（i, j）
                        for j in bkd_nid_groups_test[gid]:
                            if i != j:
                                bkd_dr_test[gid].edge_mat[i][j] = 1 # 邻接矩阵，表示这两个节点之间有连接
                                if (i,j) not in bkd_dr_test[gid].g.edges(): # 边不存在于图中
                                    bkd_dr_test[gid].g.add_edge(i, j)
                                                        
                    bkd_dr_test[gid].node_tags = list(dict(bkd_dr_test[gid].g.degree).values()) # 将每个节点的 度数（degree） 作为 节点标签 存储在 node_tags 中
                args.is_test = 0

                # test 就是计算预测标签和真实标签一样的情况如何
                acc_test_clean = test(args, global_model, device, test_graphs, tag2index) # 干净数据上的准确率
                bkd_dr_ = [bkd_dr_test[idx] for idx in test_backdoor] # 提取了带有后门攻击的图数据集，用于后门攻击的测试。
                acc_test_backdoor = test(args, global_model, device, bkd_dr_, tag2index)  # 带有后门攻击的测试集（bkd_dr_）上的准确率

                f.flush() # 将缓冲区中的内容立即写入磁盘。
            #scheduler.step() 
     # f = open('./saved_model/' + str(args.dataset) + '_triggersize_' + str(args.triggersize), 'wb')

    pickle.dump(global_model, f) # 将global_model序列化并保存
    f.close()


if __name__ == '__main__': #  用于确保 代码块只在脚本被直接执行时运行，而在脚本被作为模块导入时不会执行。
    main()