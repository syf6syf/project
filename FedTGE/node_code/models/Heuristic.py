#%%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from node_code.models.GCN import GCN
import networkx as nx

from node_code.helpers.helpers import set_random_seed
# from run_node_exps import args


#%%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight # 每个 idx_attach（要注入 trigger 的目标节点） 生成 trigger_size的的特征和权重

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class Backdoor:

    def __init__(self,args, device):
        set_random_seed(args.seed)
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size,args)
    
    def get_trigger_index(self,trigger_size,args):
        print("Start generating trigger by {}".format(args.trigger_type))
        set_random_seed(args.seed)

        if args.trigger_type == "renyi":
            G_trigger = nx.erdos_renyi_graph(trigger_size, args.density, directed=False) # 每一对节点以概率 p = args.density 连边，生成小型 trigger 结构
            if G_trigger.edges(): # 如果生成的是空图，则强制用密度为 1 的图代替
                G_trigger =  G_trigger
            else:
                G_trigger = nx.erdos_renyi_graph(trigger_size, 1.0, directed=False)


        elif args.trigger_type == "ws": #每个节点先连接 args.degree 个最近邻，然后以概率 density 重连边
            G_trigger = nx.watts_strogatz_graph(trigger_size, args.degree, args.density)
        elif args.trigger_type == "ba": # 倾向于连接度高的节点。
            if args.degree >= trigger_size: # m（degree) 是每个新节点连接的边数，不能大于 trigger_size - 1
                args.degree = trigger_size - 1
            # n: int Number of nodes
            # m: int Number of edges to attach from a new node to existing nodes
            G_trigger = nx.random_graphs.barabasi_albert_graph(n=trigger_size, m=args.degree)
        elif args.trigger_type == "rr": # 所有节点的度都是相同的。需要 trigger_size 是偶数
            # d int The degree of each node.
            # n integer The number of nodes.The value of must be even.
            if args.degree >= trigger_size:
                args.degree = trigger_size - 1
            if trigger_size % 2 != 0:
                trigger_size += 1
            G_trigger = nx.random_graphs.random_regular_graph(d=args.degree,
                                                              n=trigger_size)  # generate a regular graph which has 20 nodes & each node has 3 neghbour nodes.
        # Convert the graph to an edge list in COO format
        edge_list = np.array(list(G_trigger.edges())).T
        # Insert [0, 0] at the beginning of the edge list
        edge_list = np.insert(edge_list, 0, [0, 0], axis=1)
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        return edge_index

    def get_trojan_edge(self,start, idx_attach, trigger_size): #idx_attach是Trigger Posistion选出的注入节点
        set_random_seed(self.args.seed)


        edge_list = []

        for idx in idx_attach: # 为每个攻击目标节点 idx 构造一个独立编号的 trigger 子图，并与该节点相连。
            edges = self.trigger_index.clone()
            edges[0,0] = idx # 原 trigger 的第一条边，现在变成 [idx, start]
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start # 加上偏移量 start，给 trigger 子图重新编号

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col]) #将每条边 (i, j) 变成两条边 (i, j) 和 (j, i)，确保无向图性质

        return edge_index

    def inject_trigger(self, idx_attach, features, edge_index, edge_weight, device):
        set_random_seed(self.args.seed)

        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, _ = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        # trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        # trojan_weights = trojan_weights.flatten()



        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)
        trojan_weights = torch.ones([trojan_edge.shape[1]], device=self.device, dtype=torch.float)

        update_edge_weights = torch.cat([edge_weight,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights
    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):
        features, edge_index, labels, idx_train, idx_attach, idx_unlabeled \
        = features.to(self.device), edge_index.to(self.device), labels.to(self.device), idx_train.to(self.device), idx_attach.to(self.device),idx_unlabeled.to(self.device)

        args = self.args
        set_random_seed(args.seed)

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features

        self.edge_index = edge_index.to(self.device)
        self.edge_weights = edge_weight
        print("nhid",self.args.hidden)
        print("nclass", labels.max().item() + 1)
        print("labels", labels.shape)
        # initial a shadow model 后门攻击专用的小型 GCN；
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=int(labels.max().item() + 1),
                         dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        # get the trojan edges, which include the target-trigger edge and the edges among trigger using heuristic method
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device) # 获取 trigger 的边结构

        trojan_weights = torch.ones([trojan_edge.shape[1]],device=self.device,dtype=torch.float) # trigger 的边数量，给这些边全部分配边权 = 1
        poison_edge_weights = torch.cat(
            [edge_weight, trojan_weights]) # edge_weight = 原图权重 + trigger 权重  # repeat trojan weights beacuse of undirected edge

        # 生成 trigger 特征（类 GAN 的结构）
        # 初始化了 trigger 特征生成器 GraphTrojanNet，作为后门攻击的一部分，用于学习构造让模型“误判”的小图结构。
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size,layernum=2).to(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
        # change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class



        # update the poisoned graph's edge index
        # edge_index = 原图边 + trigger 边
        poison_edge_index = torch.cat([self.edge_index,trojan_edge],dim=1)

        # furture change it to bilevel optimization 未来可能改成双层优化（bilevel optimization）结构。
        
        loss_best = 1e8 #初始化很大的数
        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):

                optimizer_shadow.zero_grad()
                optimizer_trigger.zero_grad()

                # 把目标节点的特征送入 GraphTrojanNet，生成 trigger 节点的特征
                # （args.thrd 是触发器结构生成的边权阈值）
                trojan_feat, _ = self.trojan(features[idx_attach].to(self.device),args.thrd) # may revise the process of generate

                #trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                # trojan_weights = trojan_weights.flatten()

                # 将 trigger 生成器输出的特征 reshape 成标准的特征矩阵格式
                # 相当于[15,3*1433]->[15*3,1433] 变成了 15 个 trigger 节点，每个节点一行，特征维度是 1433
                trojan_feat = trojan_feat.view([-1,features.shape[1]])

                # 将 trigger 特征拼接到原图节点特征后，形成完整的毒化节点特征输入
                poison_x = torch.cat([features.to(self.device),trojan_feat]) #

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)

                # 用 idx_train + idx_attach 是为了保持模型原本的分类能力（对干净数据仍然准确），同时让模型把带 trigger 的节点预测为目标类
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
                loss_inner.backward()
                optimizer_shadow.step()
                optimizer_trigger.step()

        self.trojan.eval()


    def get_poisoned(self):
        set_random_seed(self.args.seed)


        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels


