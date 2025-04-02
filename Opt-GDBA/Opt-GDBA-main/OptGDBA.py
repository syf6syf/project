import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from mask import gen_mask
from input import gen_input
from main import init_trigger

from sklearn.cluster import KMeans

import networkx as nx
import random
import hashlib

#Algorithm 1 𝑘-means_gap to learn customized trigger size
#函数detection1与detection为此公式复现。
#其中detection1()为确定最佳聚类K值,detection()进行最终聚类得到触发器大小。

# 使用 KMeans 聚类 对得分进行聚类，据 gap 值的变化来选取合适的聚类数 k
def detection1(score):
    score = score.numpy() # score 是一个表示图或节点特征的得分数组
    nrefs = 10 # 设置参考聚类的数量。这个参数用于生成参考聚类结果
    ks = range(1, 8)
    if len(score) < 8:
        ks = range(1, len(score)) # 可能的聚类数 k 的范围，要尝试的不同簇数
    gaps = np.zeros(len(ks)) # 存储不同聚类数下的 gap 值 和 gap 值变化
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks)) # 标准差
    min = np.min(score) # min 和 max 用于对得分进行 归一化
    max = np.max(score)
    score = (score - min)/(max-min) # 归一化操作
    for i, k in enumerate(ks): # 对每个可能的 k 值，使用 KMeans 聚类算法对 score 进行聚类
        estimator = KMeans(n_clusters=k) # 初始化 KMeans 聚类器，指定分成k个簇
        estimator.fit(score.reshape(-1, 1)) # fit() 是 KMeans 的训练函数，接受一个数据集（特征矩阵）作为输入，并将数据进行聚类（将 score 转换为列向量输入）。
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))]) # 计算了每个数据点与其所属簇中心的 平方距离，并将这些平方距离加总，得到一个 簇内离散度

        WkRef = np.zeros(nrefs)
        for j in range(nrefs): #对于每个聚类数 k，生成 nrefs 个参考聚类结果，这些结果通过在 [0, 1] 区间生成随机数据来模拟与真实数据不同的聚类效果。
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))]) # 对参考聚类进行相同的聚类操作，计算参考聚类的离散度 WkRef
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)  # 计算每个 k 对应的 gap 值，gap 值越大，表示聚类效果越好。
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef)) #  gap 值的标准差，用来量化 gap 值的不确定性

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i] # 用于衡量当前 k 和前一个 k 之间的 gap 变化，如果变化较大，说明聚类数 k 可能更合适。
    #print(gapDiff)
    select_k = 3
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0: # 值是差，值为正表示后者更大
            select_k = i+1 # 说明当前 k 的聚类效果更好，选择当前 k 为最佳聚类数
            break
    return select_k

#返回最终触发器大小
#对 score 数据进行聚类。
#找出聚类结果中平均值最大的簇，并返回该簇中数据点的数量。
def detection(score, k_value): # 根据给定的 k_value，使用 KMeans 聚类算法对得分进行聚类，输出最大簇的节点数
    score = score.numpy()
    estimator = KMeans(n_clusters=k_value)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    trigger_size = {}
    temp_max = 0
    temp_size = 0
    for i in range(k_value): # 计算每个簇的平均得分
        trigger_size[i] = np.mean(score[label_pred==i])

    for i in range(k_value): # 找出最大平均得分对应的簇的节点数量
        if trigger_size[i] > temp_max:
            temp_max = trigger_size[i]
            temp_size = len(label_pred==i)
    return  int(temp_size)       
    
#选择合适的节点作为触发器的位置，触发器的位置决定了哪些节点将被包含在触发器中
def trigger_top(rank_value, rank_id, trigger_size, number_id): # 根据排序值选择 trigger_size 大小的节点，用于后续的后门攻击
    local_id = []
    if number_id < trigger_size:
        trigger_size = number_id
    for i in range(int(trigger_size)):
        local_id.append(rank_id[i,0].tolist())
    return local_id

def trigger_top_c(rank_value, rank_id): # trigger_top 的改进版本，结合了 detection1 和 detection 方法来自动确定 trigger_size 的大小
    k = detection1(rank_value) # 利用Gap和Kmeans确定选择聚类数量
    if k == 1:
        trigger_size = 3 # 选择 3 个节点作为触发器
    else:
        trigger_size = detection(rank_value, k) # 获取一个动态的 trigger_size。detection 方法会根据聚类的结果确定选择多少个节点作为触发器
        if trigger_size > 5:
            trigger_size = 5
        elif trigger_size < 3:
            trigger_size = 3
    
    local_id = []
    for i in range(trigger_size):
        local_id.append(rank_id[i,0].tolist())
    return local_id

class GradWhere(torch.autograd.Function): # 二值化处理，用于在图神经网络的训练过程中应用特定的梯度操作。（
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
        rst = torch.where(input>=thrd, torch.tensor(1.0, device=device, requires_grad=True),   #  实现了基于阈值的条件操作，它基于条件判断输入值是否大于某个阈值 thrd，如果是，则输出 1，否则输出 0。
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output): #  GradWhere 中的反向传播仅仅是传递从上一层接收到的梯度（grad_output），并根据前向传播的条件决定梯度的值。
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


'''
bkd_gids_train 是一个包含图 ID 的列表，表示在训练过程中被选择进行后门攻击的图。每个图通过 bkd_gids_train 标识，恶意客户端将在这些图中注入后门触发器
bkd_nid_groups 是一个字典，键为图 ID，值为一个包含节点 ID 的列表，表示在对应图中需要进行后门攻击的节点组。每个图中有一组特定的节点被选中作为后门注入的目标节点。
bkd_gids 也是一个包含图 ID 的列表，表示需要进行后门攻击的图的集合。与 bkd_gids_train 相似，但其作用范围通常更广，可能涉及多个阶段或不同的数据集
'''
class Generator(nn.Module): #  生成后门攻击图
    def __init__(self, sq_dim, feat_dim, layernum, trigger_size, dropout=0.05):
        super(Generator, self).__init__()
        layers = []
        layers_feat = []
        view = []
        view_feat = []
        if dropout > 0: # 定义了处理 结构数据（例如图的邻接矩阵等） 的神经网络层
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(sq_dim, sq_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sq_dim, sq_dim))
        
        if dropout > 0: # 处理 节点特征数据 的网络层
            layers_feat.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers_feat.append(nn.Linear(feat_dim, feat_dim))
            layers_feat.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers_feat.append(nn.Dropout(p=dropout))
        layers_feat.append(nn.Linear(feat_dim, feat_dim))

        if dropout > 0: # 用于处理 图结构数据 的网络层，与 layers 部分类似，但它用于图的表示学习。
            view.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            view.append(nn.Linear(sq_dim, sq_dim))
            view.append(nn.ReLU(inplace=True))
            if dropout > 0:
                view.append(nn.Dropout(p=dropout))
        view.append(nn.Linear(sq_dim, sq_dim))

        if dropout > 0: # 用于处理 特征数据的表示学习，与 layers_feat 部分类似，用于学习特征数据的复杂模式。
            view_feat.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            view_feat.append(nn.Linear(feat_dim, feat_dim))
            view_feat.append(nn.ReLU(inplace=True))
            if dropout > 0:
                view_feat.append(nn.Dropout(p=dropout))
        view_feat.append(nn.Linear(feat_dim, feat_dim))
        
        self.sq_dim = sq_dim
        self.feat_dim = feat_dim
        self.trigger_size = trigger_size
        self.layers = nn.Sequential(*layers)
        self.layers_feat = nn.Sequential(*layers_feat)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_feat = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Linear(1, sq_dim*sq_dim)
        self.mlp_feat = nn.Linear(1, sq_dim*feat_dim)
        self.view = nn.Sequential(*view)
        self.view_feat = nn.Sequential(*view_feat)
        #self.mlp_pool = nn.AdaptiveAvgPool1d(1)
               
    def forward(self, args, id, graphs_train, bkd_gids_train, Ainput, Xinput, nodenums_id, 
                nodemax, is_Customized , is_test , trigger_size , device=torch.device('cpu'), binaryfeat=False):
        # Ainput:图的拓扑特征矩阵（邻接矩阵）。Xinput：图的节点特征矩阵。
        # bkd_gids_train：需要进行后门处理的训练图的 ID 集合。

        bkd_nid_groups = {} # 存储每个图中\需要注入后门攻击的节点。
        GW = GradWhere.apply # 自定义的梯度操作函数，应用于后门攻击图的训练过程中。GradWhere 类会在图的梯度传播中使用。

        graphs = copy.deepcopy(graphs_train)
        nodes_len = 0
        for gid in bkd_gids_train:#tqdm(bkd_gids_train):
            rst_bkdA_backbone = self.view(Ainput[gid]) # 处理图邻接矩阵，从 Ainput（邻接矩阵）中获取图的结构数据，并传入网络中的 view 层
            if args.topo_activation=='relu':
                rst_bkdA_backbone = F.relu(rst_bkdA_backbone) # 设置的激活函数类型（如 ReLU 或 Sigmoid）对输出进行激活。
            elif args.topo_activation=='sigmoid':
                rst_bkdA_backbone = torch.sigmoid(rst_bkdA_backbone)    # nn.Functional.sigmoid is deprecated 28*28
            rst_bkdA_backbone = self.avg_pool(rst_bkdA_backbone)   # avg_pool 层进行池化操作。池化操作有助于降维，通常用于在神经网络中减少信息量，使得特征表示更加紧凑。 28*1
            
            rst_bkdX_backbone = self.view_feat(Xinput[gid]) # 处理节点特征矩阵，对每个训练图中的节点特征 Xinput 进行处理，使用 view_feat 层（全连接层）来处理节点特征。
            if args.feat_activation=='relu':
                rst_bkdX_backbone = F.relu(rst_bkdX_backbone)
            elif args.feat_activation=='sigmoid':
                rst_bkdX_backbone = torch.sigmoid(rst_bkdX_backbone)     # 28*5
            rst_bkdX_backbone = self.avg_pool_feat(rst_bkdX_backbone) # 28*1

            #########节点重要性得分学习##########
            trigger_id = torch.mul(rst_bkdA_backbone[:nodenums_id[gid]], # 节点重要性得分，通过元素级的乘法操作(应该就是注意力机制），可以确定哪些节点在图中最为关键，进而决定哪些节点将作为触发器的候选节点。
                                 rst_bkdX_backbone[:nodenums_id[gid]])

            trigger_l = GW(trigger_id, torch.mean(trigger_id), device) # 二值化处理，使用 GW 来对 trigger_id 进行处理，torch.mean(trigger_id) 计算 trigger_id 的平均值作为参考。其中仅保留高于或等于均值的节点评分。
            rank_value, rank_id = torch.sort(trigger_id, dim=0, descending=True) # 对 trigger_id 进行排序，得到节点的排名 rank_value 和对应的节点ID rank_id。这一步是为了确定哪些节点最适合被注入后门。

            #########触发器位置（大小）学习##########
            bkd_nid_groups[gid] = trigger_top(rank_value, rank_id, self.trigger_size,nodenums_id[gid])  # 预定义 选取的后门攻击节点，根据 rank_value 和 rank_id，选择 前trigger_size 数量的节点作为后门攻击的目标节点。trigger_top 选取排名前 trigger_size 的节点。


       #######触发器形状学习#########
        init_dr = init_trigger(
                        args, graphs, bkd_gids_train, bkd_nid_groups, 0.0) # 把 bkd_nid_groups[gid] 的节点之间边删除，再获取邻接矩阵，更改标签，获取度矩阵添加到图中属性。初始化生成带有后门攻击的图数据 ，通过 init_trigger 函数初始化后门图，并会在图中插入后门触发器（trigger）
        bkd_dr = copy.deepcopy(init_dr)
        topomask, featmask = gen_mask(
                        graphs[0].node_features.shape[1], nodemax, bkd_dr, bkd_gids_train, bkd_nid_groups)
        # 生成与后门攻击相关的拓扑掩码（topomask与邻接矩阵相关）和特征掩码（featmask与节点特征相关）。
        # topomask 用于控制图中哪些边在后门攻击过程中应该被修改，featmask 用于控制图中哪些节点特征在后门攻击过程中应该被修改

        Ainput_trigger, Xinput_trigger = gen_input(init_dr, bkd_gids_train, nodemax) #生成后门图的输入数据（邻接矩阵 Ainput_trigger 和节点特征矩阵 Xinput_trigger），用于训练后门攻击模型。

        id = torch.as_tensor(float(id)).unsqueeze(0)
        id_output = self.mlp(id) # 通过一个全连接层（MLP）生成一个 ID 输出 id_output，它将后门攻击的标识符 id 映射到一个高维空间。
        id_output = id_output.reshape(self.sq_dim,self.sq_dim)

        id_output_feat = self.mlp_feat(id) # 通过另一个全连接层（mlp_feat）处理 id，生成与特征相关的输出 id_output_feat，用于后门图的特征操作。
        id_output_feat = id_output_feat.reshape(self.sq_dim,self.feat_dim)


        for gid in bkd_gids_train:
            Ainput_trigger[gid] = Ainput_trigger[gid] * id_output # 客户端的嵌入信息与矩阵结合，Ainput_trigger[gid] 是图的邻接矩阵。通过与 id_output 相乘，邻接矩阵被加权。通过引入 id_output 来调整图结构的拓扑关系
            # 根据图的邻接矩阵（Ai_B）计算触发器的边注意力矩阵（Ei_tri）。这个过程用于决定哪些边应该属于触发器。
            rst_bkdA = self.layers(Ainput_trigger[gid]) # 代表了图的结构信息的变化，即图的拓扑结构经过网络的变换
            if args.topo_activation=='relu':
                rst_bkdA = F.relu(rst_bkdA)
            elif args.topo_activation=='sigmoid': #用的这个
                rst_bkdA = torch.sigmoid(rst_bkdA)    # nn.Functional.sigmoid is deprecated

            for_whom='topo'
            if for_whom == 'topo':  
                rst_bkdA = torch.div(torch.add(rst_bkdA, rst_bkdA.transpose(0, 1)), 2.0)
                # 将 rst_bkdA 与其转置相加，然后除以 2。使得图的邻接矩阵成为对称矩阵（即无向图），确保每一对节点之间的关系是双向的
            if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
                rst_bkdA = GW(rst_bkdA, args.topo_thrd, device) #   比0.5大的设置为1；进行 梯度 处理，可能是为了在训练过程中控制拓扑结构的更新。
            rst_bkdA = torch.mul(rst_bkdA, topomask[gid])
            # 进一步限制 rst_bkdA 中的某些元素。topomask 用于控制哪些节点的边应该被修改，哪些不应该。掩码的作用是对邻接矩阵应用一个遮罩，只保留指定节点之间的连接。

            bkd_dr[gid].edge_mat = torch.add(init_dr[gid].edge_mat, rst_bkdA[:nodenums_id[gid], :nodenums_id[gid]]) # 更新后门攻击图的邻接矩阵
            for i in range(nodenums_id[gid]): # 二值化处理：为了确定节点之间的连接状态，边的注意力矩阵被转换为二值矩阵.添加边到图中
                for j in range(nodenums_id[gid]):
                    if rst_bkdA[i][j] == 1 and i < j:
                        bkd_dr[gid].g.add_edge(i, j)
            bkd_dr[gid].node_tags = list(dict(bkd_dr[gid].g.degree).values()) #  更新节点标签，通过计算图中每个节点的度（即每个节点的连接数）并将其作为节点标签存储在 node_tags 中。
         
            for_whom='feat'
            Xinput_trigger[gid] = Xinput_trigger[gid]*id_output_feat # 客户端的嵌入信息与注意力矩阵结合，处理节点特征
            rst_bkdX = self.layers_feat(Xinput_trigger[gid]) # 通过全连接层处理节点特征
            if args.feat_activation=='relu': # 用的这个
                rst_bkdX = F.relu(rst_bkdX)
            elif args.feat_activation=='sigmoid':
                rst_bkdX = torch.sigmoid(rst_bkdX)
                
            if for_whom == 'topo': # not consider direct yet
                rst_bkdX = torch.div(torch.add(rst_bkdX, rst_bkdX.transpose(0, 1)), 2.0)
            # binaryfeat = True
            if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
                rst_bkdX = GW(rst_bkdX, args.feat_thrd, device)
            rst_bkdX = torch.mul(rst_bkdX, featmask[gid])
            
            bkd_dr[gid].node_features = torch.add( 
                    rst_bkdX[:nodenums_id[gid]].detach().cpu(), torch.Tensor(init_dr[gid].node_features)) # 更新节点特征
            
        edges_len_avg = 0
        return bkd_dr, bkd_nid_groups, edges_len_avg, self.trigger_size, trigger_id, trigger_l
        # 回处理后的后门图数据 bkd_dr、节点组 bkd_nid_groups、平均边长 edges_len_avg 以及其他信息（如触发器 ID 和触发器标签 trigger_id 和 trigger_l）
    
def SendtoCUDA(gid, items):
    """
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    cuda = torch.device('cuda')
    for item in items:
        item[gid] = torch.as_tensor(item[gid], dtype=torch.float32).to(cuda)
        
        
def SendtoCPU(gid, items):
    """
    Used after SendtoCUDA, target object must be torch.tensor and already in cuda.
    
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    
    cpu = torch.device('cpu')
    for item in items:
        item[gid] = item[gid].to(cpu)