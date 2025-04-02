import torch
import numpy as np

def gen_input(graphs, bkd_gids, nodemax):

    As = {}
    Xs = {}
    Adj = {}
    for gid in bkd_gids: # 获取图的邻接矩阵和节点特征
        Adj[gid] = graphs[gid].edge_mat   
        
        if gid not in As: As[gid] = Adj[gid].clone()
        if gid not in Xs: Xs[gid] = graphs[gid].node_features.clone() # 原始的节点特征，反映单个节点的初始信息，不包含图结构信息
    Ainputs = {}
    Xinputs = {}
    
    for gid in bkd_gids: # 计算图的输入
        if gid not in Ainputs: Ainputs[gid] = As[gid].clone().detach() # 邻接矩阵
        if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid].float(), Xs[gid]) # Xinputs = Ainputs * Xs.T
        # 初始特征和图结构信息相结合，可以看作是每个节点的特征经过其邻居特征的加权求和。这种机制使得每个节点的表示能够反映其局部图邻域的特性
                
    # pad each input into maxi possible size (N, N) / (N, F)

    for gid in Ainputs.keys(): # 填充输入数据
        a_input = Ainputs[gid]
        x_input = Xinputs[gid]
        
        add_dim = nodemax - a_input.shape[0]
        Ainputs[gid] = np.pad(a_input, ((0, add_dim), (0, add_dim))).tolist() # ((0, add_dim), (0, add_dim)) 表示对矩阵的行和列均在末尾填充 add_dim 个零。最终为（max,max）
        Xinputs[gid] = np.pad(x_input, ((0, add_dim), (0, 0))).tolist() # 列不填充,最终为（max,原列数F）
        Ainputs[gid] = torch.tensor(Ainputs[gid])
        Xinputs[gid] = torch.tensor(Xinputs[gid])

    return Ainputs, Xinputs
    