#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from Node_level_Models.helpers.func_utils import accuracy
from node_code.helpers.func_utils import accuracy

from copy import deepcopy
from torch_geometric.nn import SAGEConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None):

        super(GraphSage, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(layer-2):
            self.convs.append(SAGEConv(nhid,nhid))
        self.gc2 = SAGEConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            assert isinstance(edge_index, torch.Tensor), f"edge_index must be a Tensor but got {type(edge_index)}"
            assert edge_index.dtype == torch.long, f"edge_index must be of dtype torch.long but got {edge_index.dtype}"
            assert edge_index.dim() == 2 and edge_index.shape[
                0] == 2, f"edge_index must have shape [2, num_edges] but got {edge_index.shape}"

            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x,dim=1)
    def get_h(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

    def fit(self, global_model,features, edge_index, edge_weight, labels,idx_train, args,idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            loss_train, loss_val, acc_train, acc_val = self._train_with_val(global_model,self.labels, idx_train, idx_val, train_iters, verbose,args)

        return loss_train, loss_val, acc_train, acc_val
    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            print("idx_train",idx_train)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self,global_model, labels, idx_train, idx_val, train_iters, verbose, args):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = -10

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            if args.agg_method == "FedProx":
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(self.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss_train = loss_train + (args.mu / 2) * proximal_term


            loss_train.backward()
            optimizer.step()



            self.eval()
            with torch.no_grad():
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_train = accuracy(output[idx_train], labels[idx_train])
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        print("acc_val",acc_val)
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        return loss_train.item(), loss_val.item(), acc_train, acc_val

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids

    def adjust_bn_layers(self, features, edge_index, edge_weight=None, aug_edge_index=None, aug_edge_weight=None):
        """
        用于 EnergyBelief 聚合策略中，刷新模型中的 BatchNorm 层的统计信息。
        不改变模型参数，只做前向传播。
        """
        self.train()  # 切换到训练模式，以便更新 BN 层的均值和方差

        # 如果有增强后的图结构，就组合原图和增强图进行前向传播
        combined_edge_index = edge_index
        combined_edge_weight = edge_weight

        if aug_edge_index is not None:
            # 拼接边
            combined_edge_index = torch.cat([edge_index, aug_edge_index], dim=1)
            if edge_weight is not None and aug_edge_weight is not None:
                combined_edge_weight = torch.cat([edge_weight, aug_edge_weight])
            else:
                combined_edge_weight = None  # 没边权也允许

        with torch.no_grad():
            _ = self.forward(features, combined_edge_index, combined_edge_weight)

    def forward_energy(self, x, edge_index, edge_weight=None):
        """
        EnergyBelief-style energy function.
        Computes logsumexp over logits as energy (can change to -logsumexp for uncertainty-based).
        Compatible with SAGEConv (does not accept edge_weight).
        """
        x.requires_grad_(True)

        # 安全检查：如果 layer_norm_first 属性不存在，默认设为 False
        if hasattr(self, "layer_norm_first") and self.layer_norm_first:
            x = self.lns[0](x)

        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))  # ✅ 不传 edge_weight（SAGEConv 不支持）
            if hasattr(self, "use_ln") and self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, edge_index)  # ✅ 仍不传 edge_weight
        energy = x.logsumexp(dim=1)  # 可改为 -logsumexp(dim=1) 表示不确定性
        return energy

# %%
