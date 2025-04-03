import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from copy import deepcopy


class GraphGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 graph_pooling='add', device=None, layer_norm_first=True, use_ln=True):
        super(GraphGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        # GCN layers
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)

        # Layer normalization
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))

        # Graph pooling
        if graph_pooling == 'add':
            self.pool = global_add_pool
        elif graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool

        # Final classification layer
        self.graph_classifier = nn.Linear(nhid, nclass)

        # Parameters
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConv):
                nn.init.kaiming_normal_(m.lin.weight)
                if m.lin.bias is not None:
                    nn.init.constant_(m.lin.bias, 0)

    def forward(self, x, edge_index, batch, edge_weight=None):
        # Node feature processing
        if self.layer_norm_first:
            x = self.lns[0](x)

        # First GCN layer
        x = self.gc1(x, edge_index, edge_weight)
        if self.use_ln:
            x = self.lns[1](x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Second GCN layer
        x = self.gc2(x, edge_index, edge_weight)
        if self.use_ln:
            x = self.lns[2](x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Graph pooling
        x = self.pool(x, batch)

        # Classification
        x = self.graph_classifier(x)
        return F.log_softmax(x, dim=1)

    def fit(self, global_model, features, edge_index, batch, edge_weight, labels,
            idx_train, args, idx_val=None, train_iters=200, verbose=False):

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if idx_val is None:
            self._train_without_val(features, edge_index, batch, edge_weight, labels,
                                    idx_train, train_iters, verbose)
        else:
            loss_train, loss_val, acc_train, acc_val = self._train_with_val(
                global_model, features, edge_index, batch, edge_weight, labels,
                idx_train, idx_val, train_iters, verbose, args)
            return loss_train, loss_val, acc_train, acc_val

    def _train_without_val(self, features, edge_index, batch, edge_weight, labels,
                           idx_train, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            output = self.forward(features, edge_index, batch, edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print(f'Epoch {i}, training loss: {loss_train.item()}')

        self.eval()
        output = self.forward(features, edge_index, batch, edge_weight)
        self.output = output

    def _train_with_val(self, global_model, features, edge_index, batch, edge_weight,
                        labels, idx_train, idx_val, train_iters, verbose, args):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = float('inf')
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            output = self.forward(features, edge_index, batch, edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            if args.agg_method == "FedProx":
                proximal_term = 0.0
                for w, w_t in zip(self.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss_train = loss_train + (args.mu / 2) * proximal_term

            loss_train.backward()
            optimizer.step()

            # Validation
            self.eval()
            with torch.no_grad():
                output = self.forward(features, edge_index, batch, edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = self.accuracy(output[idx_val], labels[idx_val])
                acc_train = self.accuracy(output[idx_train], labels[idx_train])

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

        return loss_train.item(), loss_val.item(), acc_train, acc_val

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def test(self, features, edge_index, batch, edge_weight, labels, idx_test):
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, batch, edge_weight)
            acc_test = self.accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)