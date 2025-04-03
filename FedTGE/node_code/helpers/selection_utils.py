from random import random
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from node_code.models.construct import model_construct

import os
import time
from sklearn_extra import cluster
from sklearn.cluster import KMeans
import networkx as nx
import dgl

from node_code.helpers.helpers import set_random_seed


def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def obtain_attach_nodes(args, node_idxs, size):
    ### current random to implement

    size = min(len(node_idxs), size)
    rs = np.random.RandomState(args.seed)
    choice = np.arange(len(node_idxs))
    rs.shuffle(choice) # 打乱顺序
    return node_idxs[choice[:size]]

def sort_subset(sorted_nodes, node_idxs):
    node_idx_dict = {node_idx: i for i, node_idx in enumerate(sorted_nodes)}
    sorted_node_idxs = sorted(node_idxs.numpy(), key=lambda x: node_idx_dict[x])
    return sorted_node_idxs
def obtain_attach_nodes_degree(args, node_idxs, client_data,size):
    # Calculate the degree of each node
    deg = torch.zeros(client_data.num_nodes, dtype=torch.long)
    deg.index_add_(0, client_data.edge_index[0], torch.ones_like(client_data.edge_index[1]))
    deg.index_add_(0, client_data.edge_index[1], torch.ones_like(client_data.edge_index[0]))

    # Create a dictionary where keys are node indices and values are degrees
    degree_dict = dict(enumerate(deg.tolist()))

    # Sort the nodes by their degree in descending order
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

    # # Create a list of indices of nodes in node_idxs that correspond to their positions in the sorted_nodes list
    # node_idxs_sorted_indices = [sorted_nodes.index(node_idx) for node_idx in node_idxs]
    #
    # # Sort the node_idxs list based on the indices obtained above
    # sorted_node_idxs = [node_idxs[index] for index in
    #                     sorted(range(len(node_idxs_sorted_indices)), key=node_idxs_sorted_indices.__getitem__)]

    sorted_node_idxs = sort_subset(sorted_nodes, node_idxs)

    return sorted_node_idxs[:size]

def obtain_attach_nodes_cluster(args, node_idxs, client_data, size):

    # create a DGL graph from edge_index
    g = dgl.graph((client_data.edge_index[0], client_data.edge_index[1]), num_nodes=client_data.num_nodes)

    # convert DGL graph to NetworkX graph
    g = dgl.to_networkx(g.cpu())
    #  sort according to cluster
    simple_g = nx.Graph(g)
    clustering_dict = nx.clustering(simple_g, weight='weight')
    sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

    # # Create a list of indices of nodes in node_idxs that correspond to their positions in the sorted_nodes list
    # node_idxs_sorted_indices = [sorted_nodes.index(node_idx) for node_idx in node_idxs]
    #
    # # Sort the node_idxs list based on the indices obtained above
    # sorted_node_idxs = [node_idxs[index] for index in
    #                     sorted(range(len(node_idxs_sorted_indices)), key=node_idxs_sorted_indices.__getitem__)]
    sorted_node_idxs = sort_subset(sorted_nodes, node_idxs)

    return sorted_node_idxs[:size]


def obtain_attach_nodes_by_influential(args, model, node_idxs, x, edge_index, edge_weights, labels, device, size,
                                       selected_way='conf'):
    size = min(len(node_idxs), size)
    # return node_idxs[np.random.choice(len(node_idxs),size,replace=False)]
    loss_fn = F.nll_loss
    model = model.to(device)
    labels = labels.to(device)
    model.eval()
    output = model(x, edge_index, edge_weights)
    loss_diffs = []
    '''select based on the diff between the loss on target class and true class, nodes with larger diffs are easily selected '''
    if (selected_way == 'loss'):
        candidate_nodes = np.array([])

        for id in range(output.shape[0]):
            loss_atk = loss_fn(output[id], torch.LongTensor([args.target_class]).to(device)[0])
            loss_bef = loss_fn(output[id], labels[id])
            # print(loss_atk,loss_bef)
            loss_diff = float(loss_atk - loss_bef)
            loss_diffs.append(loss_diff)
        loss_diffs = np.array(loss_diffs)

        # split the nodes according to the label
        label_list = np.unique(labels.cpu())
        labels_dict = {}
        for i in label_list:
            labels_dict[i] = np.where(labels.cpu() == i)[0]
            # filter out labeled nodes
            labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))
        # fairly select from all the class except for the target class
        each_selected_num = int(size / len(label_list) - 1)
        last_seleced_num = size - each_selected_num * (len(label_list) - 2)
        for label in label_list:
            single_labels_nodes = labels_dict[label]  # the node idx of the nodes in single class
            single_labels_nodes = np.array(list(set(single_labels_nodes)))
            single_labels_nodes_loss = loss_diffs[single_labels_nodes]
            single_labels_nid_index = np.argsort(-single_labels_nodes_loss)  # sort descently based on the loss
            sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
            if (label != label_list[-1]):
                candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:each_selected_num]])
            else:
                candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:last_seleced_num]])
        return candidate_nodes.astype(int)
    elif (selected_way == 'conf'):
        '''select based on the diff between the conf on target class and true class, nodes with larger confidents are easily selected '''
        candidate_nodes = np.array([])
        confidences = []
        # calculate the confident of each node
        output = model(x, edge_index, edge_weights)
        softmax = torch.nn.Softmax(dim=1)
        for i in range(output.shape[0]):
            output_nids = output[[i]]
            preds = output_nids.max(1)[1].type_as(labels)
            preds = preds.cpu()
            correct = preds.eq(labels[[i]].detach().cpu()).double().sum().item()
            confidence = torch.mean(torch.max(softmax(output_nids), dim=1)[0]).item()
            confidences.append(confidence)
        confidences = np.array(confidences)
        # split the nodes according to the label
        label_list = np.unique(labels.cpu())
        labels_dict = {}
        for i in label_list:
            labels_dict[i] = np.where(labels.cpu() == i)[0]
            labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))
        # fairly select from all the class except for the target class
        each_selected_num = int(size / len(label_list) - 1)
        last_seleced_num = size - each_selected_num * (len(label_list) - 2)
        for label in label_list:
            single_labels_nodes = labels_dict[label]
            single_labels_nodes = np.array(list(set(single_labels_nodes)))
            single_labels_nodes_conf = confidences[single_labels_nodes]
            single_labels_nid_index = np.argsort(-single_labels_nodes_conf)
            sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
            if (label != label_list[-1]):
                candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:each_selected_num]])
            else:
                candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:last_seleced_num]])
        return candidate_nodes.astype(int)


# 从非目标类的未标注节点中，选择那些“远离自己簇中心、接近目标类中心”的节点作为后门注入目标，提升攻击的迷惑性与成功率。
def obtain_attach_nodes_by_cluster(args, y_pred, model, node_idxs, x, labels, device, size):
    dis_weight = args.dis_weight
    cluster_centers = model.cluster_centers_
    # y_pred = model.predict(x.detach().cpu().numpy())
    # y_true = labels.cpu().numpy()
    # calculate the distance of each nodes away from their centers
    distances = []
    distances_tar = []
    for id in range(x.shape[0]):
        # tmp_center_label = args.target_class
        tmp_center_label = y_pred[id]
        # tmp_true_label = y_true[id]
        tmp_tar_label = args.target_class

        tmp_center_x = cluster_centers[tmp_center_label]
        # tmp_true_x = cluster_centers[tmp_true_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        # dis1 = np.linalg.norm(tmp_true_x - x[id].cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[id].cpu().numpy())
        # print(dis,dis1,tmp_center_label,tmp_true_label)
        distances.append(dis)
        distances_tar.append(dis_tar)

    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    # label_list = np.unique(labels.cpu())
    print(y_pred)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        # labels_dict[i] = np.where(labels.cpu()==i)[0]
        labels_dict[i] = np.where(y_pred == i)[0]
        # filter out labeled nodes
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    each_selected_num = int(size / len(label_list) - 1)
    last_seleced_num = size - each_selected_num * (len(label_list) - 2)
    candidate_nodes = np.array([])
    for label in label_list:
        if (label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]  # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes)))
        print("single_labels_nodes",single_labels_nodes)
        single_labels_nodes_dis = distances[int(single_labels_nodes)]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)
        single_labels_nodes_dis_tar = distances_tar[single_labels_nodes]
        single_labels_nodes_dis_tar = max_norm(single_labels_nodes_dis_tar)
        # the closer to the center, the more far away from the target centers

        # distance_to_own_center（越大越边缘），distance_to_target_center（越小越接近目标类）
        # 距离自己所在聚类簇中心越远（非典型、边缘），距离目标类别簇中心越近（更容易被误分类为目标类）
        single_labels_dis_score = dis_weight * single_labels_nodes_dis + (-single_labels_nodes_dis_tar)
        single_labels_nid_index = np.argsort(
            single_labels_dis_score)  # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if (label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes


def obtain_attach_nodes_by_cluster_gpu(args, y_pred, cluster_centers, node_idxs, x, labels, device, size):
    dis_weight = args.dis_weight
    # cluster_centers = model.cluster_centers_
    # y_pred = model.predict(x.detach().cpu().numpy())
    # y_true = labels.cpu().numpy()
    # calculate the distance of each nodes away from their centers
    distances = []
    distances_tar = []
    for id in range(x.shape[0]):
        # tmp_center_label = args.target_class
        tmp_center_label = y_pred[id]
        # tmp_true_label = y_true[id]
        tmp_tar_label = args.target_class

        tmp_center_x = cluster_centers[tmp_center_label]
        # tmp_true_x = cluster_centers[tmp_true_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        # dis1 = np.linalg.norm(tmp_true_x - x[id].cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[id].cpu().numpy())
        # print(dis,dis1,tmp_center_label,tmp_true_label)
        distances.append(dis)
        distances_tar.append(dis_tar)

    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    # label_list = np.unique(labels.cpu())
    print(y_pred)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        # labels_dict[i] = np.where(labels.cpu()==i)[0]
        labels_dict[i] = np.where(y_pred == i)[0]
        # filter out labeled nodes
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    each_selected_num = int(size / len(label_list) - 1)
    last_seleced_num = size - each_selected_num * (len(label_list) - 2)
    candidate_nodes = np.array([])
    for label in label_list:
        if (label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]  # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes)))

        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)
        single_labels_nodes_dis_tar = distances_tar[single_labels_nodes]
        single_labels_nodes_dis_tar = max_norm(single_labels_nodes_dis_tar)
        # the closer to the center, the more far away from the target centers
        single_labels_dis_score = single_labels_nodes_dis + dis_weight * (-single_labels_nodes_dis_tar)
        single_labels_nid_index = np.argsort(
            single_labels_dis_score)  # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if (label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes


from torch_geometric.utils import degree


def obtain_attach_nodes_by_cluster_degree(args, edge_index, y_pred, cluster_centers, node_idxs, x, size):
    dis_weight = args.dis_weight
    # cluster_centers = model.cluster_centers_
    # y_pred = model.predict(x.detach().cpu().numpy())
    # y_true = labels.cpu().numpy()
    # calculate the distance of each nodes away from their centers
    degrees = (degree(edge_index[0]) + degree(edge_index[1])).cpu().numpy()
    distances = []
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)
    print(y_pred)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        labels_dict[i] = np.where(y_pred == i)[0]
        # filter out labeled nodes
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    each_selected_num = int(size / len(label_list) - 1)
    last_seleced_num = size - each_selected_num * (len(label_list) - 2)
    candidate_nodes = np.array([])
    for label in label_list:
        if (label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]  # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes)))
        print(distances, single_labels_nodes)
        if (len(single_labels_nodes) == 0):
            continue
        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)

        single_labels_nodes_degrees = degrees[single_labels_nodes]
        single_labels_nodes_degrees = max_norm(single_labels_nodes_degrees)

        # the closer to the center, the more far away from the target centers
        # single_labels_dis_score =  single_labels_nodes_dis + dis_weight * (-single_labels_nodes_dis_tar)
        single_labels_dis_score = single_labels_nodes_dis + dis_weight * single_labels_nodes_degrees
        single_labels_nid_index = np.argsort(
            single_labels_dis_score)  # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if (label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:each_selected_num]])
        else:
            last_seleced_num = size - len(candidate_nodes)
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes


# 在所有未标注、非目标类的节点中，计算“离簇中心的距离远 + 节点度高”的加权分数，选择得分最高的前 size 个节点用于注入。
def obtain_attach_nodes_by_cluster_degree_all(args, edge_index, y_pred, cluster_centers, node_idxs, x, size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0]) + degree(edge_index[1])).cpu().numpy()

    deg = torch.zeros(len(y_pred), dtype=torch.long)
    deg.index_add_(0, edge_index[0], torch.ones_like(edge_index[1]))
    deg.index_add_(0, edge_index[1], torch.ones_like(edge_index[0]))
    degrees = deg
    print("degree", degrees)
    print("degree",degrees.shape)
    print("y_pred",y_pred.shape)
    distances = []
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)

    # label_list = np.unique(labels.cpu())
    # label_list = np.unique(y_pred)
    # labels_dict = {}
    # for i in label_list:
    #     # labels_dict[i] = np.where(labels.cpu()==i)[0]
    #     labels_dict[i] = np.where(y_pred==i)[0]
    #     # filter out labeled nodes
    #     labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))
    nontarget_nodes = np.where(y_pred != args.target_class)[0]

    non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
    node_idxs = np.array(non_target_node_idxs)
    candiadate_distances = distances[node_idxs]
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    # print(candidate_nid_index,node_idxs)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    return selected_nodes
    each_selected_num = int(size / len(label_list) - 1)
    last_seleced_num = size - each_selected_num * (len(label_list) - 2)
    candidate_nodes = np.array([])

    for label in label_list:
        if (label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]  # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes)))

        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)

        single_labels_nodes_degrees = degrees[single_labels_nodes]
        single_labels_nodes_degrees = max_norm(single_labels_nodes_degrees)

        # the closer to the center, the more far away from the target centers
        # single_labels_dis_score =  single_labels_nodes_dis + dis_weight * (-single_labels_nodes_dis_tar)

        # 优先选择那些在聚类中越边缘、连接度越高的节点，兼顾了“攻击成功率”和“传播性”。
        single_labels_dis_score = single_labels_nodes_dis + dis_weight * single_labels_nodes_degrees
        single_labels_nid_index = np.argsort(
            single_labels_dis_score)  # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if (label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes




# from kmeans_pytorch import kmeans, kmeans_predict

# 使用 GCN encoder 学习节点表示后，通过 K-Medoids 聚类，并选出偏离聚类中心的未标注节点作为后门注入目标，使得后门更难被检测，攻击效果更强。
def cluster_distance_selection(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx, train_edge_index, size,
                               device):
    encoder_modelpath = './modelpath/{}_{}_benign.pth'.format('GCN_Encoder', args.dataset)
    if (os.path.exists(encoder_modelpath)):
        # load existing benign model
        gcn_encoder = torch.load(encoder_modelpath)
        gcn_encoder = gcn_encoder.to(device)
        edge_weights = torch.ones([data.edge_index.shape[1]], device=device, dtype=torch.float)
        print("Loading {} encoder Finished!".format(args.model))
    else:
        gcn_encoder = model_construct(args, 'GCN_Encoder', data, device).to(device)
        t_total = time.time()
        # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Length of training set: {}".format(len(idx_train)))
        gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs,
                        verbose=True)
        print("Training encoder Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # # Save trained model
        # torch.save(gcn_encoder, encoder_modelpath)
        # print("Encoder saved at {}".format(encoder_modelpath))
    # test gcn encoder
    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y, idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train.to(device), unlabeled_idx.to(device)])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x.to(device), train_edge_index.to(device), None).clone().detach()
    encoder_output = gcn_encoder(data.x.to(device), train_edge_index.to(device), None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    gcn_encoder = gcn_encoder.cpu()
    kmedoids = cluster.KMedoids(n_clusters=nclass, method='pam')
    kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    idx_attach = obtain_attach_nodes_by_cluster(args, y_pred, kmedoids, unlabeled_idx.cpu().tolist(), encoder_x, data.y,
                                                device, size).astype(int)
    return idx_attach


def cluster_degree_selection(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx, train_edge_index, size,
                             device):
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset, args.seed)
    if (os.path.exists(selected_nodes_path)):
        print(selected_nodes_path)
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
        idx_attach = idx_attach[:size]
        return idx_attach
    gcn_encoder = model_construct(args, 'GCN_Encoder', data, device).to(device)
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y, idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    idx_train, unlabeled_idx = idx_train.to(device), unlabeled_idx.to(device)
    seen_node_idx = torch.concat([idx_train, unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x.to(device), train_edge_index.to(device), None).clone().detach()
    if (args.dataset == 'Cora' or args.dataset == 'Citeseer'):
        kmedoids = cluster.KMedoids(n_clusters=nclass, method='pam')
        kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmedoids.cluster_centers_
        y_pred = kmedoids.predict(encoder_x.cpu().numpy())
    else:
        # _, cluster_centers = kmeans(X=encoder_x[seen_node_idx], num_clusters=nclass, distance='euclidean', device=device)
        kmeans = KMeans(n_clusters=nclass, random_state=1)
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())
    # kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    # kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    # cluster_centers = kmedoids.cluster_centers_

    encoder_output = gcn_encoder(data.x.to(device), train_edge_index.to(device), None)
    # y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    # cluster_centers = []
    # for label in range(nclass):
    #     idx_sing_class = (y_pred == label).nonzero()[0]
    #     print(encoder_x[idx_sing_class])
    #     # print((y_pred == label).nonzero()[0])
    #     print(idx_sing_class)
    #     _, sing_center = kmeans(X=encoder_x[idx_sing_class], num_clusters=1, distance='euclidean', device=device)
    #     cluster_centers.append(sing_center)

    # idx_attach = obtain_attach_nodes_by_cluster_degree(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
    idx_attach = obtain_attach_nodes_by_cluster_degree_all(args, train_edge_index, y_pred, cluster_centers,
                                                           unlabeled_idx.cpu().tolist(), encoder_x, size).astype(int)
    selected_nodes_foldpath = "./selected_nodes/{}/Overall/seed{}".format(args.dataset, args.seed)
    if (not os.path.exists(selected_nodes_foldpath)):
        os.makedirs(selected_nodes_foldpath)
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset, args.seed)
    if (not os.path.exists(selected_nodes_path)):
        np.savetxt(selected_nodes_path, idx_attach)
    else:
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
    idx_attach = idx_attach[:size]
    return idx_attach


def obtain_attach_nodes_by_cluster_degree_single(args, edge_index, y_pred, cluster_centers, node_idxs, x, size):
    dis_weight = args.dis_weight


    degrees = (degree(edge_index[0]) + degree(edge_index[1])).cpu().numpy()
    distances = []
    # for id in range(x.shape[0]):
    #     tmp_center_label = y_pred[id]
    #     tmp_center_x = cluster_centers[tmp_center_label]

    #     dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
    #     distances.append(dis)
    for i in range(node_idxs.shape[0]):
        id = node_idxs[i]
        tmp_center_label = y_pred[i]
        tmp_center_x = cluster_centers[tmp_center_label]
        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)
    distances = np.array(distances)
    print("y_pred", y_pred)
    print("node_idxs", node_idxs)
    # label_list = np.unique(labels.cpu())
    # label_list = np.unique(y_pred)
    # labels_dict = {}
    # for i in label_list:
    #     # labels_dict[i] = np.where(labels.cpu()==i)[0]
    #     labels_dict[i] = np.where(y_pred==i)[0]
    #     # filter out labeled nodes
    #     labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    # nontarget_nodes = np.where(y_pred!=args.target_class)[0]
    # non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
    # node_idxs = np.array(non_target_node_idxs)
    # candiadate_distances = distances[node_idxs]
    candiadate_distances = distances
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    # print(candidate_nid_index,node_idxs)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    # selected_nodes = sorted_node_idex[:size]
    selected_nodes = sorted_node_idex
    print("selected_nodes", sorted_node_idex, selected_nodes)
    return selected_nodes


def cluster_degree_selection_seperate_old(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx,
                                          train_edge_index, size, device):
    gcn_encoder = model_construct(args, 'GCN_Encoder', data, device).to(device)
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y, idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train, unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index, None).clone().detach()

    # if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
    #     kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    #     kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    #     cluster_centers = kmedoids.cluster_centers_
    # else:
    #     _, cluster_centers = kmeans(X=encoder_x[seen_node_idx], num_clusters=nclass, distance='euclidean', device=device)

    # kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    # kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    # cluster_centers = kmedoids.cluster_centers_

    encoder_output = gcn_encoder(data.x, train_edge_index, None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    cluster_centers = []
    each_class_size = int(size / (nclass - 1))
    idx_attach = np.array([])
    for label in range(nclass):
        idx_sing_class = (y_pred == label).nonzero()[0]
        # print(encoder_x[idx_sing_class])
        # print((y_pred == label).nonzero()[0])
        print("idx_sing_class", idx_sing_class)
        if (len(idx_sing_class) == 0):
            continue
        print(each_class_size)
        cluster_ids_x, sing_center = KMeans(X=encoder_x[idx_sing_class], num_clusters=each_class_size,
                                            distance='euclidean', device=device)
        # cluster_ids_x = kmeans_predict(encoder_x[idx_sing_class],sing_center, 'euclidean', device=device)
        cand_idx_sing_class = np.array(list(set(unlabeled_idx.cpu().tolist()) & set(idx_sing_class)))
        if (label != nclass - 1):
            sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args, train_edge_index, cluster_ids_x,
                                                                           sing_center, cand_idx_sing_class, encoder_x,
                                                                           each_class_size).astype(int)
        else:
            last_class_size = size - len(idx_attach)
            sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args, train_edge_index, cluster_ids_x,
                                                                           sing_center, cand_idx_sing_class, encoder_x,
                                                                           last_class_size).astype(int)
        idx_attach = np.concatenate((idx_attach, sing_idx_attach))
    return idx_attach


def cluster_degree_selection_seperate_fixed(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx,
                                            train_edge_index, size, device):
    gcn_encoder = model_construct(args, 'GCN_Encoder', data, device).to(device)
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y, idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train, unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index, None).clone().detach()

    # if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
    #     kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    #     kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    #     cluster_centers = kmedoids.cluster_centers_
    # else:
    #     _, cluster_centers = kmeans(X=encoder_x[seen_node_idx], num_clusters=nclass, distance='euclidean', device=device)

    # kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    # kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    # cluster_centers = kmedoids.cluster_centers_

    encoder_output = gcn_encoder(data.x, train_edge_index, None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    cluster_centers = []
    each_class_size = int(size / (nclass - 1))
    idx_attach = np.array([])
    for label in range(nclass):
        if (label == args.target_class):
            continue

        if (label != nclass - 1):
            sing_class_size = each_class_size
        else:
            last_class_size = size - len(idx_attach)
            sing_class_size = last_class_size
        idx_sing_class = (y_pred == label).nonzero()[0]
        # print(encoder_x[idx_sing_class])
        # print((y_pred == label).nonzero()[0])
        print("idx_sing_class", idx_sing_class)
        if (len(idx_sing_class) == 0):
            continue
        print("current_class_size", sing_class_size)
        # cluster_ids_x, sing_center = kmeans(X=encoder_x[idx_sing_class], num_clusters=2, distance='euclidean', device=device)
        # cluster_ids_x = kmeans_predict(encoder_x[idx_sing_class],sing_center, 'euclidean', device=device)
        # kmedoids = cluster.KMedoids(n_clusters=2,method='pam')
        selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset, args.seed, label)
        if (os.path.exists(selected_nodes_path)):
            print(selected_nodes_path)
            sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
            print(sing_idx_attach)
            sing_idx_attach = sing_idx_attach[:sing_class_size]
            idx_attach = np.concatenate((idx_attach, sing_idx_attach))
        else:
            kmedoids = KMeans(n_clusters=2, random_state=1)
            kmedoids.fit(encoder_x[idx_sing_class].detach().cpu().numpy())
            sing_center = kmedoids.cluster_centers_
            cluster_ids_x = kmedoids.predict(encoder_x[idx_sing_class].cpu().numpy())
            cand_idx_sing_class = np.array(list(set(unlabeled_idx.cpu().tolist()) & set(idx_sing_class)))
            if (label != nclass - 1):
                sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args, train_edge_index, cluster_ids_x,
                                                                               sing_center, cand_idx_sing_class,
                                                                               encoder_x, each_class_size).astype(int)
                selected_nodes_foldpath = "./selected_nodes/{}/Seperate/seed{}".format(args.dataset, args.seed)
                if (not os.path.exists(selected_nodes_foldpath)):
                    os.makedirs(selected_nodes_foldpath)
                selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset, args.seed,
                                                                                                label)
                if (not os.path.exists(selected_nodes_path)):
                    np.savetxt(selected_nodes_path, sing_idx_attach)
                else:
                    sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
                sing_idx_attach = sing_idx_attach[:each_class_size]
            else:
                last_class_size = size - len(idx_attach)
                sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args, train_edge_index, cluster_ids_x,
                                                                               sing_center, cand_idx_sing_class,
                                                                               encoder_x, last_class_size).astype(int)
                selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset, args.seed,
                                                                                                label)
                np.savetxt(selected_nodes_path, sing_idx_attach)
                if (not os.path.exists(selected_nodes_path)):
                    np.savetxt(selected_nodes_path, sing_idx_attach)
                else:
                    sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
                sing_idx_attach = sing_idx_attach[:each_class_size]
            idx_attach = np.concatenate((idx_attach, sing_idx_attach))

        # cluster_centers.append(sing_center)

    # idx_attach = obtain_attach_nodes_by_cluster_degree(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
    # idx_attach = obtain_attach_nodes_by_cluster_degree_all(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)

    return idx_attach


def cluster_degree_selection_seperate_fixed1(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx,
                                             train_edge_index, size, device):
    gcn_encoder = model_construct(args, 'GCN_Encoder', data, device).to(device)
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y, idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train, unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index, None).clone().detach()

    # if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
    #     kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    #     kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    #     cluster_centers = kmedoids.cluster_centers_
    # else:
    #     _, cluster_centers = kmeans(X=encoder_x[seen_node_idx], num_clusters=nclass, distance='euclidean', device=device)

    # kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    # kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    # cluster_centers = kmedoids.cluster_centers_

    encoder_output = gcn_encoder(data.x, train_edge_index, None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    cluster_centers = []
    each_class_size = int(size / (nclass - 1))
    idx_attach = np.array([])
    for label in range(nclass):
        if (label == args.target_class):
            continue
        idx_sing_class = (y_pred == label).nonzero()[0]
        # print(encoder_x[idx_sing_class])
        # print((y_pred == label).nonzero()[0])
        print("idx_sing_class", idx_sing_class)
        if (len(idx_sing_class) == 0):
            continue
        print("each_class_size", each_class_size)
        # cluster_ids_x, sing_center = kmeans(X=encoder_x[idx_sing_class], num_clusters=2, distance='euclidean', device=device)
        # cluster_ids_x = kmeans_predict(encoder_x[idx_sing_class],sing_center, 'euclidean', device=device)
        # kmedoids = cluster.KMedoids(n_clusters=2,method='pam')
        kmedoids = KMeans(n_clusters=2, random_state=1)
        kmedoids.fit(encoder_x[idx_sing_class].detach().cpu().numpy())
        sing_center = kmedoids.cluster_centers_
        cluster_ids_x = kmedoids.predict(encoder_x[idx_sing_class].cpu().numpy())
        cand_idx_sing_class = np.array(list(set(unlabeled_idx.cpu().tolist()) & set(idx_sing_class)))
        if (label != nclass - 1):
            sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args, train_edge_index, cluster_ids_x,
                                                                           sing_center, cand_idx_sing_class, encoder_x,
                                                                           each_class_size).astype(int)
        else:
            last_class_size = size - len(idx_attach)
            sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args, train_edge_index, cluster_ids_x,
                                                                           sing_center, cand_idx_sing_class, encoder_x,
                                                                           last_class_size).astype(int)
        idx_attach = np.concatenate((idx_attach, sing_idx_attach))

        # cluster_centers.append(sing_center)

    # idx_attach = obtain_attach_nodes_by_cluster_degree(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
    # idx_attach = obtain_attach_nodes_by_cluster_degree_all(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)

    return idx_attach


def cluster_degree_selection_seperate(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx, train_edge_index,
                                      size, device):
    gcn_encoder = model_construct(args, 'GCN_Encoder', data, device).to(device)
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y, idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train, unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index, None).clone().detach()

    # if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
    #     kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    #     kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    #     cluster_centers = kmedoids.cluster_centers_
    # else:
    #     _, cluster_centers = kmeans(X=encoder_x[seen_node_idx], num_clusters=nclass, distance='euclidean', device=device)

    # kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    # kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    # cluster_centers = kmedoids.cluster_centers_

    encoder_output = gcn_encoder(data.x, train_edge_index, None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    cluster_centers = []
    each_class_size = int(size / (nclass - 1))
    idx_attach = np.array([])
    print("each_class_size", each_class_size)
    for label in range(nclass):
        if (label == args.target_class):
            continue
        if (label != nclass - 1):
            single_size = each_class_size
        else:
            last_class_size = size - len(idx_attach)
            single_size = last_class_size
        idx_sing_class = (y_pred == label).nonzero()[0]
        # print(encoder_x[idx_sing_class])
        # print((y_pred == label).nonzero()[0])
        print("idx_sing_class", idx_sing_class)
        if (len(idx_sing_class) == 0):
            continue
        print("current_class_size", single_size)
        # cluster_ids_x, sing_center = kmeans(X=encoder_x[idx_sing_class], num_clusters=2, distance='euclidean', device=device)
        # cluster_ids_x = kmeans_predict(encoder_x[idx_sing_class],sing_center, 'euclidean', device=device)
        # kmedoids = cluster.KMedoids(n_clusters=2,method='pam')
        kmedoids = KMeans(n_clusters=single_size, random_state=1)
        kmedoids.fit(encoder_x[idx_sing_class].detach().cpu().numpy())
        sing_center = kmedoids.cluster_centers_
        cluster_ids_x = kmedoids.predict(encoder_x[idx_sing_class].cpu().numpy())
        cand_idx_sing_class = np.array(list(set(unlabeled_idx.cpu().tolist()) & set(idx_sing_class)))

        sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args, train_edge_index, cluster_ids_x,
                                                                       sing_center, cand_idx_sing_class, encoder_x,
                                                                       single_size).astype(int)

        # if(label != nclass - 1):
        #     sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,each_class_size).astype(int)
        # else:
        #     last_class_size= size - len(idx_attach)
        #     sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,last_class_size).astype(int)
        idx_attach = np.concatenate((idx_attach, sing_idx_attach))

        # cluster_centers.append(sing_center)

    # idx_attach = obtain_attach_nodes_by_cluster_degree(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
    # idx_attach = obtain_attach_nodes_by_cluster_degree_all(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)

    return idx_attach
