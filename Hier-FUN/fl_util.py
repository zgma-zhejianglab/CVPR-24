import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def update(paras, grads, lr):
    return [p-lr*g for (p,g) in zip(paras, grads)]

def flatten_paras(paras):
    flatten_paras = torch.tensor([]).to(paras[0].device)
    for p in paras:
        flatten_paras = torch.cat([flatten_paras, torch.flatten(p.data.clone().detach())])
    return flatten_paras

def aggregate_paras(local_paras, num_samples):
    temp_para = []
    total_samples = np.sum(num_samples)
    for i in range(len(local_paras[0])):
        for j in range(len(local_paras)):
            if j == 0:
                temp_para.append(local_paras[j][i].data * (num_samples[j] / total_samples))
            else:
                temp_para[i] = torch.add(temp_para[i], (local_paras[j][i].data * (num_samples[j] / total_samples)).to(temp_para[i].device))
    return temp_para

def sum_grads(grads):
    grad_sum = []
    for i in range(len(grads[0])):
        for j in range(len(grads)):
            if j == 0:
                grad_sum.append(grads[j][i].data)
            else:
                grad_sum[i] = torch.add(grad_sum[i], grads[j][i].data.to(grad_sum[i].device))
    return grad_sum

def create_dir(path):
    os.makedirs(path)

def extract_grads(new_para, old_para, lr):
    return [torch.div(torch.sub(p,p1), lr) for (p,p1) in zip(old_para, new_para)]

def reshape_para(flat_para, shapes, gpu_device):
    start = 0
    shaped_para = []
    for i in range(len(shapes)):
        temp = flat_para[start:(start+shapes[i].numel())]
        shaped_para.append(torch.reshape(temp, shapes[i]).to(gpu_device))
        start += shapes[i].numel()
    return shaped_para

## generate main labels for different groups of devices
def gen_classes(dataset):
    if dataset in ['fmnist', 'c10']:
        return [[0,1,2], [3,4,5], [0,3,6], [1,4,7],[2,5,6], [8,9]]
    if dataset == 'c100':
        classes = [[],[],[],[],[]]
        for i in range(95):
            classes[i//19].append(i)
        classes.append([95,96,97,98,99])
        return classes
    raise Exception('Specified dataset NotFound:', dataset)

def get_group_id(classes, cls):
    group_id = []
    for g in range(len(classes)):
        if cls in classes[g]:
            group_id.append(g)
    return group_id

## generate data size with different non-iid level
def gen_partiion_sizes(dataset, num_clients, ratio):
    num_classes = 10 if dataset in ['fmnist', 'c10'] else 100
    par_sizes = np.zeros([num_clients, num_classes])
    clients_per_group = num_clients // 5
    classes = gen_classes(dataset)
    for cls in range(num_classes):
        if cls in classes[-1]:
            par_sizes[num_clients-1][cls] = 1
            continue
        groups = get_group_id(classes, cls)
        for client in range(num_clients-1):
            if client // clients_per_group in groups:
                par_sizes[client][cls] = ratio / (clients_per_group * len(groups))
            else:
                par_sizes[client][cls] = (1-ratio) / (num_clients-1 - clients_per_group * len(groups))
    return par_sizes

def find(clusters, x):
    if clusters[x] == x:
        return x
    else:
        return find(clusters, clusters[x])
def union(clusters, x, y):
    par_x = find(clusters, x)
    par_y = find(clusters, y)
    clusters[par_x] = par_y
def clustering(local_delays, adj_mat, delta):
    parents = {}
    num_clients = len(local_delays)

    ## merge the devices when cosine similarity is larger than delta
    for i in range(num_clients):
        parents[i+1] = i+1
    for i in range(num_clients):
        for j in range(i, num_clients):
            if adj_mat[i][j] >= delta:
                union(parents, i+1, j+1)
    ## collect the divided groups, note that each group contains the devices with similar data distribution
    groups = {}
    for i in range(num_clients):
        if parents[i+1] not in groups:
            groups[parents[i+1]] = []
        groups[parents[i+1]].append([i+1,local_delays[i]])

    ## sort the groups according to their possessed resources, i.e., training delay
    for k in groups.keys():
        groups[k] = sorted(groups[k], key=lambda x:x[1])

    ## Set K as int(np.ceil(len(local_delays) / len(groups.keys())))
    clusters = []
    K = int(np.ceil(len(local_delays) / len(groups.keys())))
    indices = [0 for _ in range(len(groups.keys()))]

    ## divide the devices in each group into K clusters
    for k in range(K):
        cur_cluster = []
        idx = 0
        for key in groups.keys():
            for i in range(indices[idx], int(np.ceil((k+1)/K * len(groups[key])))):
                cur_cluster.append(groups[key][i])
            indices[idx] = int(np.ceil((k+1)/K * len(groups[key])))
            idx += 1
        sorted_clu = sorted(cur_cluster, key=lambda x:x[1])
        clusters.append([ele[0] for ele in sorted_clu])
    return clusters

## clustering for KNOT
def knot_clustering(local_delay, local_grads, K):
    a=b=1
    min_delay, max_delay = min(local_delay), max(local_delay)
    cluster_delay, cluster_dis = [], []
    for k in range(K):
        cluster_delay.append(min_delay+(max_delay-min_delay)*k/(K-1))
        cluster_dis.append((k+1)/K)

    global_grad = flatten_paras(aggregate_paras(local_grads, np.ones([len(local_grads)])))
    clusters = [[] for _ in range(K)]
    for i in range(len(local_delay)):
        min_val, cand_id = float('inf'), 0
        grad_i = flatten_paras(local_grads[i])
        for k in range(K):
            s_i = (torch.matmul(global_grad, grad_i) / (torch.norm(global_grad)* torch.norm(grad_i))).item()
            ## to incorporate the data sizes, we add the number of devices in each cluster into val
            val = torch.norm(torch.tensor([a*(local_delay[i]-cluster_delay[k]), b*s_i])).item() + len(clusters[k])*10
            if val < min_val:
                min_val = val
                cand_id = k
        clusters[cand_id].append(i+1)
    return clusters

## radnomly clustering while mataining the number of devices in each cluster balanced
def random_clustering(num_clients, K):
    random.seed(0)
    clusters = [[] for _ in range(K)]
    shuffle_devices = [i+1 for i in range(num_clients)]
    random.shuffle(shuffle_devices)
    for i in range(num_clients):
        clusters[i%K].append(shuffle_devices[i])
    return clusters

## plot figures
def plot_res(dir, name, data):
    plt.ioff()
    plt.figure()
    plt.title(name)
    plt.xlabel('epochs')
    plt.ylabel(name)
    plt.plot(data)
    plt.savefig((dir+name)+'.png')
    plt.ioff()
    plt.close()

## save the result to the file
def save_res(data, path, name):
    with open(path+name, 'a+') as f:
        for i in range(len(data)):
            f.write(str(data[i]))