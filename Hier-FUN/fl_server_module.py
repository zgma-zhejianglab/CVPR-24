import torch
import fl_util
import argparse
import numpy as np
import com_module as com
import torch.nn.functional as F
import torch.distributed as dist
from fl_models import create_models
from fl_datasets import load_test_dataset, load_target_test_dataset, load_retain_test_dataset

## initialize the necessary parameters
parser = argparse.ArgumentParser(description='PyTorch Hierarchical Federated Unlearning')
parser.add_argument('--dataset', type=str, default='c10',metavar='N',
                        help='dataset type(fmnist, c10, c100), default: c10')
parser.add_argument('--num_clients', type=int, default=100, metavar='N',
                        help='number of clients (default: 100)')
parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='index of clients (default: 0)')
parser.add_argument('--ip', type=str, default='127.0.0.1', metavar='N',
                        help=' ip address')
parser.add_argument('--port', type=str, default='23333', metavar='N',
                        help=' ip port')
parser.add_argument('--dir', type=str, default='./',metavar='N',
                        help='results directory, default: mnist')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='total epochs for training (default: 300)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 100)')
parser.add_argument('--GPU', type=int, default=0, metavar='N',
                        help='used GPU device (default: 0)')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
parser.add_argument('--lu', type=int, default=100, metavar='N',
                        help='local iterations (default: 0.003)')
parser.add_argument('--ratio', type=float, default=1.0, metavar='N',
                        help='ratio of non-iid data [0-1]')
parser.add_argument('--method', type=int, default=0, metavar='N',
                        help='method type')
args = parser.parse_args()

## establish the connection
torch.cuda.set_device(args.rank)
print('\nestablish connection for server >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
dist.init_process_group(backend="gloo",
                        init_method="tcp://"+args.ip+":"+args.port,
                        world_size=args.num_clients+1,
                        rank=args.rank) 
print("connection connected<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

class Server():
    def __init__(self, rank, model, test_data, test_label, target_test_data, target_test_label, retain_test_data, retain_test_label, test_batch_size, lr):
        super(Server, self).__init__()
        self.rank = rank
        self.model = model
        self.test_data = test_data
        self.test_label = test_label
        self.target_test_data = target_test_data
        self.target_test_label = target_test_label
        self.retain_test_data = retain_test_data
        self.retain_test_label = retain_test_label
        self.bs = test_batch_size
        self.lr = lr

    ## evaluate the performance of global model on different test datasets
    # ds=0, for whole test dataset
    # ds=1, for target dataset
    # ds=2, for retain dataset
    def test(self, ds=0):
        test_batch_size = self.bs
        if ds == 0:
            if self.bs >= len(self.test_label):
                N, test_batch_size = 1, len(self.test_label)
            else:
                N = int((len(self.test_label)-1)/self.bs) + 1
        elif ds == 1:
            if self.bs >= len(self.target_test_label):
                N, test_batch_size = 1, len(self.target_test_label)
            else:
                N = int((len(self.target_test_label)-1)/self.bs)+1
        else:
            if self.bs >= len(self.retain_test_label):
                N, test_batch_size = 1, len(self.retain_test_label)
            else:
                N = int((len(self.retain_test_label)-1)/self.bs)+1

        sum_loss = 0.0
        sum_acc = 0.0
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            for i in range(N):
                start_sub = i * test_batch_size
                end_sub = (i+1) * test_batch_size
                if ds == 0:
                    end_sub = min(end_sub, len(self.test_label))
                    batch_data = (self.test_data[start_sub:end_sub]).to(device)
                    batch_label = (self.test_label[start_sub:end_sub]).to(device)
                elif ds == 1:
                    end_sub = min(end_sub, len(self.target_test_label))
                    batch_data = (self.target_test_data[start_sub:end_sub]).to(device)
                    batch_label = (self.target_test_label[start_sub:end_sub]).to(device)
                else:
                    end_sub = min(end_sub, len(self.retain_test_label))
                    batch_data = (self.retain_test_data[start_sub:end_sub]).to(device)
                    batch_label = (self.retain_test_label[start_sub:end_sub]).to(device)
                out = self.model(batch_data)
                loss = F.nll_loss(out, batch_label)
                sum_loss += loss.item()
                predictions = torch.argmax(out, 1)
                sum_acc += np.mean([float(predictions[i] == batch_label[i]) for i in range(len(batch_label))])
        return sum_loss/N, sum_acc/N

def initialize():
    ## load different datasets for the server
    print('load test dataset for server', ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    test_data, test_label = load_test_dataset(args)
    target_test_data, target_test_label = load_target_test_dataset(args)
    retain_test_data, retain_test_label = load_retain_test_dataset(args)
    print("test set size: %d, contained labels:"%len(test_label),set(list(test_label.numpy())), ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("target test set: %d, contained labels::"%len(target_test_label),set(list(target_test_label.numpy())), ">>>>>>>>>>>>>>>>>>>")
    print("retain test set: %d, contained labels::"%len(retain_test_label),set(list(retain_test_label.numpy())), ">>>>>>>>>>>>>>>>>>>")
    print("load complete>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    ## create folder to store results
    print('create folders for device[%d]'%args.rank, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    dir_name = args.dir + '/server'
    fl_util.create_dir(dir_name)
    dir_name = dir_name + '/'
    print("create complete>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    ## create local models for server
    print('create initlized model for server>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model = create_models(args.dataset, args.GPU)
    print("create complete>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    ## initialized the local model
    print('Send initialized model to all devices>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # global_para = model.get_paras()
    for i in range(args.num_clients):
        print('Send initialized model to device [%d]>>>>>>>>>>>>>>>>>>>>>'%(i+1))
        com.send_data(model.get_full_paras(), i+1)
    print("send complete>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    return Server(args.rank, model, test_data, test_label, target_test_data, target_test_label, retain_test_data, retain_test_label, args.test_batch_size, args.lr)

def overall_test(server, dir_name):  
    l, a = server.test(0)   
    l1, a1 = server.test(1)
    l2, a2 = server.test(2)
    fl_util.save_res([str(a), '\n'], dir_name, 'acc.txt')
    fl_util.save_res([str(l), '\n'], dir_name, 'loss.txt')
    fl_util.save_res([str(a1), '\n'], dir_name, 'target_acc.txt')
    fl_util.save_res([str(l1), '\n'], dir_name, 'target_loss.txt')
    fl_util.save_res([str(a2), '\n'], dir_name, 'retain_acc.txt')
    fl_util.save_res([str(l2), '\n'], dir_name, 'retain_loss.txt')
    return a, l, a1, l1, a2, l2

## plot the figures according to the stored metrics
def overall_plot(dir_name, plot_acc, plot_loss, target_acc, target_loss, retain_acc, retain_loss):
    fl_util.plot_res(dir_name, 'acc', plot_acc)
    fl_util.plot_res(dir_name, 'loss', plot_loss)
    fl_util.plot_res(dir_name, 'target_acc', target_acc)
    fl_util.plot_res(dir_name, 'target_loss', target_loss)
    fl_util.plot_res(dir_name, 'retain_acc', retain_acc)
    fl_util.plot_res(dir_name, 'retain_loss', retain_loss)

## the training process of FedAvg at the server
def fedavg(server):
    dir_name = args.dir+'/server/'
    total_time = []
    cur_time = 0
    plot_acc = []
    plot_loss = []
    target_acc = []
    target_loss = []
    retain_acc = []
    retain_loss = []
    for epoch in range(args.epochs):
        local_paras = []
        local_time = []

        ## receive the local para from all clients
        for i in range(args.num_clients):
            recv_data = com.recv_data(i+1, server.model.get_full_num_paras()+1)
            local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
            local_time.append(float(recv_data[0]))
        cur_time += max(local_time)
        total_time.append(cur_time)

        ## aggregate the local paras and send the updated para to all clients
        agg_para = fl_util.aggregate_paras(local_paras, np.ones([len(local_paras)]))
        for i in range(args.num_clients):
            com.send_data(agg_para, i+1)
        server.model.set_full_paras(agg_para)

        ## evaluate the performance of global model
        a,l,a1,l1,a2,l2 = overall_test(server, dir_name)
        print("Epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, cur_time, l, a))
        plot_acc.append(a)
        plot_loss.append(l)
        target_acc.append(a1)
        target_loss.append(l1)
        retain_acc.append(a2)
        retain_loss.append(l2)
        fl_util.save_res([str(cur_time), '\n'], dir_name, 'time.txt')
        overall_plot(dir_name, plot_acc, plot_loss, target_acc, target_loss, retain_acc, retain_loss)

## the training process of ReTrain at the server, similar to FedAvg
def retrain(server):
    dir_name = args.dir+'/server/'
    total_time = []
    cur_time = 0
    plot_acc = []
    plot_loss = []
    target_acc = []
    target_loss = []
    retain_acc = []
    retain_loss = []
    for epoch in range(args.epochs):
        local_paras = []
        local_time = []
        for i in range(args.num_clients-1):
            recv_data = com.recv_data(i+1, server.model.get_full_num_paras()+1)
            local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
            local_time.append(float(recv_data[0]))
        cur_time += max(local_time)
        total_time.append(cur_time)

        agg_para = fl_util.aggregate_paras(local_paras, np.ones([len(local_paras)]))
        for i in range(args.num_clients-1):
            com.send_data(agg_para, i+1)
        server.model.set_full_paras(agg_para)

        a,l,a1,l1,a2,l2 = overall_test(server, dir_name)
        print("Epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, cur_time, l, a))
        plot_acc.append(a)
        plot_loss.append(l)
        target_acc.append(a1)
        target_loss.append(l1)
        retain_acc.append(a2)
        retain_loss.append(l2)
        fl_util.save_res([str(cur_time), '\n'], dir_name, 'time.txt')
        overall_plot(dir_name, plot_acc, plot_loss, target_acc, target_loss, retain_acc, retain_loss)

## the training process of FedEraser at the server
def federaser(server):
    dir_name = args.dir+'/server/'
    total_time = []
    cur_time = 0
    plot_acc = []
    plot_loss = []
    target_acc = []
    target_loss = []
    retain_acc = []
    retain_loss = []

    ## train 
    for epoch in range(args.epochs):
        local_grads = []
        local_time = []
        para_bak = server.model.get_full_paras()
        for i in range(args.num_clients):
            recv_data = com.recv_data(i+1, server.model.get_full_num_paras()+1)
            local_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
            local_time.append(float(recv_data[0]))
        cur_time += max(local_time)
        total_time.append(cur_time)

        agg_grads = fl_util.aggregate_paras(local_grads, np.ones([len(local_grads)]))
        if epoch == 0:
            target_grads = fl_util.sum_grads([local_grads[-1]])
        else:
            target_grads = fl_util.sum_grads([target_grads, local_grads[-1]])
        updated_para = fl_util.update(server.model.get_full_paras(), agg_grads, args.lr)
        server.model.set_full_paras(updated_para)
        if epoch != args.epochs-1:
            for i in range(args.num_clients):
                com.send_data(updated_para, i+1)

        a,l,a1,l1,a2,l2 = overall_test(server, dir_name)
        print("Epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, cur_time, l, a))
        plot_acc.append(a)
        plot_loss.append(l)
        target_acc.append(a1)
        target_loss.append(l1)
        retain_acc.append(a2)
        retain_loss.append(l2)
        fl_util.save_res([str(cur_time), '\n'], dir_name, 'time.txt')
        overall_plot(dir_name, plot_acc, plot_loss, target_acc, target_loss, retain_acc, retain_loss)
    ## remove the contribution of target device
    target_grads = [g/args.num_clients for g in target_grads]
    global_para = server.model.get_full_paras()
    recover_para = fl_util.update(global_para, target_grads, -args.lr)
    for i in range(args.num_clients-1):
        com.send_data(recover_para, i+1)
    server.model.set_full_paras(recover_para)
    
    l1, a1 = server.test(1)
    l2, a2 = server.test(2)
    tune_target_acc = [a1]
    tune_retain_acc = [a2]
    tune_target_loss = [l1]
    tune_retain_loss = [l2]
    fl_util.save_res([str(a1), '\n'], dir_name, 'tune_target_acc.txt')
    fl_util.save_res([str(l1), '\n'], dir_name, 'tune_target_loss.txt')
    fl_util.save_res([str(a2), '\n'], dir_name, 'tune_retain_acc.txt')
    fl_util.save_res([str(l2), '\n'], dir_name, 'tune_retain_loss.txt')

    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    cur_time = 0
    ## fine-tune on the retain training dataset
    for epoch in range(args.epochs):
        local_grads = []
        local_time = []
        para_bak = server.model.get_full_paras()
        for i in range(args.num_clients-1):
            recv_data = com.recv_data(i+1, server.model.get_full_num_paras()+1)
            local_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
            local_time.append(float(recv_data[0]))
        cur_time += max(local_time)
        total_time.append(cur_time)

        agg_grads = fl_util.aggregate_paras(local_grads, np.ones([len(local_grads)]))
        updated_para = fl_util.update(server.model.get_full_paras(), agg_grads, args.lr)
        server.model.set_full_paras(updated_para)
        for i in range(args.num_clients-1):
            com.send_data(updated_para, i+1)

        l1, a1 = server.test(1)
        l2, a2 = server.test(2)
        tune_target_acc.append(a1)
        tune_retain_acc.append(a2)
        tune_target_loss.append(l1)
        tune_retain_loss.append(l2)
        fl_util.save_res([str(cur_time), '\n'], dir_name, 'tune_time.txt')
        fl_util.save_res([str(tune_target_acc[-1]), '\n'], dir_name, 'tune_target_acc.txt')
        fl_util.save_res([str(tune_target_loss[-1]), '\n'], dir_name, 'tune_target_loss.txt')
        fl_util.save_res([str(tune_retain_acc[-1]), '\n'], dir_name, 'tune_retain_acc.txt')
        fl_util.save_res([str(tune_retain_loss[-1]), '\n'], dir_name, 'tune_retain_loss.txt')
        print("Fine-tune epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, cur_time, l2, a2))
        print()

## the training process of KNOT at each device
def knot(server):
    dir_name = args.dir+'/server/'
    local_grads = []
    local_delay = []
    init_para = server.model.get_full_paras()
    for i in range(args.num_clients):
        recv_data = com.recv_data(i+1, server.model.get_full_num_paras()+1)
        local_grads.append(fl_util.extract_grads(fl_util.flatten_paras(fl_util.reshape_para(recv_data[1:], server.model.get_para_shapes(), args.GPU)), init_para, args.lr))
        local_delay.append(float(recv_data[0]))
    
    ## cluster devices into different clusters
    print('Clustering the devices into clusters>>>>>>>>>>>>>>>>>>>')
    clusters = fl_util.knot_clustering(local_delay, local_grads, 5)
    print("The divided clusters are:",clusters)
    print()
    cluster_heads = []
    for clu in clusters:
        cluster_heads.append(clu[0])
        for i in range(len(clu)):
            if i == 0:
                com.send_data(torch.tensor([0]), clu[i])
                com.send_data(torch.tensor([len(clu)]), clu[i])
                com.send_data(torch.tensor(clu), clu[i])
            else:
                com.send_data(torch.tensor([clu[0]]), clu[i])

    total_time = []
    plot_acc = []
    plot_loss = []
    target_acc = []
    target_loss = []
    retain_acc = []
    retain_loss = []
    cur_time = [0] * len(cluster_heads)
    ## Train
    for epoch in range(args.epochs):
        cand_head, cand_acc, cand_time, cand_para = 0, 0.0, float('inf'), None
        local_paras = []
        for i in range(len(cluster_heads)):
            recv_data = com.recv_data(cluster_heads[i], server.model.get_full_num_paras()+1)
            server.model.set_full_paras(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))

            l, a = server.test(0)
            cur_time[i] += float(recv_data[0])
            if a/cur_time[i] > cand_acc/cand_time:
                cand_head, cand_time, cand_acc, cand_para = i, cur_time[i], a, server.model.get_full_paras()
            print("Current cluster: acc=%.3f, loss=%.3f, time=%.3f, value=%.3f"%(a, l, cur_time[i], a/cur_time[i]))
        print('Selected head:',cluster_heads[cand_head])
        server.model.set_full_paras(cand_para)
        a,l,a1,l1,a2,l2 = overall_test(server, dir_name)
        plot_acc.append(a)
        plot_loss.append(l)
        target_acc.append(a1)
        target_loss.append(l1)
        retain_acc.append(a2)
        retain_loss.append(l2)
        total_time.append(cand_time)
        fl_util.save_res([str(total_time[-1]), '\n'], dir_name, 'time.txt')
        print("Epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, total_time[-1], l, a))
        overall_plot(dir_name, plot_acc, plot_loss, target_acc, target_loss, retain_acc, retain_loss)
        print()

    local_paras = []
    for head in cluster_heads:
        recv_data = com.recv_data(head, server.model.get_full_num_paras())
        local_paras.append(fl_util.reshape_para(recv_data, server.model.get_full_para_shapes(), args.GPU))
    agg_paras = fl_util.aggregate_paras(local_paras, np.ones([len(local_paras)]))
    server.model.set_full_paras(agg_paras)
    for head in cluster_heads:
        com.send_data(agg_paras, head)

    l1, a1 = server.test(1)
    l2, a2 = server.test(2)
    tune_target_acc = [a1]
    tune_retain_acc = [a2]
    tune_target_loss = [l1]
    tune_retain_loss = [l2]
    fl_util.save_res([str(a1), '\n'], dir_name, 'tune_target_acc.txt')
    fl_util.save_res([str(l1), '\n'], dir_name, 'tune_target_loss.txt')
    fl_util.save_res([str(a2), '\n'], dir_name, 'tune_retain_acc.txt')
    fl_util.save_res([str(l2), '\n'], dir_name, 'tune_retain_loss.txt')

    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    ## Unlearn
    local_paras = []
    total_time = []
    cur_time = [0] * len(cluster_heads)
    for i in range(len(cluster_heads)):
        recv_data = com.recv_data(cluster_heads[i], server.model.get_full_num_paras()+1)
        local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
        cur_time[i] += float(recv_data[0])
        total_time.append([i, len(total_time), cur_time[i]])
    total_time = sorted(total_time, key=lambda x:x[2])

    ## Fine-Tune
    for epoch in range(args.epochs):
        print("The current time is:", total_time)
        updated_para = fl_util.aggregate_paras([server.model.get_full_paras(), local_paras[total_time[epoch][1]]], [0.6, 0.4])
        server.model.set_full_paras(updated_para)
        com.send_data(updated_para, cluster_heads[total_time[epoch][0]])
        recv_data = com.recv_data(cluster_heads[total_time[epoch][0]], server.model.get_full_num_paras()+1)

        l1, a1 = server.test(1)
        l2, a2 = server.test(2)
        tune_target_acc.append(a1)
        tune_retain_acc.append(a2)
        tune_target_loss.append(l1)
        tune_retain_loss.append(l2)
        fl_util.save_res([str(total_time[epoch][2]), '\n'], dir_name, 'tune_time.txt')
        fl_util.save_res([str(tune_target_acc[-1]), '\n'], dir_name, 'tune_target_acc.txt')
        fl_util.save_res([str(tune_target_loss[-1]), '\n'], dir_name, 'tune_target_loss.txt')
        fl_util.save_res([str(tune_retain_acc[-1]), '\n'], dir_name, 'tune_retain_acc.txt')
        fl_util.save_res([str(tune_retain_loss[-1]), '\n'], dir_name, 'tune_retain_loss.txt')
        print("Fine-tune epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, total_time[epoch][2], l2, a2))
        print()

        local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
        cur_time[total_time[epoch][0]] += float(recv_data[0])
        total_time.append([total_time[epoch][0], len(total_time), cur_time[total_time[epoch][0]]])
        total_time = sorted(total_time, key=lambda x:x[2])

## training process of the proposed framework, Hier-FUN
def hier_fun(server):
    dir_name = args.dir+'/server/'
    local_grads = []
    local_delay = []
    for i in range(args.num_clients):
        recv_data = com.recv_data(i+1, server.model.get_full_num_paras()+1)
        local_grads.append(fl_util.flatten_paras(fl_util.reshape_para(recv_data[1:], server.model.get_para_shapes(), args.GPU)))
        local_delay.append(float(recv_data[0]))

    ## calculate the correlation matrix
    adj_mat = np.ones([args.num_clients, args.num_clients])
    for i in range(args.num_clients):
        for j in range(i, args.num_clients):
            adj_mat[i][j] = adj_mat[j][i] = (torch.matmul(local_grads[i], local_grads[j]) / (torch.norm(local_grads[i]) * torch.norm(local_grads[j]))).item()
    fl_util.plot_res(dir_name, args.dataset+"_cos_grad", adj_mat)
    for i in range(args.num_clients):
        fl_util.save_res(adj_mat[i].append('\n'), dir_name, 'adj_mat-'+str(args.ratio)+'.txt')
    del local_grads

    ## cluster all devices according to the correlation matrix
    print('Clustering the devices into clusters>>>>>>>>>>>>>>>>>>>')
    delta = 0.9
    if args.dataset == 'c10':
        delta = 0.2
    if args.dataset == 'c100':
        delta = 0.1
    clusters = fl_util.clustering(local_delay, adj_mat, delta)
    print("The divided clusters are:",clusters)
    print()
    cluster_heads = []
    for clu in clusters:
        cluster_heads.append(clu[0])
        for i in range(len(clu)):
            if i == 0:
                com.send_data(torch.tensor([0]), clu[i])
                com.send_data(torch.tensor([len(clu)]), clu[i])
                com.send_data(torch.tensor(clu), clu[i])
            else:
                com.send_data(torch.tensor([clu[0]]), clu[i])
    total_time = []
    plot_acc = []
    plot_loss = []
    target_acc = []
    target_loss = []
    retain_acc = []
    retain_loss = []
    cur_time = [0] * len(cluster_heads)
    ## In-cluster Training
    for epoch in range(args.epochs):
        cand_head, cand_acc, cand_time, cand_para = 0, 0.0, float('inf'), None
        local_paras = []
        for i in range(len(cluster_heads)):
            recv_data = com.recv_data(cluster_heads[i], server.model.get_full_num_paras()+1)
            server.model.set_full_paras(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))

            l, a = server.test(0)
            cur_time[i] += float(recv_data[0])
            if a/cur_time[i] > cand_acc/cand_time:
                cand_head, cand_time, cand_acc, cand_para = i, cur_time[i], a, server.model.get_full_paras()
            print("Current cluster: acc=%.3f, loss=%.3f, time=%.3f, value=%.3f"%(a, l, cur_time[i], a/cur_time[i]))
        print('Selected head:',cluster_heads[cand_head])
        server.model.set_full_paras(cand_para)
        a,l,a1,l1,a2,l2 = overall_test(server, dir_name)
        plot_acc.append(a)
        plot_loss.append(l)
        target_acc.append(a1)
        target_loss.append(l1)
        retain_acc.append(a2)
        retain_loss.append(l2)
        total_time.append(cand_time)
        fl_util.save_res([str(total_time[-1]), '\n'], dir_name, 'time.txt')
        print("Epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, total_time[-1], l, a))
        overall_plot(dir_name, plot_acc, plot_loss, target_acc, target_loss, retain_acc, retain_loss)
        print()

    ## Unlearn
    local_paras = []
    for head in cluster_heads:
        recv_data = com.recv_data(head, server.model.get_full_num_paras())
        local_paras.append(fl_util.reshape_para(recv_data, server.model.get_full_para_shapes(), args.GPU))
    agg_paras = fl_util.aggregate_paras(local_paras, np.ones([len(local_paras)]))
    server.model.set_full_paras(agg_paras)
    for head in cluster_heads:
        com.send_data(agg_paras, head)

    l1, a1 = server.test(1)
    l2, a2 = server.test(2)
    tune_target_acc = [a1]
    tune_retain_acc = [a2]
    tune_target_loss = [l1]
    tune_retain_loss = [l2]
    fl_util.save_res([str(a1), '\n'], dir_name, 'tune_target_acc.txt')
    fl_util.save_res([str(l1), '\n'], dir_name, 'tune_target_loss.txt')
    fl_util.save_res([str(a2), '\n'], dir_name, 'tune_retain_acc.txt')
    fl_util.save_res([str(l2), '\n'], dir_name, 'tune_retain_loss.txt')

    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    ## Fine-Tune
    local_paras = []
    total_time = []
    cur_time = [0] * len(cluster_heads)
    for i in range(len(cluster_heads)):
        recv_data = com.recv_data(cluster_heads[i], server.model.get_full_num_paras()+1)
        local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
        cur_time[i] += float(recv_data[0])
        total_time.append([i, len(total_time), cur_time[i]])
    total_time = sorted(total_time, key=lambda x:x[2])

    for epoch in range(args.epochs):
        print("The current time is:", total_time)
        updated_para = fl_util.aggregate_paras([server.model.get_full_paras(), local_paras[total_time[epoch][1]]], [0.6, 0.4])
        server.model.set_full_paras(updated_para)
        com.send_data(updated_para, cluster_heads[total_time[epoch][0]])
        recv_data = com.recv_data(cluster_heads[total_time[epoch][0]], server.model.get_full_num_paras()+1)

        l1, a1 = server.test(1)
        l2, a2 = server.test(2)
        tune_target_acc.append(a1)
        tune_retain_acc.append(a2)
        tune_target_loss.append(l1)
        tune_retain_loss.append(l2)
        fl_util.save_res([str(total_time[epoch][2]), '\n'], dir_name, 'tune_time.txt')
        fl_util.save_res([str(tune_target_acc[-1]), '\n'], dir_name, 'tune_target_acc.txt')
        fl_util.save_res([str(tune_target_loss[-1]), '\n'], dir_name, 'tune_target_loss.txt')
        fl_util.save_res([str(tune_retain_acc[-1]), '\n'], dir_name, 'tune_retain_acc.txt')
        fl_util.save_res([str(tune_retain_loss[-1]), '\n'], dir_name, 'tune_retain_loss.txt')
        print("Fine-tune epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, total_time[epoch][2], l2, a2))
        print()

        local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
        cur_time[total_time[epoch][0]] += float(recv_data[0])
        total_time.append([total_time[epoch][0], len(total_time), cur_time[total_time[epoch][0]]])
        total_time = sorted(total_time, key=lambda x:x[2])

## training process of Random at the server
def random_(server):
    dir_name = args.dir+'/server/'

    ## radnomly cluster all devices
    print('Clustering the devices into clusters>>>>>>>>>>>>>>>>>>>')
    clusters = fl_util.random_clustering(args.num_clients, 5)
    print("The divided clusters are:",clusters)
    print()
    cluster_heads = []
    for clu in clusters:
        cluster_heads.append(clu[0])
        for i in range(len(clu)):
            if i == 0:
                com.send_data(torch.tensor([0]), clu[i])
                com.send_data(torch.tensor([len(clu)]), clu[i])
                com.send_data(torch.tensor(clu), clu[i])
            else:
                com.send_data(torch.tensor([clu[0]]), clu[i])
    total_time = []
    plot_acc = []
    plot_loss = []
    target_acc = []
    target_loss = []
    retain_acc = []
    retain_loss = []
    cur_time = [0] * len(cluster_heads)
    ## In-cluster Training
    for epoch in range(args.epochs):
        cand_head, cand_acc, cand_time, cand_para = 0, 0.0, float('inf'), None
        local_paras = []
        for i in range(len(cluster_heads)):
            recv_data = com.recv_data(cluster_heads[i], server.model.get_full_num_paras()+1)
            server.model.set_full_paras(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))

            l, a = server.test(0)
            cur_time[i] += float(recv_data[0])
            if a/cur_time[i] > cand_acc/cand_time:
                cand_head, cand_time, cand_acc, cand_para = i, cur_time[i], a, server.model.get_full_paras()
            print("Current cluster: acc=%.3f, loss=%.3f"%(a, l))
        print('Selected head:',cluster_heads[cand_head])
        server.model.set_full_paras(cand_para)
        a,l,a1,l1,a2,l2 = overall_test(server, dir_name)
        plot_acc.append(a)
        plot_loss.append(l)
        target_acc.append(a1)
        target_loss.append(l1)
        retain_acc.append(a2)
        retain_loss.append(l2)
        total_time.append(cand_time)
        fl_util.save_res([str(total_time[-1]), '\n'], dir_name, 'time.txt')
        print("Epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, total_time[-1], l, a))
        overall_plot(dir_name, plot_acc, plot_loss, target_acc, target_loss, retain_acc, retain_loss)
        print()
    ## Unlearn
    local_paras = []
    for head in cluster_heads:
        recv_data = com.recv_data(head, server.model.get_full_num_paras())
        local_paras.append(fl_util.reshape_para(recv_data, server.model.get_full_para_shapes(), args.GPU))
    agg_paras = fl_util.aggregate_paras(local_paras, np.ones([len(local_paras)]))
    server.model.set_full_paras(agg_paras)
    for head in cluster_heads:
        com.send_data(agg_paras, head)

    l1, a1 = server.test(1)
    l2, a2 = server.test(2)
    tune_target_acc = [a1]
    tune_retain_acc = [a2]
    tune_target_loss = [l1]
    tune_retain_loss = [l2]
    fl_util.save_res([str(a1), '\n'], dir_name, 'tune_target_acc.txt')
    fl_util.save_res([str(l1), '\n'], dir_name, 'tune_target_loss.txt')
    fl_util.save_res([str(a2), '\n'], dir_name, 'tune_retain_acc.txt')
    fl_util.save_res([str(l2), '\n'], dir_name, 'tune_retain_loss.txt')

    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    local_paras = []
    total_time = []
    cur_time = [0] * len(cluster_heads)
    ## Fine-Tune
    for i in range(len(cluster_heads)):
        recv_data = com.recv_data(cluster_heads[i], server.model.get_full_num_paras()+1)
        local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
        cur_time[i] += float(recv_data[0])
        total_time.append([i, len(total_time), cur_time[i]])
    total_time = sorted(total_time, key=lambda x:x[2])

    for epoch in range(args.epochs):
        print("The current time is:", total_time)
        # updated_para = fl_util.update(server.model.get_full_paras(), local_grads[total_time[epoch][1]], args.lr)
        updated_para = fl_util.aggregate_paras([server.model.get_full_paras(), local_paras[total_time[epoch][1]]], [0.6, 0.4])
        server.model.set_full_paras(updated_para)
        com.send_data(updated_para, cluster_heads[total_time[epoch][0]])
        recv_data = com.recv_data(cluster_heads[total_time[epoch][0]], server.model.get_full_num_paras()+1)

        l1, a1 = server.test(1)
        l2, a2 = server.test(2)
        tune_target_acc.append(a1)
        tune_retain_acc.append(a2)
        tune_target_loss.append(l1)
        tune_retain_loss.append(l2)
        fl_util.save_res([str(total_time[epoch][2]), '\n'], dir_name, 'tune_time.txt')
        fl_util.save_res([str(tune_target_acc[-1]), '\n'], dir_name, 'tune_target_acc.txt')
        fl_util.save_res([str(tune_target_loss[-1]), '\n'], dir_name, 'tune_target_loss.txt')
        fl_util.save_res([str(tune_retain_acc[-1]), '\n'], dir_name, 'tune_retain_acc.txt')
        fl_util.save_res([str(tune_retain_loss[-1]), '\n'], dir_name, 'tune_retain_loss.txt')
        print("Fine-tune epoch[%d]>>>>>>>>>time=%.2f>>>>>>>>>loss=%.2f>>>>>>>>>>>accuracy=%.2f"%(epoch, total_time[epoch][2], l2, a2))
        print()

        local_paras.append(fl_util.reshape_para(recv_data[1:], server.model.get_full_para_shapes(), args.GPU))
        cur_time[total_time[epoch][0]] += float(recv_data[0])
        total_time.append([total_time[epoch][0], len(total_time), cur_time[total_time[epoch][0]]])
        total_time = sorted(total_time, key=lambda x:x[2])

def main():
    server = initialize()
    if args.method == 0:
        fedavg(server)
    elif args.method == 1:
        retrain(server)
    elif args.method == 2:
        federaser(server)    
    elif args.method == 3:
        knot(server)  
    elif args.method == 4:
        hier_fun(server) 
    elif args.method == 5:
        random_(server)
main()