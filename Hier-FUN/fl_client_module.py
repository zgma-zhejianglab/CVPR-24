import torch
import random
import fl_util
import argparse
import numpy as np
import com_module as com
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from fl_models import create_models
from fl_datasets import load_train_dataset, load_target_test_dataset

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
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=20, metavar='N',
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
print('\nestablish connection for device[%d] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'%(args.rank))
dist.init_process_group(backend="gloo",
                        init_method="tcp://"+args.ip+":"+args.port,
                        world_size=args.num_clients+1,
                        rank=args.rank) 
print("connection connected<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

## specify the delay for each device
def get_delays(dataset):
    random.seed(args.rank-1)
    if dataset == 'fmnist':
        delays = [5.387, 7.183, 10.775, 16.523]
    elif dataset == 'c10':
        delays = [9.298, 12.364, 18.546, 28.437]
    elif dataset == 'c100':
        delays = [11.728, 15.637, 23.436, 35.965]
    kind = random.randint(0,3)
    return delays[kind]

class Client():
    def __init__(self, rank, model, train_data, train_label, batch_size, opt, lr, local_updates, delay):
        super(Client, self).__init__()
        self.rank = rank
        self.model = model
        self.train_data = train_data
        self.train_label = train_label
        self.bs = batch_size
        self.opt = opt
        self.lr = lr
        self.lu = local_updates
        self.delay = delay
    
    ## train local models with self.lu iterations
    def train(self, clustering=False):
        loss_sum = 0
        acc_sum = 0   
        num_samples = len(self.train_label)
        indices = [i for i in range(num_samples)]
        random.shuffle(indices)
        device = next(self.model.parameters()).device
        for pg in self.opt.param_groups:
            pg["lr"] = self.lr
        agg_grads = []
        itr = idx = 0
        self.model.train()
        while itr < self.lu:
            self.opt.zero_grad()            
            if (idx+1) * self.bs > len(indices):
                random.shuffle(indices)
                idx = 0
            start_idx = (idx * self.bs)
            end_idx = ((idx+1) * self.bs)
            batch_data = (self.train_data[indices[start_idx:end_idx]]).to(device)
            batch_label = (self.train_label[indices[start_idx:end_idx]]).to(device)
            batch_output = self.model(batch_data)
            batch_loss = F.nll_loss(batch_output, batch_label)
            batch_loss.backward()
            self.opt.step()
            loss_sum += batch_loss.item()
            if clustering:
                agg_grads.append(self.model.get_grads())
            idx += 1
            itr += 1
            predictions = torch.argmax(batch_output, 1)
            acc_sum += np.mean([float(predictions[i] == batch_label[i]) for i in range(len(batch_label))])
        print('loss=%.2f'%(loss_sum/self.lu),', acc=%.5f'%(acc_sum/self.lu),  ', lr=%.5f'%self.lr, end='')
        if clustering:
            return loss_sum / self.lu, fl_util.aggregate_paras(agg_grads, np.ones([len(agg_grads)]))
        else:
            return loss_sum / self.lu

def initialize():
    ## initialize the delay for each device
    delay = get_delays(args.dataset)
    print("The delay is %.3f"%delay)

    ## load dataset for each devices.
    print('load dataset for device[%d]'%args.rank, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    train_data, train_label = load_train_dataset(args.dataset, args.num_clients, args.ratio, args.rank)
    print("load completed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

    ## create local models for each device
    print('create model for device[%d]'%args.rank, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    model = create_models(args.dataset, args.GPU)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99)
    print("create completed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

    ## receive initilized global model from the server
    print('receive initialized model for device[%d]'%args.rank, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    init_para = com.recv_data(0, model.get_full_num_paras())
    init_para = fl_util.reshape_para(init_para, model.get_full_para_shapes(), args.GPU)
    model.set_full_paras(init_para)
    print("initialization completed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    return Client(args.rank, model, train_data, train_label, args.batch_size, opt, args.lr, args.lu, delay)

## evaluate the performance of model on test dataset
def test(model, test_data, test_label):
    sum_loss = 0.0
    sum_acc = 0.0
    test_batch_size = args.test_batch_size
    device = next(model.parameters()).device
    N = int(len(test_label)/test_batch_size)
    if N == 0:
        N = 1
        test_batch_size = len(test_data)
    model.eval()
    with torch.no_grad():
        for i in range(N):
            start_sub = i * test_batch_size
            end_sub = (i+1) * test_batch_size
            batch_data = (test_data[start_sub:end_sub]).to(device)
            batch_label = (test_label[start_sub:end_sub]).to(device)
            out = model(batch_data)
            loss = F.nll_loss(out, batch_label)
            sum_loss += loss.item()
            predictions = torch.argmax(out, 1)
            sum_acc += np.mean([float(predictions[i] == batch_label[i]) for i in range(len(batch_label))])
    return sum_loss/N, sum_acc/N

## the training process of FedAvg at each device
def fedavg(client):
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.GPU)
    else:
        device = 'cpu'
    for epoch in range(args.epochs):
        print("Epoch [%d]>>>>>>>"%(epoch), end=' ')
        client.train()
        print()

        ## send the updated local para and consumed time to the server
        local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
        local_para.extend(client.model.get_full_paras())
        com.send_data(local_para, 0)

        ## receive the global para from the server
        global_para = com.recv_data(0, client.model.get_full_num_paras())
        client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

## the training process of ReTrain at each device
def retrain(client):
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.GPU)
    else:
        device = 'cpu'
    
    ## train local models except for the target device
    if args.rank != args.num_clients:
        for epoch in range(args.epochs):
            print("Epoch [%d]>>>>>>>"%(epoch), end=' ')
            client.train()
            print()
            local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
            local_para.extend(client.model.get_full_paras())
            com.send_data(local_para, 0)
            global_para = com.recv_data(0, client.model.get_full_num_paras())
            client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

## the training process of FedEraser at each device
def federaser(client):
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.GPU)
    else:
        device = 'cpu'
    
    ## train local models with args.epochs like FedAvg
    for epoch in range(args.epochs):
        print("Epoch [%d]>>>>>>>"%(epoch), end=' ')
        client.train()
        print()
        local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
        local_para.extend(client.model.get_full_paras())
        com.send_data(local_para, 0)
        if epoch != args.epochs-1 or args.rank != args.num_clients:
            global_para = com.recv_data(0, client.model.get_full_num_paras())
            client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

    ## start fine-tuning the unlearned model
    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    if args.rank != args.num_clients:
        for epoch in range(args.epochs):
            print("Epoch [%d]>>>>>>>"%(epoch), end=' ')
            client.train()
            print()
            local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
            local_para.extend(client.model.get_full_paras())
            com.send_data(local_para, 0)
            global_para = com.recv_data(0, client.model.get_full_num_paras())
            client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

## the training process of KNOT at each device
def knot(client):
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.GPU)
    else:
        device = 'cpu'
    
    ## receive the clustering results from the server
    print('Preparing for clustering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Pre-train: ', end='')
    client.train()
    local_para = [torch.tensor([client.delay]).to(device)]
    local_para.extend(client.model.get_full_paras())
    com.send_data(local_para, 0)
    head = int(com.recv_data(0, 1).item())
    print('\nThe head is: ', head)
    if head == 0:
        num_cluster_clients = int(com.recv_data(0, 1).item())
        groups = com.recv_data(0, num_cluster_clients).int().tolist()
        print('The group contains: ', groups)
    print()
    target_grads = None
    for epoch in range(args.epochs):
        para_bak = client.model.get_full_paras()
        print("Epoch [%d]>>>>>>>"%(epoch), end=' ')
        client.train()
        print()
        ## For cluster heads
        if head == 0:
            cluster_grads = []
            cluster_time = []
            for c in groups:
                ## send the cluster para to the devices within the cluster
                if c == args.rank:
                    cluster_grads.append(fl_util.extract_grads(client.model.get_full_paras(), para_bak, args.lr))
                    cluster_time.append(client.delay-1+random.random()*2)
                else:
                    recv_data = com.recv_data(c, client.model.get_full_num_paras()+1)
                    cluster_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], client.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
                    cluster_time.append(float(recv_data[0]))
                ## store the historical gradients of all devices within the cluster
                ## note that only the gradient of target device is retained here for reducing the GPU memory overhead of our workstation
                if c == args.num_clients:
                    if epoch == 0:
                        target_grads = fl_util.sum_grads([cluster_grads[-1]])
                    else:
                        target_grads = fl_util.sum_grads([target_grads, cluster_grads[-1]])
            ## aggregate the received gradients
            agg_grads = fl_util.aggregate_paras(cluster_grads, np.ones([len(cluster_grads)]))
            updated_para = fl_util.update(para_bak, agg_grads, args.lr)
            client.model.set_full_paras(updated_para)
            for c in groups:
                if c != args.rank:
                    com.send_data(updated_para, c)
            local_para = [torch.tensor([max(cluster_time)]).to(device)]
            local_para.extend(updated_para)
        else:
            ## for the dveices within the cluster
            local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
            local_para.extend(client.model.get_full_paras())
            com.send_data(local_para, head)
            global_para = com.recv_data(head, client.model.get_full_num_paras())
            client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    if head == 0:
        ## cluster head removes the contribution of target device
        if args.num_clients in groups:
            target_grads = [g/len(groups)for g in target_grads]
            cur_para = client.model.get_full_paras()
            recover_para = fl_util.update(cur_para, target_grads, -args.lr)
        else:
            recover_para = client.model.get_full_paras()
        ## send the unlearned model to the server
        com.send_data(recover_para, 0)
        global_para = com.recv_data(head, client.model.get_full_num_paras())
        client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))
        for c in groups:
            if c != args.rank:
                com.send_data(client.model.get_full_paras(), c)
    else:
        global_para = com.recv_data(head, client.model.get_full_num_paras())
        client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))
    
    ## fine-tune the unlearned model with the established clustered architecture
    if args.rank != args.num_clients:
        for epoch in range(args.epochs):
            print("Fine-tune epoch [%d]>>>>>>>"%(epoch), end=' ')
            para_bak = client.model.get_full_paras()
            client.train()
            print()
            if head == 0:
                cluster_grads = []
                cluster_time = []
                for c in groups:
                    if c == args.rank:
                        cluster_grads.append(fl_util.extract_grads(client.model.get_full_paras(), para_bak, args.lr))
                        cluster_time.append(client.delay-1+random.random()*2)
                    elif c != args.num_clients:
                        recv_data = com.recv_data(c, client.model.get_full_num_paras()+1)
                        cluster_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], client.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
                        cluster_time.append(float(recv_data[0]))
                agg_grads = fl_util.aggregate_paras(cluster_grads, np.ones([len(cluster_grads)]))
                local_para = [torch.tensor([max(cluster_time)]).to(device)]
                local_para.extend(fl_util.update(para_bak, agg_grads, args.lr))
                com.send_data(local_para, 0)
                global_para = com.recv_data(head, client.model.get_full_num_paras())
                global_para = fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU)
                client.model.set_full_paras(global_para)
                for c in groups:
                    if c != args.rank and c != args.num_clients:
                        com.send_data(global_para, c)
            else:
                local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
                local_para.extend(client.model.get_full_paras())
                com.send_data(local_para, head)
                global_para = com.recv_data(head, client.model.get_full_num_paras())
                client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

## training process of the proposed framework, Hier-FUN
def hier_fun(client):
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.GPU)
    else:
        device = 'cpu'
    
    ## receive the clustering results from the server
    print('Preparing for clustering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Pre-train: ', end='')
    init_para = client.model.get_full_paras()
    _, grad = client.train(clustering=True)
    client.model.set_full_paras(init_para)
    local_para = [torch.tensor([client.delay]).to(device)]
    local_para.extend(grad)
    com.send_data(local_para, 0)
    head = int(com.recv_data(0, 1).item())
    print('\nThe head is: ', head)
    if head == 0:
        num_cluster_clients = int(com.recv_data(0, 1).item())
        groups = com.recv_data(0, num_cluster_clients).int().tolist()
        print('The group contains: ', groups)
    print()
    target_grads = None
    for epoch in range(args.epochs):
        para_bak = client.model.get_full_paras()
        print("Epoch [%d]>>>>>>>"%(epoch), end=' ')
        client.train()
        print()
        ## For cluster heads
        if head == 0:
            cluster_grads = []
            cluster_time = []
            ## send the cluster para to the devices within the cluster
            for c in groups:
                if c == args.rank:
                    cluster_grads.append(fl_util.extract_grads(client.model.get_full_paras(), para_bak, args.lr))
                    cluster_time.append(client.delay-1+random.random()*2)
                else:
                    recv_data = com.recv_data(c, client.model.get_full_num_paras()+1)
                    cluster_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], client.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
                    cluster_time.append(float(recv_data[0]))
                ## store the historical gradients of all devices within the cluster
                ## note that only the gradient of target device is retained here for reducing the GPU memory overhead of our workstation
                if c == args.num_clients:
                    if epoch == 0:
                        target_grads = fl_util.sum_grads([cluster_grads[-1]])
                    else:
                        target_grads = fl_util.sum_grads([target_grads, cluster_grads[-1]])
            ## aggregate the received gradients
            agg_grads = fl_util.aggregate_paras(cluster_grads, np.ones([len(cluster_grads)]))
            updated_para = fl_util.update(para_bak, agg_grads, args.lr)
            client.model.set_full_paras(updated_para)
            for c in groups:
                if c != args.rank:
                    com.send_data(updated_para, c)
            local_para = [torch.tensor([max(cluster_time)]).to(device)]
            local_para.extend(updated_para)
            com.send_data(local_para, 0)
        else:
            local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
            local_para.extend(client.model.get_full_paras())
            com.send_data(local_para, head)
            global_para = com.recv_data(head, client.model.get_full_num_paras())
            client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    ## cluster head removes the contribution of target device
    if head == 0:
        if args.num_clients in groups:
            target_grads = [g/len(groups)for g in target_grads]
            cur_para = client.model.get_full_paras()
            recover_para = fl_util.update(cur_para, target_grads, -args.lr)
        else:
            recover_para = client.model.get_full_paras()
        ## send the unlearned model to the server
        com.send_data(recover_para, 0)
        global_para = com.recv_data(head, client.model.get_full_num_paras())
        client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))
        for c in groups:
            if c != args.rank:
                com.send_data(client.model.get_full_paras(), c)
    else:
        global_para = com.recv_data(head, client.model.get_full_num_paras())
        client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))
    ## fine-tune the unlearned model with the established clustered architecture
    if args.rank != args.num_clients:
        for epoch in range(args.epochs):
            print("Fine-tune epoch [%d]>>>>>>>"%(epoch), end=' ')
            para_bak = client.model.get_full_paras()
            client.train()
            print()
            if head == 0:
                cluster_grads = []
                cluster_time = []
                for c in groups:
                    if c == args.rank:
                        cluster_grads.append(fl_util.extract_grads(client.model.get_full_paras(), para_bak, args.lr))
                        cluster_time.append(client.delay-1+random.random()*2)
                    elif c != args.num_clients:
                        recv_data = com.recv_data(c, client.model.get_full_num_paras()+1)
                        cluster_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], client.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
                        cluster_time.append(float(recv_data[0]))
                agg_grads = fl_util.aggregate_paras(cluster_grads, np.ones([len(cluster_grads)]))
                local_para = [torch.tensor([max(cluster_time)]).to(device)]
                local_para.extend(fl_util.update(para_bak, agg_grads, args.lr))
                com.send_data(local_para, 0)
                global_para = com.recv_data(head, client.model.get_full_num_paras())
                global_para = fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU)
                client.model.set_full_paras(global_para)
                for c in groups:
                    if c != args.rank and c != args.num_clients:
                        com.send_data(global_para, c)
            else:
                local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
                local_para.extend(client.model.get_full_paras())
                com.send_data(local_para, head)
                global_para = com.recv_data(head, client.model.get_full_num_paras())
                client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))
    
## training process of Random at each device
def random_(client):
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.GPU)
    else:
        device = 'cpu'
    ## receive the clustering results from the server
    print('Preparing for clustering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    head = int(com.recv_data(0, 1).item())
    print('\nThe head is: ', head)
    if head == 0:
        num_cluster_clients = int(com.recv_data(0, 1).item())
        groups = com.recv_data(0, num_cluster_clients).int().tolist()
        print('The group contains: ', groups)
        if args.num_clients in groups:
            target_test_data, target_test_label = load_target_test_dataset(args)
    print()
    target_grads = None
    for epoch in range(args.epochs):
        para_bak = client.model.get_full_paras()
        print("Epoch [%d]>>>>>>>"%(epoch), end=' ')
        client.train()
        print()
        ## for cluster heads
        if head == 0:
            cluster_grads = []
            cluster_time = []
            for c in groups:
                if c == args.rank:
                    cluster_grads.append(fl_util.extract_grads(client.model.get_full_paras(), para_bak, args.lr))
                    cluster_time.append(client.delay-1+random.random()*2)
                else:
                    recv_data = com.recv_data(c, client.model.get_full_num_paras()+1)
                    cluster_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], client.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
                    cluster_time.append(float(recv_data[0]))
                if c == args.num_clients:
                    if epoch == 0:
                        target_grads = fl_util.sum_grads([cluster_grads[-1]])
                    else:
                        target_grads = fl_util.sum_grads([target_grads, cluster_grads[-1]])
            agg_grads = fl_util.aggregate_paras(cluster_grads, np.ones([len(cluster_grads)]))
            updated_para = fl_util.update(para_bak, agg_grads, args.lr)
            client.model.set_full_paras(updated_para)
            for c in groups:
                if c != args.rank:
                    com.send_data(updated_para, c)
            local_para = [torch.tensor([max(cluster_time)]).to(device)]
            local_para.extend(updated_para)
            com.send_data(local_para, 0)
            if args.num_clients in groups:
                tar_loss, tar_acc  = test(client.model, target_test_data, target_test_label)
                print("Traget accuracy:%.2f, target loss: %.2f"%(tar_acc, tar_loss))
        else:
            local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
            local_para.extend(client.model.get_full_paras())
            com.send_data(local_para, head)
            global_para = com.recv_data(head, client.model.get_full_num_paras())
            client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

    print("\nStartle fine-tune>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    ## remove the contribution of target device
    if head == 0:
        if args.num_clients in groups:
            target_grads = [g/len(groups)for g in target_grads]
            cur_para = client.model.get_full_paras()
            recover_para = fl_util.update(cur_para, target_grads, -args.lr)
        else:
            recover_para = client.model.get_full_paras()
        com.send_data(recover_para, 0)
        global_para = com.recv_data(head, client.model.get_full_num_paras())
        client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))
        for c in groups:
            if c != args.rank:
                com.send_data(client.model.get_full_paras(), c)
    else:
        global_para = com.recv_data(head, client.model.get_full_num_paras())
        client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))
    ## fine-tune the unelarned model with the established clustered architecture
    if args.rank != args.num_clients:
        for epoch in range(args.epochs):
            print("Fine-tune epoch [%d]>>>>>>>"%(epoch), end=' ')
            para_bak = client.model.get_full_paras()
            client.train()
            print()
            if head == 0:
                cluster_grads = []
                cluster_time = []
                for c in groups:
                    if c == args.rank:
                        cluster_grads.append(fl_util.extract_grads(client.model.get_full_paras(), para_bak, args.lr))
                        cluster_time.append(client.delay-1+random.random()*2)
                    elif c != args.num_clients:
                        recv_data = com.recv_data(c, client.model.get_full_num_paras()+1)
                        cluster_grads.append(fl_util.extract_grads(fl_util.reshape_para(recv_data[1:], client.model.get_full_para_shapes(), args.GPU), para_bak, args.lr))
                        cluster_time.append(float(recv_data[0]))
                agg_grads = fl_util.aggregate_paras(cluster_grads, np.ones([len(cluster_grads)]))
                local_para = [torch.tensor([max(cluster_time)]).to(device)]
                local_para.extend(fl_util.update(para_bak, agg_grads, args.lr))
                com.send_data(local_para, 0)

                global_para = com.recv_data(head, client.model.get_full_num_paras())
                global_para = fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU)
                client.model.set_full_paras(global_para)
                for c in groups:
                    if c != args.rank and c != args.num_clients:
                        com.send_data(global_para, c)
            else:
                local_para = [torch.tensor([client.delay-1+random.random()*2]).to(device)]
                local_para.extend(client.model.get_full_paras())
                com.send_data(local_para, head)
                global_para = com.recv_data(head, client.model.get_full_num_paras())
                client.model.set_full_paras(fl_util.reshape_para(global_para, client.model.get_full_para_shapes(), args.GPU))

def main():
    client = initialize()
    print("startle the experiments >>>>>>>>>>>>>>>>>>>>>>>\n")
    if args.method == 0:
        fedavg(client)
    elif args.method == 1:
        retrain(client)
    elif args.method == 2:
        federaser(client)
    elif args.method == 3:
        knot(client)
    elif args.method == 4:
        hier_fun(client)
    elif args.method == 5:
        random_(client)
main()