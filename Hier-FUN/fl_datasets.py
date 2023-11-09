import torch
import random
import fl_util
import numpy as np
from torchvision import datasets, transforms

## specify the path for dataset, such as
# dataset_path = r'xxx/zgma/dataset'
dataset_path = r'????'

def find_index(targets, labels):
    return [i for (i,v) in enumerate(targets) if v in labels]

## load train dataset for each device
def load_train_dataset(dataset, num_clients, ratio, rank):
    if dataset == 'fmnist':
        num_classes = 10
        samples_per_cls = 6000
        train_dataset = datasets.FashionMNIST(dataset_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
    elif dataset == 'c10':
        num_classes = 10
        samples_per_cls = 5000
        train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif dataset == 'c100':
        num_classes = 100
        samples_per_cls = 500
        train_dataset = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    else:
        raise Exception('Specified dataset NotFound:', dataset)

    ## generate the partition rate of each category on each device
    par_sizes = fl_util.gen_partiion_sizes(dataset, num_clients, ratio)

    train_labels = train_dataset.targets
    start_sample_idx = np.zeros([num_classes])
    for client in range(rank-1):
        start_sample_idx += par_sizes[client,:]
    end_sample_idx = start_sample_idx + par_sizes[rank-1,:]
    start_sample_idx = start_sample_idx * samples_per_cls
    end_sample_idx = end_sample_idx * samples_per_cls

    accu_samples = np.zeros([num_classes])
    indices = []
    for idx in range(len(train_labels)):
        accu_samples[train_labels[idx]] += 1
        if start_sample_idx[train_labels[idx]] < accu_samples[train_labels[idx]] <= end_sample_idx[train_labels[idx]]:
            indices.append(idx)
    
    train_data, train_label = zip(*([train_dataset[j] for j in indices]))
    print('train set size:%d'%len(train_label), [train_label.count(j) for j in range(num_classes)])
    return torch.stack(train_data), torch.tensor(train_label)

## load test dataset for the server
def load_test_dataset(args):
    if args.dataset== 'fmnist':
        test_dataset = datasets.FashionMNIST(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
    elif args.dataset== 'c10':
        test_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif args.dataset== 'c100':
        test_dataset = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    else:
        raise Exception('Specified dataset NotFound:', args.dataset)
    
    test_labels = test_dataset.targets
    indices = [i for i in range(len(test_labels))]
    test_data, test_label = zip(*([test_dataset[j] for j in indices]))
    return torch.stack(test_data), torch.tensor(test_label)

## load the test dataset with target data distribution
def load_target_test_dataset(args):
    if args.dataset== 'fmnist':
        select_labels = [8,9]
        test_dataset = datasets.FashionMNIST(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
    elif args.dataset== 'c10':
        select_labels = [8,9]
        test_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
    elif args.dataset== 'c100':
        select_labels = [95,96,97,98,99]
        test_dataset = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
    else:
        raise Exception('Specified dataset NotFound:', args.dataset)
    
    test_labels = test_dataset.targets
    target_test_indices = find_index(test_labels, select_labels)

    target_test_data, target_test_label = zip(*([test_dataset[j] for j in target_test_indices]))
    return torch.stack(target_test_data), torch.tensor(target_test_label)

## load the test dataset with the data distributions except for the target dveice
def load_retain_test_dataset(args):
    if args.dataset== 'fmnist':
        select_labels = [0,1,2,3,4,5,6,7]
        test_dataset = datasets.FashionMNIST(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
    elif args.dataset== 'c10':
        select_labels = [0,1,2,3,4,5,6,7]
        test_dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif args.dataset== 'c100':
        select_labels = [i for i in range(95)]
        test_dataset = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    else:
        raise Exception('Specified dataset NotFound:', args.dataset)
    
    test_labels = test_dataset.targets
    target_test_indices = find_index(test_labels, select_labels)

    target_test_data, target_test_label = zip(*([test_dataset[j] for j in target_test_indices]))
    return torch.stack(target_test_data), torch.tensor(target_test_label)