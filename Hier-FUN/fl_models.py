import torch
import torch.nn as nn
import torch.nn.functional as F

class Base_Model():
    def get_num_paras(self):
        return sum([p.numel() for p in self.parameters()])

    def get_full_num_paras(self):
        para_dict = self.state_dict()
        return sum([para_dict[n].numel() for n in para_dict.keys()])
        
    def set_paras(self, new_para):
        count_paras = 0
        paras = self.named_parameters()
        para_dict = dict(paras)

        with torch.no_grad():
            for n,_ in para_dict.items():
                # if 'bn' not in n:
                para_dict[n].set_(new_para[count_paras].clone().detach().float().to(para_dict[n].device))
                count_paras += 1

    def set_full_paras(self, new_para):
        count = 0
        para_dict = self.state_dict()
        with torch.no_grad():
            for n in para_dict.keys():
                para_dict[n].data = new_para[count].clone().detach().to(para_dict[n].device)
                count += 1
        self.load_state_dict(para_dict)

    def get_grads(self):
        return [p[1].grad.clone().detach() for p in self.named_parameters()]
    
    def get_paras(self):
        return [p[1].data.clone().detach() for p in self.named_parameters()]
    
    def get_para_shapes(self):
        return [p[1].shape for p in self.named_parameters()]
    
    def get_full_paras(self):
        para_dict = self.state_dict()
        return [para_dict[n].clone().detach() for n in para_dict.keys()]

    def get_full_para_shapes(self):
        para_dict = self.state_dict()
        return [para_dict[n].data.shape for n in para_dict.keys()]

    def get_para_names(self):        
        self_para_name = []
        for p in self.named_parameters():
            if p[1].requires_grad:
                self_para_name.append(p[0])
        return self_para_name

## Multi Layer Perception for Fashion Mnist dataset
class FMNIST_MLP(nn.Module, Base_Model):
    def __init__(self):
        super(FMNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 100)
        self.fc5 = nn.Linear(100, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

## AlextNet for CIFAR-10 dataset
class CIFAR10_AlexNet(nn.Module, Base_Model):
    def __init__(self):
        super(CIFAR10_AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),nn.BatchNorm2d(96),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),stride=1,padding=2),nn.ReLU(),nn.BatchNorm2d(256),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),nn.BatchNorm2d(384),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),nn.BatchNorm2d(384),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding=1),nn.ReLU(),nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.layer1(x)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)

## ReNet-18 for CIFAR-100 dataset
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
        nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True),
        nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
## For CIFAR datasets
class CIFAR100_ResNet(nn.Module, Base_Model): 
    def __init__(self, ResidualBlock, num_blocks, num_classes=100):
        super(CIFAR100_ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

## create models according to the dataset type
def create_models(dataset, GPU=0):
    if torch.cuda.is_available():
        device = 'cuda:'+str(GPU)
    else:
        device = 'cpu'
    if dataset == 'fmnist':
        return FMNIST_MLP().to(device)
    if dataset == 'c10':
        return CIFAR10_AlexNet().to(device)
    if dataset == 'c100':
        return CIFAR100_ResNet(ResidualBlock, [2,2,2,2]).to(device)
    raise Exception('Specified dataset NotFound:', dataset)
