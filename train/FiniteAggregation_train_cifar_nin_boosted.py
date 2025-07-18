from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
sys.path.append('../FeatureLearningRotNet/architectures')

from NetworkInNetwork import NetworkInNetwork
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy
import random

import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--k', default = 50, type=int, help='the inverse of sensitivity')
parser.add_argument('--d', default = 1, type=int, help='number of duplicates per sample')

parser.add_argument('--start', required=True, type=int, help='starting subset number')
parser.add_argument('--range', default=250, type=int, help='number of subsets to train')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')

args = parser.parse_args()

args.n_subsets = args.k * args.d
args.dataset='cifar'

print(f'zero_seed: {args.zero_seed} should be zero.')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dirbase = 'cifar_nin_baseline'
if (args.zero_seed):
    dirbase += '_zero_seed'

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_subdir = f'./{checkpoint_dir}/' + dirbase + f'_FiniteAggregation_k{args.k}_d{args.d}_boosted'
if not os.path.exists(checkpoint_subdir):
    os.makedirs(checkpoint_subdir, exist_ok=True)
print("==> Checkpoint directory", checkpoint_subdir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

# load partition file from dpa with k=args.k
partitions_file = torch.load("FiniteAggregation_hash_mean_" +args.dataset+'_k'+str(args.k)+'_d1.pth', weights_only=False)
partitions = partitions_file['idx']
means = partitions_file['mean']
stds = partitions_file['std']

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

avg_time = 0

random.seed(999999999+208)
seeds = random.sample(range(1000000), args.d) 
for part in range(args.start, args.start + args.range):
    # train partition on d different seeds
    for i, seed in enumerate(seeds):
        # seed = part
        if (args.zero_seed):
            seed = 0
            
        print(f'(part, seed): {part, seed}')
        
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        curr_lr = 0.1
        print(f'\Training partition: {part} ({i}) on partition {part}')
        part_indices = torch.tensor(partitions[part])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means[part], stds[part])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means[part], stds[part])
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        nomtestloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)
        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,part_indices), batch_size=128, shuffle=True, num_workers=1)
        net  = NetworkInNetwork({'num_classes':10})
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(net.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0005, nesterov= True)

        st_time = time.time()
    # Training
        net.train()
        for epoch in range(200):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if (epoch in [60,120,160]):
                curr_lr = curr_lr * 0.2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr
        en_time = time.time()

        print(f'time to train part {part}: {en_time - st_time}')
        avg_time += en_time - st_time

        net.eval()

        (inputs, targets)  = next(iter(nomtestloader)) #Just use one test batch
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
                #breakpoint()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            total = targets.size(0)
        acc = 100.*correct/total
        print('Accuracy: '+ str(acc)+'%') 
        # Save checkpoint.
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'partition': part,
            'norm_mean' : means[part],
            'norm_std' : stds[part]
        }
        model_num = part * args.d + i
        torch.save(state, checkpoint_subdir + '/FiniteAggregation_'+ str(model_num) + '.pth')

print(f'avg time: {avg_time / args.range} to train a single model ...')
print('done ...')
