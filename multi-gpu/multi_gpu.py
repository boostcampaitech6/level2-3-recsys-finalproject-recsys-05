import os
import time
import torch
import argparse
import torch.distributed as dist

# for dataset
from torch.utils.data.distributed import DistributedSampler

# for model
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument("--local_rank", type=int,
    #                     help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./save')
    parser.add_argument('--save_file_name', type=str, default='vgg_cifar')
    return parser


def main(opts):
	# 1. set random seeds
    set_random_seeds(random_seed=0)
	
    # 2. initialization
    init_for_distributed(opts)

    # 4. data set
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 데이터를 훈련 세트와 테스트 세트로 분리
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=int(opts.batch_size / opts.world_size),
                              shuffle=False,
                              num_workers=int(opts.num_workers / opts.world_size),
                              sampler=train_sampler,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=int(opts.batch_size / opts.world_size),
                             shuffle=False,
                             num_workers=int(opts.num_workers / opts.world_size),
                             sampler=test_sampler,
                             pin_memory=True)

    # 5. model
    # 2. 모델 정의
    class IrisNet(nn.Module):
        def __init__(self):
            super(IrisNet, self).__init__()
            self.fc1 = nn.Linear(4, 100)
            self.fc2 = nn.Linear(100, 3)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    
    model = IrisNet()
    model = model.cuda(opts.local_rank)
    model = DDP(module=model,
                device_ids=[opts.local_rank])

    # 6. criterion
    criterion = nn.CrossEntropyLoss().to(opts.local_rank)

    # 7. optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    # 4. 모델 학습
    for epoch in range(opts.epoch):
        for inputs, labels in train_loader:
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{opts.epoch}], Loss: {loss.item():.4f}')

    end_time = time.time()  # 종료 시간 측정
    print(f"Execution time: {end_time - start_time} seconds")

    # 5. 모델 평가
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            labels = labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total}%')


def init_for_distributed(opts):

    # 1. setting for distributed training
    opts.global_rank = int(os.environ['RANK'])
    opts.local_rank = int(os.environ['LOCAL_RANK'])
    opts.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(opts.local_rank)
    if opts.global_rank is not None and opts.local_rank is not None:
        print("Use GPU: [{}/{}] for training".format(opts.global_rank, opts.local_rank))

    # 2. init_process_group
    dist.init_process_group(backend="nccl")
    # if put this function, the all processes block at all.
    # torch.distributed.barrier()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser('test', parents=[get_args_parser()])
    opts = parser.parse_args()
    main(opts)