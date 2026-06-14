import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    # 初始化分布式环境
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if rank == 0:
        print("正在下载/加载 MNIST 数据集...")

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, sampler=train_sampler)

    model = SimpleCNN().to(device)
    model = DDP(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if rank == 0:
        print(f"开始分布式训练，使用 {world_size} 个进程...")

    start_time = time.time()
    for epoch in range(1, 4):
        train_sampler.set_epoch(epoch)
        train(model, device, train_loader, optimizer, epoch)
        if rank == 0:
            print(f"Epoch {epoch} 完成")
    end_time = time.time()

    if rank == 0:
        print(f"分布式版 ({world_size} 进程) 训练时间: {end_time - start_time:.2f} 秒")