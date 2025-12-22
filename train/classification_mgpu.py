# coding=utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class SimpleNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super().__init__()
        self.linear_0 = nn.Linear(input_channel, input_channel)
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(input_channel, num_classes)

    def forward(self, x):
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        return x


def main():
    torch.manual_seed(42)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model = SimpleNet(10, 2).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 100
    batch_size = 5

    for epoch in range(num_epochs):
        input_data = torch.randn(batch_size, 10, device=device)
        labels = torch.randint(0, 2, (batch_size,), device=device)

        y = model(input_data)
        loss = loss_fn(y, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 and rank == 0:
            with torch.no_grad():
                predictions = torch.argmax(y, dim=1)
                accuracy = (predictions == labels).float().mean()
            print(
                f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}"
            )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
